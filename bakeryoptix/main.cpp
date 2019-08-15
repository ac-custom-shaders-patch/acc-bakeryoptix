/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define USE_TRYCATCH

#include <cassert>
#include <cstdio>
#include <iostream>
#include <utility>
#include <vector>
#include <Windows.h>

#include <utils/std_ext.h>
#include <utils/filesystem.h>
#include <utils/perf_moment.h>
#include <utils/load_util.h>
#include <utils/vector_operations.h>

#include <bake_api.h>
#include <bake_util.h>
#include <utils/ini_file.h>
#include <cuda_runtime_api.h>
#include <set>
#include <bake_wrap.h>
#include <fstream>

#pragma comment(lib, "Shlwapi.lib")
#pragma comment(lib, "WinMM.lib")
#pragma comment(lib, "cudart_static.lib")
#pragma comment(lib, "optix_prime.6.0.0.lib")

struct mixing_params
{
	bool use_max;
	float max_mult;
};

bool is_car(const utils::path& filename)
{
	auto parent = filename.parent_path().string();
	std::transform(parent.begin(), parent.end(), parent.begin(), tolower);
	std_ext::replace(parent, "\\", "/");
	return parent.find("/cars/") != std::string::npos;
}

utils::path get_executable_filename()
{
	wchar_t result[1024];
	GetModuleFileNameW(nullptr, result, 1024);
	return result;
}

utils::ini_file load_config(const utils::path& filename)
{
	auto config = filename.parent_path() / "baked_shadow_params.ini";
	if (!exists(config)) config = filename.parent_path() / "baked shadow params.ini";
	if (!exists(config)) config = get_executable_filename().parent_path() / "baked_shadow_params.ini";
	if (!exists(config)) config = get_executable_filename().parent_path() / "baked shadow params.ini";
	return utils::ini_file{config};
}

void add_materials(std::set<std::string>& dest, const std::vector<std::shared_ptr<bake::Mesh>>& meshes)
{
	for (const auto& m : meshes)
	{
		if (m->material->shader == "ksTyres" || m->material->shader == "ksBrakeDisc") continue;
		dest.insert(m->material->name);
	}
}

void add_materials(std::set<std::string>& dest,
	const std::vector<std::shared_ptr<bake::Node>>& include,
	const std::vector<std::shared_ptr<bake::Node>>& except)
{
	std::vector<std::shared_ptr<bake::Mesh>> meshes;
	for (const auto& i : include)
	{
		if (!i) continue;
		meshes += i->get_meshes();
	}
	for (const auto& e : except)
	{
		if (!e) continue;
		meshes -= e->get_meshes();
	}

	add_materials(dest, meshes);
}

template <typename T>
std::vector<T> to_vector(const std::set<T>& input)
{
	std::vector<T> r;
	std::copy(input.begin(), input.end(), std::back_inserter(r));
	return r;
}

void fill_animation(const std::shared_ptr<bake::Node>& moving_root, utils::ini_file& config, const std::shared_ptr<bake::Animation>& animation, const std::string& file)
{
	animation->init(moving_root);
	const auto node_names = where(
		apply(animation->entries, [](const bake::NodeTransition& t) { return t.node ? t.node->name : ""; }),
		[](const std::string& t) { return !t.empty(); });
	std::cout << "Animation `" << file << "` affects nodes: " << std_ext::join_to_string(node_names, ", ") << "\n";

	auto name_lc = utils::path(file).filename().string();
	std::transform(name_lc.begin(), name_lc.end(), name_lc.begin(), tolower);

	std::string key;
	if (name_lc == "lights.ksanim")
	{
		key = "HEADLIGHTS_NODES";
	}
	else if (name_lc == "car_door_l.ksanim" || name_lc == "car_door_r.ksanim")
	{
		key = "DOOR_NODES";
	}
	else
	{
		for (auto i = 0; i < 100; i++)
		{
			auto v = config.get("SPLIT_AO", "WING_ANIM_" + std::to_string(i) + "_NAME", std::string());
			std::transform(v.begin(), v.end(), v.begin(), tolower);
			if (v == name_lc)
			{
				key = "WING_ANIM_" + std::to_string(i) + "_NODES";
				break;
			}
		}
	}

	if (!key.empty())
	{
		std::vector<std::string> list_new;
		for (const auto& item : config.get("SPLIT_AO", key, std::vector<std::string>()))
		{
			if (item == "@AUTO")
			{
				list_new |= node_names;
			}
			else
			{
				list_new.push_back(item);
			}
		}

		if (!list_new.empty())
		{
			config.set("SPLIT_AO", key, list_new);
		}
	}
}

void dump_obj(const utils::path& filename, const std::vector<std::shared_ptr<bake::Mesh>>& meshes)
{
	std::ofstream o(filename.string());
	std::vector<int> start;
	start.resize(meshes.size() + 1);
	start[0] = 1;

	for (auto j = 0U; j < meshes.size(); j++)
	{
		const auto& m = meshes[j];
		const auto& x = (*(const optix::Matrix4x4*)&m->matrix).transpose();
		float4 f4;
		f4.w = 1.f;
		for (const auto& vertice : m->vertices)
		{
			*(bake::Vec3*)&f4 = vertice;
			const auto w = f4 * x;
			o << "v " << w.x << " " << w.y << " " << w.z << std::endl;
		}
		start[j + 1] = start[j] + m->vertices.size();
	}
	for (const auto& m : meshes)
	{
		const auto& x = (*(const optix::Matrix4x4*)&m->matrix).transpose();
		float4 f4;
		f4.w = 0.f;
		for (const auto& normal : m->normals)
		{
			*(bake::Vec3*)&f4 = normal;
			const auto w = f4 * x;
			o << "vn " << w.x << " " << w.y << " " << w.z << std::endl;
		}
	}
	for (auto j = 0U; j < meshes.size(); j++)
	{
		const auto& m = meshes[j];
		o << "g " << m->name << std::endl;
		for (const auto& triangle : m->triangles)
		{
			const auto tr = int3{start[j], start[j], start[j]} + *(int3*)&triangle;
			o << "f " << tr.x << "//" << tr.x << " " << tr.y << "//" << tr.y << " " << tr.z << "//" << tr.z << std::endl;
		}
	}
}

struct file_processor
{
	#define FEATURE_ACTIVE(N) config.get(section, "EXT_" N, false)

	utils::path input_file;
	utils::path resulting_name;
	utils::ini_file config;
	std::string section;

	load_params cfg_load{};
	bake_params cfg_bake{};
	save_params cfg_save{};

	std::vector<std::string> rotating_x;
	std::vector<std::string> rotating_y;
	std::vector<std::string> rotating_z;
	float rotating_step = 30.f;
	mixing_params rotating_mixing{};

	std::vector<std::string> animations;
	std::vector<float> animations_steps;
	mixing_params animations_mixing{};
	std::vector<std::shared_ptr<bake::Mesh>> animations_mixing_inverse;
	std::vector<std::shared_ptr<bake::Animation>> animation_instances;

	float interior_mult = 1.f;
	float exterior_mult = 1.f;
	float driver_mult = 1.f;
	std::set<std::string> interior_materials;
	std::set<std::string> exterior_materials;
	std::set<std::string> driver_materials;

	std::shared_ptr<bake::Node> root;
	std::shared_ptr<bake::Node> driver_root;
	std::shared_ptr<bake::Node> driver_root_lodb;
	std::shared_ptr<bake::Scene> driver_root_scene;
	std::vector<std::shared_ptr<bake::Animation>> driver_steer_animations;
	std::vector<std::shared_ptr<bake::Animation>> driver_animations;
	std::vector<float> driver_animations_steps;
	std::vector<float> driver_animations_steer_steps;
	mixing_params driver_animations_mixing{};

	struct
	{
		std::vector<std::string> blurred_names;
		std::vector<std::string> static_names;
		bool ready{};

		void ensure_ready(const utils::path& filename)
		{
			if (ready) return;
			ready = true;
			const auto data_blurred = utils::ini_file(filename.parent_path() / "data" / "blurred_objects.ini");
			for (const auto& s : data_blurred.iterate_break("OBJECT"))
			{
				(data_blurred.get(s, "MIN_SPEED", 0.f) == 0.f ? static_names : blurred_names)
					.push_back(data_blurred.get(s, "NAME", std::string()));
			}
		}

		bool any(const utils::path& filename)
		{
			ensure_ready(filename);
			return !blurred_names.empty();
		}
	} blurred_objects;

	const std::vector<std::string> names_hr{"COCKPIT_HR", "STEER_HR"};
	const std::vector<std::string> names_lr{"COCKPIT_LR", "STEER_LR"};

	std::vector<std::string> bake_as_trees;
	std::vector<std::string> bake_as_grass;

	struct reduced_ground
	{
		std::vector<std::shared_ptr<bake::Mesh>> meshes;
		float factor;
	};
	std::vector<reduced_ground> reduced_ground_params;

	file_processor(utils::path filename, bool is_cuda_available)
		: input_file(std::move(filename))
	{
		config = load_config(input_file);
		resulting_name = input_file.filename_without_extension() + ".vao-patch";

		auto mode = config.get("GENERAL", "MODE", std::string("AUTO"));
		if (mode == "AUTO")
		{
			mode = is_car(input_file) ? "CAR" : "TRACK";
		}

		// Specialized options
		section = "MODE_" + mode;

		// Override for certain cars or tracks
		auto id = input_file.parent_path().filename().string();
		for (const auto& s : config.iterate_nobreak("SPECIFIC"))
		{
			if (std_ext::match(config.get(s, mode, std::string()), id))
			{
				const auto desc = config.get(s, "DESCRIPTION", std::string());
				std::cout << "Found spec. options" << (desc.empty() ? " for `" + id + "`" : ": " + desc) << "\n";
				for (const auto& p : config.sections[s])
				{
					auto i = p.first.find("__");
					if (i == std::string::npos)
					{
						config.sections[section][p.first] = p.second;
					}
					else
					{
						config.sections[p.first.substr(0, i)][p.first.substr(i + 2)] = p.second;
					}
				}
			}
		}

		// General options
		cfg_bake.use_cuda = config.get("GENERAL", "CPU_ONLY", std::string("AUTO")) == "AUTO" ? is_cuda_available : config.get("GENERAL", "CPU_ONLY", false);
		cfg_load.exclude_patch = config.get(section, "EXCLUDE_FROM_BAKING", std::vector<std::string>());
		cfg_load.exclude_blockers = config.get(section, "FORCE_PASSING_LIGHT", std::vector<std::string>());
		cfg_load.exclude_blockers_alpha_test = config.get(section, "FORCE_ALPHA_TEST_PASSING_LIGHT", false);
		cfg_load.normals_bias = config.get(section, "NORMALS_Y_BIAS", 0.f);

		cfg_bake.num_samples = config.get(section, "SAMPLES", std::string("AUTO")) == "AUTO" ? 0 : config.get(section, "SAMPLES", 0);
		cfg_bake.min_samples_per_face = config.get(section, "MIN_SAMPLES_PER_FACE", 4);
		cfg_bake.num_rays = config.get(section, "RAYS_PER_SAMPLE", 64);
		cfg_bake.scene_offset_scale = config.get(section, "RAYS_OFFSET", 0.1f);
		cfg_bake.use_ground_plane_blocker = config.get(section, "USE_GROUND", false);
		cfg_bake.ground_scale_factor = config.get(section, "GROUND_SCALE", 10.f);
		cfg_bake.scene_maxdistance_scale = config.get(section, "SCENE_MAX_DISTANCE_SCALE", 10.f);
		cfg_bake.ground_upaxis = config.get(section, "GROUND_UP_AXIS", 1);
		cfg_bake.ground_offset_factor = config.get(section, "GROUND_OFFSET", 0.f);
		cfg_bake.filter_mode = config.get(section, "FILTER_MODE", std::string("LEAST_SQUARES")) == "LEAST_SQUARES"
			? bake::VERTEX_FILTER_LEAST_SQUARES
			: bake::VERTEX_FILTER_AREA_BASED;
		cfg_bake.regularization_weight = config.get(section, "FILTER_REGULARIZATION_WEIGHT", 0.1f);

		cfg_save.averaging_threshold = config.get(section, "AVERAGING_THRESHOLD", 0.f);
		cfg_save.averaging_cos_threshold = config.get(section, "AVERAGING_COS_THRESHOLD", 0.95f);
		cfg_save.brightness = config.get(section, "BRIGHTNESS", 1.02f);
		cfg_save.gamma = config.get(section, "GAMMA", 0.92f);
		cfg_save.opacity = config.get(section, "OPACITY", 0.97f);

		for (const auto& s : config.sections)
		{
			if (s.first != "GENERAL" && s.first.find("MODE_") != 0)
			{
				cfg_save.extra_config.sections[s.first] = s.second;
			}
		}

		// Baking flow options
		rotating_x = config.get(section, "ROTATING_X", std::vector<std::string>());
		rotating_y = config.get(section, "ROTATING_Y", std::vector<std::string>());
		rotating_z = config.get(section, "ROTATING_Z", std::vector<std::string>());
		rotating_step = config.get(section, "ROTATING_STEP", 30.f);
		rotating_mixing = mixing_params{
			config.get(section, "ROTATING_MIXING", std::string("MAX")) == "MAX",
			config.get(section, "ROTATING_MIXING_MULT", 1.f)
		};

		animations = config.get(section, "ANIMATIONS", std::vector<std::string>());
		animations_steps = config.get(section, "ANIMATIONS_STEPS", std::vector<float>());
		animations_mixing = mixing_params{
			config.get(section, "ANIMATIONS_MIXING", std::string("MAX")) == "MAX",
			config.get(section, "ANIMATIONS_MIXING_MULT", 0.8f)
		};

		if (FEATURE_ACTIVE("USE_RIGHT_DOOR_ANIM_IF_NEEDED")
			&& utils::ini_file(input_file.parent_path() / "data" / "car.ini").get("GRAPHICS", "DRIVEREYES", 0.f) < 0.f)
		{
			animations -= {"car_door_L.ksanim", "car_door_l.ksanim"};
			animations.emplace_back("car_door_R.ksanim");
		}

		interior_mult = config.get(section, "BRIGHTER_AMBIENT_INTERIOR", 1.f);
		exterior_mult = config.get(section, "BRIGHTER_AMBIENT_EXTERIOR", 1.f);
		driver_mult = config.get(section, "BRIGHTER_AMBIENT_DRIVER", 1.f);

		bake_as_trees = config.get(section, "BAKE_AS_TREES", std::vector<std::string>());
		bake_as_grass = config.get(section, "BAKE_AS_GRASS", std::vector<std::string>());
		cfg_load.exclude_blockers += bake_as_grass;

		// Actual baking starts here, with loading of model
		{
			PERF("Loading model `" + input_file.relative_ac() + "`");
			root = load_model(input_file, cfg_load);
		}

		// Print scene stats
		auto meshes = root->get_meshes();
		{
			size_t num_vertices = 0;
			size_t num_triangles = 0;
			for (const auto& mesh : meshes)
			{
				num_vertices += mesh->vertices.size();
				num_triangles += mesh->triangles.size();
			}
			std::cout << "\t" << meshes.size() << " meshes" << std::endl;
			std::cout << "\t" << num_vertices / 1000 << "K vertices, " << num_triangles / 1000 << "K triangles" << std::endl;
		}

		// Loading extra offsets
		for (auto i = 0; i < 100; i++)
		{
			std::vector<std::string> names;
			float offset;
			if (config.try_get(section, "EXTRA_OFFSET_" + std::to_string(i) + "_NAMES", names)
				&& config.try_get(section, "EXTRA_OFFSET_" + std::to_string(i) + "_VALUE", offset)
				&& offset != 0.f)
			{
				for (const auto& mesh : root->find_meshes(names))
				{
					mesh->extra_samples_offset = offset;
				}
			}
		}

		// Resetting some objects just in case
		if (FEATURE_ACTIVE("RESET_BLURRED") && blurred_objects.any(input_file))
		{
			root->set_active(blurred_objects.blurred_names, false);
			root->set_active(blurred_objects.static_names, true);
		}

		if (FEATURE_ACTIVE("HIDE_SEATBELT"))
		{
			root->set_active({"CINTURE_ON"}, false);
			root->set_active({"CINTURE_OFF"}, false);
		}

		// Adding driver
		if (FEATURE_ACTIVE("DRIVER_CASTS_SHADOWS") || FEATURE_ACTIVE("DRIVER_RECEIVES_SHADOWS"))
		{
			const auto data_driver3d = utils::ini_file(input_file.parent_path() / "data" / "driver3d.ini");
			const auto driver_kn5 = input_file.parent_path().parent_path().parent_path() / "driver"
				/ (data_driver3d.get("MODEL", "NAME", std::string("driver")) + ".kn5");
			const auto driver_pos = input_file.parent_path() / "driver_base_pos.knh";
			if (exists(driver_kn5) && exists(driver_pos))
			{
				driver_root = load_model(driver_kn5, cfg_load);

				const auto hierarcy = load_hierarchy(driver_pos);
				hierarcy->align(driver_root);

				std::shared_ptr<bake::Animation> steer_anim;
				if (FEATURE_ACTIVE("DRIVER_POSITION_WITH_STEER_ANIM"))
				{
					steer_anim = load_ksanim(input_file.parent_path() / "animations" / "steer.ksanim", true);
					steer_anim->apply(driver_root, 0.5f);
				}

				driver_root_scene = std::make_shared<bake::Scene>(driver_root);

				if (FEATURE_ACTIVE("DRIVER_INCLUDE_LOD_B"))
				{
					auto driver_lodb = driver_kn5.parent_path() / driver_kn5.filename_without_extension() + "_B.kn5";
					if (exists(driver_lodb))
					{
						driver_root_lodb = load_model(driver_lodb, cfg_load);
						hierarcy->align(driver_root_lodb);
						if (steer_anim)
						{
							steer_anim->apply(driver_root_lodb, 0.5f);
						}
					}
				}

				if (FEATURE_ACTIVE("DRIVER_BAKE_ANIMS"))
				{
					const auto anims = config.get(section, "DRIVER_ANIMATIONS", std::vector<std::string>());
					for (const auto& name : anims)
					{
						const auto anim_filename = input_file.parent_path() / "animations" / name;
						if (exists(anim_filename))
						{
							(name == "steer.ksanim" ? driver_steer_animations : driver_animations).push_back(load_ksanim(anim_filename));
						}
					}

					driver_animations_steps = config.get(section, "DRIVER_ANIMATIONS_STEPS", std::vector<float>());
					driver_animations_steer_steps = config.get(section, "DRIVER_ANIMATIONS_STEER_STEPS", std::vector<float>());
					driver_animations_mixing = mixing_params{
						config.get(section, "DRIVER_ANIMATIONS_MIXING", std::string("MAX")) == "MAX",
						config.get(section, "DRIVER_ANIMATIONS_MIXING_MULT", 0.8f)
					};
				}
			}
		}

		// Loading ground adjustments
		if (cfg_bake.use_ground_plane_blocker)
		{
			for (auto i = 0; i < 100; i++)
			{
				std::vector<std::string> names;
				float factor;
				if (config.try_get(section, "REDUCED_GROUND_" + std::to_string(i) + "_NAMES", names)
					&& config.try_get(section, "REDUCED_GROUND_" + std::to_string(i) + "_FACTOR", factor)
					&& factor != 0.f)
				{
					auto list = root->find_any_meshes(names);
					if (driver_root != nullptr)
					{
						list += driver_root->find_any_meshes(names);
					}
					reduced_ground_params.push_back({list, factor});
				}
			}
		}

		// Getting list of materials if needed
		if (exterior_mult != 1.f)
		{
			add_materials(exterior_materials, {root}, root->find_nodes(names_hr + names_lr));
		}
		if (interior_mult != 1.f)
		{
			add_materials(interior_materials, root->find_nodes(names_hr + names_lr), {});
		}
		if (driver_mult != 1.f && driver_root != nullptr)
		{
			add_materials(driver_materials, {driver_root}, {});
		}

		// Preparing animations
		for (const auto& file : animations)
		{
			auto loaded = load_ksanim(input_file.parent_path() / "animations" / file);
			if (!loaded->entries.empty())
			{
				animation_instances.push_back(loaded);
				fill_animation(root, cfg_save.extra_config, loaded, file);
			}
		}

		if (config.get(section, "ANIMATIONS_LOAD_WINGS", false))
		{
			const auto data_wing_animations = utils::ini_file(input_file.parent_path() / "data" / "wing_animations.ini");
			for (const auto& s : data_wing_animations.iterate_break("ANIMATION"))
			{
				const auto file = data_wing_animations.get(s, "FILE", std::string());
				auto loaded = load_ksanim(input_file.parent_path() / "animations" / file);
				if (!loaded->entries.empty())
				{
					animation_instances.push_back(loaded);
					fill_animation(root, cfg_save.extra_config, loaded, file);
				}
			}
		}

		const auto animations_mixing_inverse_names = config.get(section, "ANIMATIONS_MIXING_INVERSE", std::vector<std::string>());
		animations_mixing_inverse = root->find_meshes(animations_mixing_inverse_names);
		for (const auto& n : root->find_nodes(animations_mixing_inverse_names))
		{
			animations_mixing_inverse += n->get_meshes();
		}
	}

	baked_data bake_scene(const std::shared_ptr<bake::Scene>& scene, const bake_params& config, bool verbose = false)
	{
		return bake_scene(scene, scene->blockers, config, verbose);
	}

	baked_data bake_scene(const std::shared_ptr<bake::Scene>& scene, const std::vector<std::shared_ptr<bake::Mesh>>& blockers,
		const bake_params& config, bool verbose = false)
	{
		if (!reduced_ground_params.empty())
		{
			auto scene_adj = std::make_shared<bake::Scene>(*scene);
			for (const auto& r : reduced_ground_params)
			{
				if (r.factor == 1.f)
				{
					scene_adj->receivers -= r.meshes;
				}
			}

			auto result = bake_wrap::bake_scene(scene_adj, blockers, config, verbose);
			auto config_noground = config;
			config_noground.use_ground_plane_blocker = false;

			for (const auto& r : reduced_ground_params)
			{
				scene_adj->receivers = scene->receivers & r.meshes;
				if (!scene_adj->receivers.empty())
				{
					result.max(bake_wrap::bake_scene(scene_adj, blockers, config_noground, verbose), r.factor, {}, true);
				}
			}

			return result;
		}
		else
		{
			return bake_wrap::bake_scene(scene, blockers, config, verbose);
		}
	}

	void run()
	{
		// Optional dumping
		std::string dump_as;
		if (config.try_get(section, "DUMP_INSTEAD_OF_BAKING", dump_as) && !dump_as.empty())
		{
			bake::Animation::apply_all(root, {
				load_ksanim(input_file.parent_path() / "animations" / "car_DOOR_L.ksanim"),
				load_ksanim(input_file.parent_path() / "animations" / "car_DOOR_R.ksanim"),
			}, 1.f);
			driver_root->update_matrix();
			driver_root->resolve_skinned();
			dump_obj(dump_as, root->get_meshes() + driver_root->get_meshes());
			return;
		}

		auto ao = bake_stuff(root);

		// Generating alternative AO with driver shadows
		if (FEATURE_ACTIVE("DRIVER_CASTS_SHADOWS"))
		{
			const auto driver_shadow_targets = root->find_nodes({"COCKPIT_HR", "COCKPIT_LR"});
			const auto driver_shadow_blockers = std::make_shared<bake::Scene>(root)->blockers + std::make_shared<bake::Scene>(driver_root)->blockers;
			const auto root_scene = std::make_shared<bake::Scene>(driver_shadow_targets);
			for (const auto& n : root->find_nodes({"PAD_UP", "PAD_DOWN", "STEER_HR", "STEER_LR"}))
			{
				root_scene->receivers -= n->get_meshes();
			}
			auto ao_alt = bake_scene(root_scene, driver_shadow_blockers, cfg_bake, true);
			bake_animations(root, driver_shadow_targets, driver_shadow_blockers, cfg_bake, ao_alt);
			ao.set_alternative_set(ao_alt);
		}

		// Saving result
		{
			if (!exterior_materials.empty())
			{
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_EXT", "TAGS", std::string("VAO_PATCH"));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_EXT", "MATERIALS", to_vector(exterior_materials));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_EXT", "PROP_0", std::vector<std::string>{"ksAmbient", "*" + std::to_string(exterior_mult)});
			}

			if (!interior_materials.empty())
			{
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_INT", "TAGS", std::string("VAO_PATCH"));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_INT", "MATERIALS", to_vector(interior_materials));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_INT", "PROP_0", std::vector<std::string>{"ksAmbient", "*" + std::to_string(interior_mult)});
			}

			if (!driver_materials.empty())
			{
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_DRIVER", "TAGS", std::string("VAO_PATCH"));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_DRIVER", "MATERIALS", to_vector(driver_materials));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_DRIVER", "PROP_0", std::vector<std::string>{"ksAmbient", "*" + std::to_string(driver_mult)});
			}

			const auto destination = input_file.parent_path() / resulting_name;
			PERF("Saving to `" + destination.relative_ac() + "`");
			ao.save(destination, cfg_save, FEATURE_ACTIVE("SPLIT_AO"));
		}
	}

	void bake_animations(const std::shared_ptr<bake::Node>& moving_root, const std::vector<std::shared_ptr<bake::Node>>& targets,
		const std::vector<std::shared_ptr<bake::Mesh>>& blockers, const bake_params& cfg_bake, baked_data& ao, const std::string& comment = "",
		bool verbose = true, bool apply_to_both_sets = false)
	{
		bake_animations(moving_root, targets, blockers, animation_instances, animations_steps, animations_mixing, animations_mixing_inverse,
			cfg_bake, ao, comment, verbose, apply_to_both_sets);
	}

	void bake_animations(const std::shared_ptr<bake::Node>& moving_root, const std::vector<std::shared_ptr<bake::Node>>& targets,
		const std::vector<std::shared_ptr<bake::Mesh>>& blockers,
		std::vector<std::shared_ptr<bake::Animation>>& animation_instances, const std::vector<float>& animations_steps, const mixing_params& animations_mixing,
		const std::vector<std::shared_ptr<bake::Mesh>>& animations_mixing_inverse, const bake_params& cfg_bake, baked_data& ao, const std::string& comment = "",
		bool verbose = true, bool apply_to_both_sets = false)
	{
		if (animation_instances.empty()) return;
		if (verbose)
		{
			std::cout << "Baking animations (" << animation_instances.size() << (comment.empty() ? " found):" : " found, " + comment + "):") << std::endl;
		}
		for (auto i = 0U; i < animations_steps.size(); i++)
		{
			const auto pos = animations_steps[i];
			if (verbose)
			{
				std::cout << (i == 0 ? "\t" : "\r\t") << "Position: " << round(pos * 100.f) << "%";
			}
			if (bake::Animation::apply_all(moving_root, animation_instances, pos))
			{
				const auto baked = bake_scene(std::make_shared<bake::Scene>(targets), blockers, cfg_bake);
				if (animations_mixing.use_max)
				{
					ao.max(baked, animations_mixing.max_mult, animations_mixing_inverse, apply_to_both_sets);
				}
				else
				{
					const auto avg_mult = 1.f / float(animations_steps.size() + 1);
					ao.average(baked, avg_mult, i == 0 ? avg_mult : 1.f, animations_mixing_inverse, apply_to_both_sets);
				}
			}
		}
		bake::Animation::apply_all(moving_root, animation_instances, 0.f);
		if (verbose)
		{
			std::cout << std::endl;
		}
	}

	void bake_rotating(const std::shared_ptr<bake::Node>& root, baked_data& ao, const bake_params& params, const mixing_params& mixing, const std::string& comment = "",
		const bool verbose = true)
	{
		const auto root_scene = std::make_shared<bake::Scene>(root);
		const auto nodes_x = root->find_nodes(rotating_x);
		const auto nodes_y = root->find_nodes(rotating_y);
		const auto nodes_z = root->find_nodes(rotating_z);
		const auto rotating = nodes_x + nodes_y + nodes_z;
		if (rotating.empty()) return;

		if (verbose)
		{
			std::cout << "Baking rotating objects (" << rotating.size() << (comment.empty() ? " found):" : " found, " + comment + "):") << std::endl;
		}

		const auto iterations = int(ceilf(359.f / rotating_step));
		for (auto i = 1; i < iterations; i++)
		{
			const auto deg = i * rotating_step;
			if (verbose)
			{
				std::cout << (i == 1 ? "\t" : "\r\t") << "Angle: " << deg << utf16_to_utf8(L"°");
			}
			for (const auto& n : nodes_x)
			{
				const auto axis = float3{1.f, 0.f, 0.f};
				n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(deg, &axis.x);
			}
			for (const auto& n : nodes_y)
			{
				const auto axis = float3{0.f, 1.f, 0.f};
				n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(deg, &axis.x);
			}
			for (const auto& n : nodes_z)
			{
				const auto axis = float3{0.f, 0.f, 1.f};
				n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(deg, &axis.x);
			}
			const auto baked = bake_scene(std::make_shared<bake::Scene>(rotating), root_scene->blockers, params);
			if (mixing.use_max)
			{
				ao.max(baked, mixing.max_mult);
			}
			else
			{
				const auto avg_mult = 1.f / float(iterations);
				ao.average(baked, avg_mult, i == 1 ? avg_mult : 1.f);
			}
		}
		if (verbose)
		{
			std::cout << std::endl;
		}
	}

	baked_data bake_driver_shadows()
	{
		baked_data ret;
		const auto scene_with_seatbelts = std::make_shared<bake::Scene>(root);
		root->set_active({"CINTURE_ON"}, true);

		{
			PERF("Special: baking AO for driver model")
			ret = bake_scene(std::make_shared<bake::Scene>(driver_root), scene_with_seatbelts->blockers + driver_root_scene->blockers, cfg_bake);
		}

		if (FEATURE_ACTIVE("DRIVER_BAKE_ANIMS"))
		{
			PERF("Special: baking animations for driver AO")
			bake_animations(driver_root, {driver_root}, scene_with_seatbelts->blockers + driver_root_scene->blockers,
				driver_steer_animations, driver_animations_steer_steps, driver_animations_mixing, {},
				cfg_bake, ret, "", false, true);
			bake_animations(driver_root, {driver_root}, scene_with_seatbelts->blockers + driver_root_scene->blockers,
				driver_animations, driver_animations_steps, driver_animations_mixing, {},
				cfg_bake, ret, "", false, true);
		}

		if (FEATURE_ACTIVE("DRIVER_INCLUDE_LOD_B") && driver_root_lodb)
		{
			PERF("Special: baking AO with animations for driver model (LOD B)")

			auto lodb_cfg_bake = cfg_bake;
			lodb_cfg_bake.filter_mode = bake::VertexFilterMode::VERTEX_FILTER_AREA_BASED;
			lodb_cfg_bake.ground_offset_factor = 0.2f;

			const auto lodb_scene = std::make_shared<bake::Scene>(driver_root_lodb);
			ret.extend(bake_scene(std::make_shared<bake::Scene>(driver_root_lodb), scene_with_seatbelts->blockers + lodb_scene->blockers, lodb_cfg_bake));
			bake_animations(driver_root_lodb, {driver_root_lodb}, scene_with_seatbelts->blockers + lodb_scene->blockers,
				driver_steer_animations, driver_animations_steer_steps, driver_animations_mixing, {},
				lodb_cfg_bake, ret, "", false, true);
			bake_animations(driver_root_lodb, {driver_root_lodb}, scene_with_seatbelts->blockers + lodb_scene->blockers,
				driver_animations, driver_animations_steps, driver_animations_mixing, {},
				lodb_cfg_bake, ret, "", false, true);
		}

		root->set_active({"CINTURE_ON"}, false);
		resulting_name = "main_geometry.vao-patch";
		return ret;
	}

	utils::path get_ai_lane_filename(const std::string& name) const
	{
		const auto base_name = input_file.filename_without_extension().string();
		return (input_file.extension() == ".ini" && base_name != "models" && base_name.find("models_") == 0
			? input_file.parent_path() / base_name.substr(strlen("models_"))
			: input_file.parent_path()) / "ai" / name;
	}

	baked_data bake_stuff(const std::shared_ptr<bake::Node>& root)
	{
		// Generating AO
		std::cout << "Main pass:" << std::endl;
		const auto root_scene = std::make_shared<bake::Scene>(root);
		root_scene->receivers -= root->find_meshes(bake_as_trees);
		root_scene->receivers -= root->find_meshes(bake_as_grass);

		if (FEATURE_ACTIVE("EXCLUDE_POBJECTS"))
		{
			root_scene->receivers -= root->find_any_meshes({"AC_POBJECT?"});
		}

		auto ao = bake_scene(root_scene, cfg_bake, true);

		// Applying trees shadow factor
		const auto trees_shadow_factor = config.get(section, "TREES_SHADOW_FACTOR", 1.f);
		if (!bake_as_trees.empty() && trees_shadow_factor != 1.f)
		{
			PERF("Special: applying trees shadow factor")
			const auto scene_without_trees = std::make_shared<bake::Scene>(*root_scene);
			scene_without_trees->blockers -= root->find_meshes(bake_as_trees);
			ao.max(bake_scene(scene_without_trees, cfg_bake), 1.f - trees_shadow_factor);
		}

		// Generating AO for trees
		if (!bake_as_trees.empty())
		{
			PERF("Special: baking trees")
			const auto trees_scene = std::make_shared<bake::Scene>(root);
			auto trees_meshes = root->find_meshes(bake_as_trees);
			trees_meshes -= root->find_meshes(bake_as_grass);
			auto trees_cfg_bake = cfg_bake;
			trees_cfg_bake.scene_offset_scale = 2.f;
			trees_cfg_bake.disable_normals = true;
			trees_cfg_bake.missing_normals_up = true;
			trees_cfg_bake.use_ground_plane_blocker = false;
			trees_cfg_bake.filter_mode = bake::VertexFilterMode::VERTEX_FILTER_AREA_BASED;
			trees_scene->receivers = trees_meshes;
			trees_scene->blockers -= trees_meshes;
			ao.replace(bake_scene(trees_scene, trees_cfg_bake));
		}

		// Generating AO for grass
		if (!bake_as_grass.empty())
		{
			baked_data grass_ao;
			const auto grass_scene = std::make_shared<bake::Scene>(root);
			auto grass_cfg_bake = cfg_bake;
			{
				PERF("Special: baking grass")
				const auto grass_meshes = root->find_meshes(bake_as_grass);
				grass_cfg_bake.scene_offset_scale = 0.5f;
				grass_cfg_bake.disable_normals = true;
				grass_cfg_bake.missing_normals_up = true;
				grass_cfg_bake.use_ground_plane_blocker = false;
				grass_cfg_bake.filter_mode = bake::VertexFilterMode::VERTEX_FILTER_AREA_BASED;
				grass_scene->receivers = grass_meshes;
				grass_ao = bake_scene(grass_scene, grass_cfg_bake);
			}
			if (!bake_as_trees.empty() && trees_shadow_factor != 1.f)
			{
				PERF("Special: baking grass, applying trees shadow factor")
				grass_scene->blockers -= root->find_meshes(bake_as_trees);
				grass_ao.max(bake_scene(grass_scene, grass_cfg_bake), 1.f - trees_shadow_factor);
			}
			ao.replace(grass_ao);
		}

		// Generating AO for AI lane
		const auto ao_samples_pits = config.get(section, "AO_SAMPLES_PIT_POINTS", false);
		const auto ao_samples_ai_lanes = config.get(section, "AO_SAMPLES_AI_LANES", false);
		if (ao_samples_pits || ao_samples_ai_lanes)
		{
			const auto ai_lanes_y_offset = config.get(section, "AO_SAMPLES_AI_LANES_Y_OFFSET", 0.5f);
			const auto ai_lanes_sides = config.get(section, "AO_SAMPLES_AI_LANES_SIDES", 0.67f) / 2.f;

			PERF("Special: baking extra AO samples")

			auto mesh = std::make_shared<bake::Mesh>();
			mesh->name = "@@__EXTRA_AO@";
			mesh->cast_shadows = false;
			mesh->receive_shadows = true;
			mesh->visible = true;
			mesh->material = std::make_shared<bake::Material>();
			mesh->signature_point = bake::Vec3{};

			static const auto distance_threshold = 4.f;
			static const float2 poisson_disk[10] = {
				optix::make_float2(-0.2027472f, -0.7174203f),
				optix::make_float2(-0.4839617f, -0.1232477f),
				optix::make_float2(0.4924171f, -0.06338801f),
				optix::make_float2(-0.6403998f, 0.6834511f),
				optix::make_float2(-0.8817205f, -0.4650014f),
				optix::make_float2(0.04554421f, 0.1661989f),
				optix::make_float2(0.1042245f, 0.9336259f),
				optix::make_float2(0.6152743f, 0.6344957f),
				optix::make_float2(0.5085323f, -0.7106467f),
				optix::make_float2(-0.9731231f, 0.1328296f),
			};

			if (ao_samples_pits)
			{
				for (const auto& n : root->find_nodes({"AC_PIT_?"}))
				{
					mesh->vertices.push_back({n->matrix[3], n->matrix[7], n->matrix[11]});
					for (auto p : poisson_disk)
					{
						mesh->vertices.push_back({n->matrix[3] + p.x * 4.f, n->matrix[7], n->matrix[11] + p.y * 4.f});
					}
				}
			}

			if (ao_samples_ai_lanes)
			{
				const auto ai_fast = load_ailane(get_ai_lane_filename("fast_lane.ai"));
				const auto ai_pits = load_ailane(get_ai_lane_filename("pit_lane.ai"));

				for (const auto& ai : {ai_fast, ai_pits})
				{
					auto dist = distance_threshold;
					for (const auto& v : ai)
					{
						dist += v.length;
						if (dist > distance_threshold)
						{
							mesh->vertices.push_back(v.point);
							dist = 0.f;
						}
					}
				}

				if (ai_lanes_sides > 0.f)
				{
					for (const auto& ai : {ai_fast, ai_pits})
					{
						auto dist = distance_threshold;
						for (auto i = 1U; i < ai.size(); i++)
						{
							dist += ai[i - 1].length;
							if (dist > distance_threshold)
							{
								auto v0 = *(float3*)&ai[i - 1].point;
								auto v1 = *(float3*)&ai[i].point;
								auto side = optix::normalize(optix::cross(v1 - v0, float3{0.f, 1.f, 0.f}));
								auto s0 = (ai[i - 1].side_left + ai[i - 1].side_left) * ai_lanes_sides;
								auto s1 = (ai[i - 1].side_right + ai[i - 1].side_right) * ai_lanes_sides;
								if (s0 == 0) s0 = 4.f;
								if (s1 == 0) s1 = 4.f;
								auto p0 = (v0 + v1) / 2.f - side * s0;
								auto p1 = (v0 + v1) / 2.f + side * s1;
								mesh->vertices.push_back(*(bake::Vec3*)&p0);
								mesh->vertices.push_back(*(bake::Vec3*)&p1);
								dist = 0.f;
							}
						}
					}
				}
			}

			auto cfg_extra_ao = cfg_bake;
			cfg_extra_ao.ground_offset_factor = 2.f;
			cfg_extra_ao.disable_normals = true;
			cfg_extra_ao.sample_on_points = true;
			cfg_extra_ao.sample_offset = {0.f, ai_lanes_y_offset, 0.f};
			cfg_extra_ao.use_ground_plane_blocker = false;
			cfg_extra_ao.filter_mode = bake::VertexFilterMode::VERTEX_FILTER_AREA_BASED;

			const auto extra_scene = std::make_shared<bake::Scene>(root);
			extra_scene->receivers = {mesh};
			ao.extend(bake_scene(extra_scene, cfg_extra_ao));
		}

		// Generating AO for driver
		if (FEATURE_ACTIVE("DRIVER_RECEIVES_SHADOWS"))
		{
			ao.extend(bake_driver_shadows());
		}

		// Special options
		if (FEATURE_ACTIVE("BAKE_BLURRED") && blurred_objects.any(input_file))
		{
			auto blurred = root->find_nodes(blurred_objects.blurred_names);
			PERF("Special: blurred objects: " + std::to_string(blurred.size()));
			root->set_active(blurred_objects.blurred_names, true);
			root->set_active(blurred_objects.static_names, false);
			ao.extend(bake_scene(std::make_shared<bake::Scene>(blurred), root_scene->blockers, cfg_bake));
			root->set_active(blurred_objects.blurred_names, false);
			root->set_active(blurred_objects.static_names, true);
		}

		if (FEATURE_ACTIVE("BAKE_COCKPIT_LR") && !root->find_nodes(names_lr).empty())
		{
			PERF("Special: baking low-res cockpit");
			root->set_active(names_lr, true);
			root->set_active(names_hr, false);
			ao.extend(bake_scene(std::make_shared<bake::Scene>(root->find_nodes(names_lr)), root_scene->blockers, cfg_bake));
			root->set_active(names_lr, false);
			root->set_active(names_hr, true);
		}

		if (FEATURE_ACTIVE("BAKE_SEATBELT"))
		{
			if (root->find_node("CINTURE_ON"))
			{
				PERF("Special: baking seatbelt (on)")
				root->set_active({"CINTURE_ON"}, true);
				auto seatbelt_on_blockers = root_scene->blockers;
				if (FEATURE_ACTIVE("DRIVER_CASTS_SHADOWS"))
				{
					seatbelt_on_blockers += driver_root_scene->blockers;
				}
				ao.extend(bake_scene(std::make_shared<bake::Scene>(root->find_node("CINTURE_ON")), seatbelt_on_blockers, cfg_bake));
				root->set_active({"CINTURE_ON"}, false);
			}
			if (root->find_node("CINTURE_OFF"))
			{
				PERF("Special: baking seatbelt (off)")
				root->set_active({"CINTURE_OFF"}, true);
				ao.extend(bake_scene(std::make_shared<bake::Scene>(root->find_node("CINTURE_OFF")), root_scene->blockers, cfg_bake));
				root->set_active({"CINTURE_OFF"}, false);
			}
		}

		// Applying animations
		if (!animation_instances.empty())
		{
			bake_animations(root, {root}, root_scene->blockers, cfg_bake, ao);

			if (FEATURE_ACTIVE("DRIVER_RECEIVES_SHADOWS"))
			{
				root->set_active({"CINTURE_ON"}, true);
				auto scene_with_seatbelts = std::make_shared<bake::Scene>(root);
				bake_animations(root, {driver_root}, scene_with_seatbelts->blockers + driver_root_scene->blockers, cfg_bake, ao);
				root->set_active({"CINTURE_ON"}, false);
			}

			if (FEATURE_ACTIVE("BAKE_COCKPIT_LR") && root->find_node("COCKPIT_LR"))
			{
				root->set_active({"COCKPIT_LR"}, true);
				root->set_active({"COCKPIT_HR"}, false);
				bake_animations(root, {root->find_node("COCKPIT_LR")}, root_scene->blockers, cfg_bake, ao, "low-res cockpit");
				root->set_active({"COCKPIT_LR"}, false);
				root->set_active({"COCKPIT_HR"}, true);
			}

			if (FEATURE_ACTIVE("BAKE_SEATBELT"))
			{
				if (root->find_node("CINTURE_ON"))
				{
					root->set_active({"CINTURE_ON"}, true);
					bake_animations(root, {root->find_node("CINTURE_ON")}, root_scene->blockers, cfg_bake, ao, "seatbelt on");
					root->set_active({"CINTURE_ON"}, false);
				}
				if (root->find_node("CINTURE_OFF"))
				{
					root->set_active({"CINTURE_OFF"}, true);
					bake_animations(root, {root->find_node("CINTURE_OFF")}, root_scene->blockers, cfg_bake, ao, "seatbelt off");
					root->set_active({"CINTURE_OFF"}, false);
				}
			}
		}

		// Special case for rotating meshes
		if (!rotating_x.empty() || !rotating_y.empty() || !rotating_z.empty())
		{
			bake_rotating(root, ao, cfg_bake, rotating_mixing);
			if (FEATURE_ACTIVE("BAKE_BLURRED") && blurred_objects.any(input_file))
			{
				root->set_active(blurred_objects.blurred_names, true);
				root->set_active(blurred_objects.static_names, false);
				bake_rotating(root, ao, cfg_bake, rotating_mixing, "blurred meshes");
				root->set_active(blurred_objects.blurred_names, false);
				root->set_active(blurred_objects.static_names, true);
			}
		}

		if (FEATURE_ACTIVE("SPLIT_AO"))
		{
			const auto steering_wheel = root->find_nodes({"STEER_HR", "STEER_LR"});
			if (!steering_wheel.empty())
			{
				PERF("Special: baking steering wheel for AO splitting")

				auto steering_wheel_scene = std::make_shared<bake::Scene>(steering_wheel);
				steering_wheel_scene->receivers = root->find_any_meshes({"STEER_HR", "STEER_LR"});

				// Baking turned by 180° and setting it as primary AO:
				for (const auto& n : steering_wheel)
				{
					const auto axis = float3{0.f, 0.f, 1.f};
					n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(180.f, &axis.x);
				}
				for (const auto& n : steering_wheel)
				{
					n->update_matrix();
					n->resolve_skinned();
				}
				ao.replace_primary(bake_scene(steering_wheel_scene, root_scene->blockers, cfg_bake));

				// Turning wheel back to 0°, baking and mixing it in primary AO with multiplier set to 0.67:
				for (const auto& n : steering_wheel)
				{
					n->matrix_local = n->matrix_local_orig;
				}
				for (auto n : steering_wheel)
				{
					n->update_matrix();
					n->resolve_skinned();
				}
				ao.max(bake_scene(steering_wheel_scene, root_scene->blockers, cfg_bake), 0.67f);
			}
		}

		// Optionally include LODs
		std::vector<std::shared_ptr<bake::Node>> lod_roots{root};
		if (FEATURE_ACTIVE("INCLUDE_LODS"))
		{
			auto data_lods = utils::ini_file(input_file.parent_path() / "data" / "lods.ini");
			for (const auto& s : data_lods.iterate_break("LOD"))
			{
				const auto name = data_lods.get(s, "FILE", std::string());
				const auto lod_in = data_lods.get(s, "IN", 0.f);
				if (name == input_file.filename().string())
				{
					continue;
				}

				const auto lod_root = load_model(input_file.parent_path() / name, cfg_load);
				lod_roots.push_back(lod_root);

				if (exterior_mult != 1.f)
				{
					add_materials(exterior_materials, {lod_root}, lod_root->find_nodes(names_hr + names_lr));
				}
				if (interior_mult != 1.f)
				{
					add_materials(interior_materials, lod_root->find_nodes(names_hr + names_lr), {});
				}

				auto lod_cfg_bake = cfg_bake;
				if (lod_in > 30.f)
				{
					if (FEATURE_ACTIVE("DISTANT_LODS_INCREASE_OFFSET"))
					{
						lod_cfg_bake.scene_offset_scale = 0.2f;
					}

					if (FEATURE_ACTIVE("DISTANT_LODS_AREA_BASED"))
					{
						lod_cfg_bake.filter_mode = bake::VERTEX_FILTER_AREA_BASED;
					}

					if (FEATURE_ACTIVE("DISTANT_LODS_SKIP_WHEELS"))
					{
						lod_root->set_active({"WHEEL_LF", "WHEEL_LR", "WHEEL_RF", "WHEEL_RR"}, false);
						for (auto w : {"WHEEL_LF", "WHEEL_LR", "WHEEL_RF", "WHEEL_RR"})
						{
							const auto& node = lod_root->find_node(w);
							if (node)
							{
								for (const auto& m : node->get_meshes())
								{
									ao.fill(m, 0.5f);
								}
							}
						}
					}
				}

				const auto lod_scene = std::make_shared<bake::Scene>(lod_root);

				{
					PERF("Special: adding LOD `" + name + "`")
					ao.extend(bake_scene(lod_scene, lod_cfg_bake));
				}

				if (!animation_instances.empty())
				{
					PERF("Special: baking animations for LOD `" + name + "`")
					bake_animations(lod_root, {lod_root}, lod_scene->blockers, lod_cfg_bake, ao, "", false);
				}

				if (!rotating_x.empty() || !rotating_y.empty() || !rotating_z.empty())
				{
					PERF("Special: baking rotating objects for LOD `" + name + "`")
					bake_rotating(lod_root, ao, lod_cfg_bake, rotating_mixing, "", false);
					if (FEATURE_ACTIVE("BAKE_BLURRED") && blurred_objects.any(input_file))
					{
						lod_root->set_active(blurred_objects.blurred_names, true);
						lod_root->set_active(blurred_objects.static_names, false);
						bake_rotating(lod_root, ao, lod_cfg_bake, rotating_mixing, "", false);
						lod_root->set_active(blurred_objects.blurred_names, false);
						lod_root->set_active(blurred_objects.static_names, true);
					}
				}

				resulting_name = "main_geometry.vao-patch";
			}
		}

		for (auto i = 0, j = 100; i < j; i++)
		{
			std::vector<std::string> names;
			auto mult = config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_MULT", 1.f);
			auto offset = config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_OFFSET", 0.f);
			auto exp = config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_EXP", 1.f);
			auto primary_only = config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_PRIMARY_ONLY", false);
			if (config.try_get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_NAMES", names)
				&& (mult != 1.f || offset != 0.f || exp != 1.f))
			{
				j = std::min(i + 100, 10000);

				std::vector<std::shared_ptr<bake::Mesh>> dimmed_meshes;
				for (const auto& r : lod_roots)
				{
					dimmed_meshes += r->find_meshes(names);
					for (const auto& node : r->find_nodes(names))
					{
						dimmed_meshes += node->get_meshes();
					}
				}

				for (const auto& mesh : dimmed_meshes)
				{
					auto f = ao.entries.find(mesh);
					if (f != ao.entries.end())
					{
						for (auto& e : f->second.main_set)
						{
							e.x = powf(optix::clamp(e.x, 0.f, 1.f), exp) * mult + offset;
							if (!primary_only)
							{
								e.y = powf(optix::clamp(e.y, 0.f, 1.f), exp) * mult + offset;
							}
						}
					}
				}
			}
		}

		return ao;
	}
};

#include <cuda_runtime.h>

struct cuda_capabilities
{
	bool query_failed;  // True on error
	int device_count{}; // Number of CUDA devices found 
	cudaDeviceProp best{};

	std::string info() const
	{
		std::stringstream ret;
		ret << best.name << ", cap.: " << best.major << "." << best.minor;
		return ret.str();
	}

	static cuda_capabilities get()
	{
		cuda_capabilities ret{};
		if (cudaGetDeviceCount(&ret.device_count) != cudaSuccess)
		{
			ret.query_failed = true;
		}
		else
		{
			for (auto dev = 0; dev < ret.device_count; dev++)
			{
				cudaDeviceProp device_prop{};
				cudaGetDeviceProperties(&device_prop, dev);
				if (device_prop.major > ret.best.major
					|| device_prop.major == ret.best.major && device_prop.minor > ret.best.minor)
				{
					ret.best = device_prop;
				}
			}
		}
		return ret;
	}
};

int main(int argc, const char** argv)
{
	#ifdef USE_TRYCATCH
	try
	{
		#endif
		SetConsoleOutputCP(CP_UTF8);

		bool is_cuda_available;
		int cuda_version;
		const auto cuda_result = cudaRuntimeGetVersion(&cuda_version);
		if (cuda_result == cudaSuccess && cuda_version)
		{
			is_cuda_available = true;
			std::cout << "Installed CUDA toolkit: " << cuda_version / 1000 << "." << cuda_version / 10 % 100 << "\n";
		}
		else
		{
			is_cuda_available = false;
			std::cout << "CUDA toolkit is not installed (you can get it here: https://developer.nvidia.com/cuda-80-ga2-download-archive, at least v8.0 is recommended)\n";
		}

		if (is_cuda_available)
		{
			const auto cap = cuda_capabilities::get();
			if (cap.query_failed)
			{
				std::cout << "Failed to check existance of any CUDA-compatible devices\n";
				is_cuda_available = false;
			}
			else if (cap.device_count == 0)
			{
				std::cout << "No CUDA-compatible devices found\n";
				is_cuda_available = false;
			}
			else if (cap.device_count == 1)
			{
				std::cout << "Found a CUDA-compatible device (" << cap.info() << ")\n";
			}
			else
			{
				std::cout << "Found " << cap.device_count << " CUDA-compatible devices (best one: " << cap.info() << ")\n";
			}
		}

		if (argc == 1)
		{
			wchar_t filename[1024]{};
			OPENFILENAME ofn{};
			ofn.lStructSize = sizeof ofn;
			ofn.hwndOwner = nullptr;
			ofn.lpstrFilter = L"AC Models\0*.kn5;*.ini\0Any Files\0*.*\0";
			ofn.lpstrFile = filename;
			ofn.nMaxFile = 1024;
			ofn.lpstrTitle = L"Choose a KN5 or INI file (for tracks) to make a VAO patch for";
			ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;
			if (GetOpenFileNameW(&ofn))
			{
				file_processor(utils::path(filename), is_cuda_available).run();
			}
		}
		else
		{
			for (auto i = 1; i < argc; i++)
			{
				file_processor(argv[i], is_cuda_available).run();
			}
		}

		return 0;
		#ifdef USE_TRYCATCH
	}
	catch (const std::exception& e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
		std::getchar();
		return 1;
	}
	catch (...)
	{
		std::cerr << "Unknown exception" << std::endl;
		std::getchar();
		return 1;
	}
	#endif
}
