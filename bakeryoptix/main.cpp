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
// #define REPLACEMENT_OPTIMIZATION_MODE

#include <bake_api.h>
#include <bake_wrap.h>
#include <cassert>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <set>
#include <ShlObj_core.h>
#include <utility>
#include <utils/cout_progress.h>
#include <utils/custom_crash_handler.h>
#include <utils/filesystem.h>
#include <utils/ini_file.h>
#include <utils/load_util.h>
#include <utils/nanort.h>
#include <utils/perf_moment.h>
#include <utils/std_ext.h>
#include <utils/vector_operations.h>
#include <vector>
#include <Windows.h>
#include <utils/binary_reader.h>
#include <utils/blob.h>
#include <utils/half.h>

#pragma comment(lib, "Shlwapi.lib")
#pragma comment(lib, "WinMM.lib")
#pragma comment(lib, "cudart_static.lib")
// #pragma comment(lib, "optix_prime.6.0.0.lib")
#pragma comment(lib, "optix.6.0.0.lib")

DEBUGTIME

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

	auto ret = utils::ini_file{config};
	const auto local_settings = get_executable_filename().parent_path() / "baked_shadow_params__local.ini";
	if (exists(local_settings))
	{
		ret.extend(utils::ini_file{local_settings});
	}
	return ret;
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

static void dump_obj(const utils::path& filename, const std::vector<std::shared_ptr<bake::Mesh>>& meshes)
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
		for (const auto& vertex : m->vertices)
		{
			*(bake::Vec3*)&f4 = vertex.pos;
			const auto w = f4 * x;
			o << "v " << w.x << " " << w.y << " " << w.z << std::endl;
		}
		start[j + 1] = start[j] + int(m->vertices.size());
	}
	for (auto j = 0U; j < meshes.size(); j++)
	{
		const auto& m = meshes[j];
		for (const auto& triangle : m->triangles)
		{
			const auto tr = int3{start[j], start[j], start[j]} + *(int3*)&triangle;
			o << "f " << tr.x << " " << tr.y << " " << tr.z << std::endl;
		}
	}
}

using namespace bake;

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
	std::vector<std::shared_ptr<Mesh>> animations_mixing_inverse;
	std::vector<std::shared_ptr<Animation>> animation_instances;

	float interior_mult = 1.f;
	float exterior_mult = 1.f;
	float driver_mult = 1.f;
	std::set<std::string> interior_materials;
	std::set<std::string> exterior_materials;
	std::set<std::string> driver_materials;

	std::shared_ptr<Node> root;
	std::shared_ptr<Node> driver_root;
	std::shared_ptr<Node> driver_root_lodb;
	std::shared_ptr<Scene> driver_root_scene;
	std::vector<std::shared_ptr<Animation>> driver_steer_animations;
	std::vector<std::shared_ptr<Animation>> driver_animations;
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

	const std::vector<std::string> names_hr{"@COCKPIT_HR", "STEER_HR"};
	const std::vector<std::string> names_lr{"@COCKPIT_LR", "STEER_LR"};

	std::vector<std::string> resolving_cockpit_hr;
	std::vector<std::string> resolving_cockpit_lr;

	Filter resolve_filter(const std::vector<std::string>& input) const
	{
		std::vector<std::string> result;
		for (const auto& i : input)
		{
			if (i == "@COCKPIT_HR")
			{
				result |= resolving_cockpit_hr;
			}
			else if (i == "@COCKPIT_LR")
			{
				result |= resolving_cockpit_lr;
			}
			else if (std::find(result.begin(), result.end(), i) == result.end())
			{
				result.push_back(i);
			}
		}
		return Filter{result};
	}

	void fill_animation(const std::shared_ptr<Node>& moving_root, const std::shared_ptr<Animation>& animation, const std::string& file)
	{
		auto& config = cfg_save.extra_config;

		animation->init(moving_root);
		const auto node_names = where(
			apply(animation->entries, [](const NodeTransition& t) { return t.node ? t.node->name : ""; }),
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
				else if (item == "@COCKPIT_HR")
				{
					list_new |= resolving_cockpit_hr;
				}
				else if (item == "@COCKPIT_LR")
				{
					list_new |= resolving_cockpit_lr;
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

	std::vector<std::string> bake_as_trees;
	std::vector<std::string> bake_as_grass;

	struct reduced_ground
	{
		std::vector<std::shared_ptr<Mesh>> meshes;
		float factor;
	};
	std::vector<reduced_ground> reduced_ground_params;

	file_processor(utils::path filename)
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
				std::cout << "Found spec. options" << (desc.empty() ? " for `" + id + "`" : ": " + desc) << std::endl;
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
		cfg_load.exclude_patch = config.get(section, "EXCLUDE_FROM_BAKING", std::vector<std::string>());
		cfg_load.exclude_blockers = config.get(section, "FORCE_PASSING_LIGHT", std::vector<std::string>());
		cfg_load.exclude_blockers_alpha_test = config.get(section, "FORCE_ALPHA_TEST_PASSING_LIGHT", false);
		cfg_load.exclude_blockers_alpha_blend = config.get(section, "FORCE_ALPHA_BLEND_PASSING_LIGHT", false);
		cfg_load.normals_bias = config.get(section, "NORMALS_Y_BIAS", 0.f);
		cfg_load.car_configs_dir = config.get("GENERAL", "CAR_CONFIGS_DIR", utils::path());
		cfg_load.track_configs_dir = config.get("GENERAL", "TRACK_CONFIGS_DIR", utils::path());

		resolving_cockpit_hr = config.get(section, "COCKPIT_HR_NODES", std::vector<std::string>{"COCKPIT_HR"});
		resolving_cockpit_lr = config.get(section, "COCKPIT_LR_NODES", std::vector<std::string>{"COCKPIT_LR"});

		cfg_bake.sample_on_points = false;
		cfg_bake.disable_normals = false;
		cfg_bake.batch_size = config.get("GENERAL", "BATCH_SIZE", 2000000ULL);
		cfg_bake.stack_size = config.get("GENERAL", "STACK_SIZE", 1024U);
		cfg_bake.num_samples = config.get(section, "SAMPLES", std::string("AUTO")) == "AUTO" ? 0 : config.get(section, "SAMPLES", 0);
		cfg_bake.min_samples_per_face = config.get(section, "MIN_SAMPLES_PER_FACE", 4);
		cfg_bake.num_rays = config.get(section, "RAYS_PER_SAMPLE", 64);
		cfg_bake.fix_incorrect_normals = config.get(section, "FIX_INCORRECT_NORMALS", true);
		cfg_bake.scene_offset_scale_horizontal = config.get(section, "RAYS_OFFSET", 0.1f);
		cfg_bake.scene_offset_scale_vertical = cfg_bake.scene_offset_scale_horizontal;
		cfg_bake.trees_light_pass_chance = config.get(section, "TREES_LIGHT_PASS_CHANCE", 0.9f);
		cfg_bake.use_ground_plane_blocker = config.get(section, "USE_GROUND", false);
		cfg_bake.ground_scale_factor = config.get(section, "GROUND_SCALE", 10.f);
		cfg_bake.scene_albedo = config.get(section, "BASE_ALBEDO", 0.9f);
		cfg_bake.bounce_counts = config.get(section, "BOUNCES", 5);
		cfg_bake.ground_upaxis = config.get(section, "GROUND_UP_AXIS", 1);
		cfg_bake.ground_offset_factor = config.get(section, "GROUND_OFFSET", 0.f);
		cfg_bake.filter_mode = config.get(section, "FILTER_MODE", std::string("LEAST_SQUARES")) == "LEAST_SQUARES"
			? VERTEX_FILTER_LEAST_SQUARES
			: VERTEX_FILTER_AREA_BASED;
		cfg_bake.regularization_weight = config.get(section, "FILTER_REGULARIZATION_WEIGHT", 0.1f);

		for (auto i = 0; i < 100; ++i)
		{
			const auto key = "EMISSIVE_" + std::to_string(i);
			float intensity;
			if (config.try_get(section, key, intensity))
			{
				const auto p0 = config.get(section, key + "_P0", float3());
				const auto p1 = config.get(section, key + "_P1", float3());
				const auto p2 = config.get(section, key + "_P2", float3());
				const auto p3 = config.get(section, key + "_P3", float3());
				cfg_bake.light_emitters.push_back({
					{
						Vec3{p0.x, p0.y, p0.z},
						Vec3{p1.x, p1.y, p1.z},
						Vec3{p2.x, p2.y, p2.z},
						Vec3{p3.x, p3.y, p3.z},
					},
					intensity
				});
			}
		}

		cfg_save.averaging_threshold = config.get(section, "AVERAGING_THRESHOLD", 0.f);
		cfg_save.averaging_cos_threshold = config.get(section, "AVERAGING_COS_THRESHOLD", 0.95f);
		cfg_save.average_ao_in_same_pos = config.get(section, "AVERAGING_OVERALL", true);
		cfg_save.brightness = config.get(section, "BRIGHTNESS", 1.02f);
		cfg_save.gamma = config.get(section, "GAMMA", 0.92f);
		cfg_save.opacity = config.get(section, "OPACITY", 0.97f);

		for (const auto& s : config.sections)
		{
			if (s.first != "GENERAL" && s.first.find("MODE_") != 0 && s.first.find("SPECIFIC_") != 0)
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
				for (const auto& mesh : root->find_meshes(resolve_filter(names)))
				{
					mesh->extra_samples_offset = offset;
				}
			}
		}

		// Resetting some objects just in case
		if (FEATURE_ACTIVE("RESET_BLURRED") && blurred_objects.any(input_file))
		{
			root->set_active(resolve_filter(blurred_objects.blurred_names), false);
			root->set_active(resolve_filter(blurred_objects.static_names), true);
		}

		if (FEATURE_ACTIVE("HIDE_SEATBELT"))
		{
			root->set_active(resolve_filter({"CINTURE_ON"}), false);
			root->set_active(resolve_filter({"CINTURE_OFF"}), false);
		}

		// Adding driver
		if (FEATURE_ACTIVE("DRIVER_CASTS_SHADOWS") || FEATURE_ACTIVE("DRIVER_RECEIVES_SHADOWS"))
		{
			const auto data_driver3d = utils::ini_file(input_file.parent_path() / "data" / "driver3d.ini");
			const auto driver_kn5 = input_file.parent_path().parent_path().parent_path() / "driver"
				/ (data_driver3d.get("MODEL", "NAME", std::string("driver")) + ".kn5");
			if (exists(driver_kn5))
			{
				driver_root = load_model(driver_kn5, cfg_load);

				const auto driver_pos = input_file.parent_path() / "driver_base_pos.knh";
				const auto hierarcy = load_hierarchy(driver_pos);
				if (hierarcy)
				{
					hierarcy->align(driver_root);
				}

				std::shared_ptr<Animation> steer_anim;
				if (FEATURE_ACTIVE("DRIVER_POSITION_WITH_STEER_ANIM"))
				{
					steer_anim = load_ksanim(input_file.parent_path() / "animations" / "steer.ksanim", true);
					steer_anim->apply(driver_root, 0.5f);
				}

				driver_root_scene = std::make_shared<Scene>(driver_root);

				if (FEATURE_ACTIVE("DRIVER_INCLUDE_LOD_B"))
				{
					auto driver_lodb = driver_kn5.parent_path() / driver_kn5.filename_without_extension() + "_B.kn5";
					if (exists(driver_lodb))
					{
						driver_root_lodb = load_model(driver_lodb, cfg_load);
						if (hierarcy)
						{
							hierarcy->align(driver_root_lodb);
						}
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
					auto list = root->find_any_meshes(resolve_filter(names));
					if (driver_root != nullptr)
					{
						list += driver_root->find_any_meshes(resolve_filter(names));
					}
					reduced_ground_params.push_back({list, factor});
				}
			}
		}

		// Getting list of materials if needed
		if (exterior_mult != 1.f)
		{
			add_materials(exterior_materials, {root}, root->find_nodes(resolve_filter(names_hr + names_lr)));
		}
		if (interior_mult != 1.f)
		{
			add_materials(interior_materials, root->find_nodes(resolve_filter(names_hr + names_lr)), {});
		}
		if (driver_mult != 1.f && driver_root != nullptr)
		{
			add_materials(driver_materials, {driver_root}, {});
		}

		// Preparing animations
		cfg_save.extra_config.set("SPLIT_AO", "COCKPIT_HR",
			resolve_filter(config.get("SPLIT_AO", "COCKPIT_HR", std::vector<std::string>{"@COCKPIT_HR"})).items);

		for (const auto& file : animations)
		{
			auto loaded = load_ksanim(input_file.parent_path() / "animations" / file);
			if (!loaded->entries.empty())
			{
				animation_instances.push_back(loaded);
				fill_animation(root, loaded, file);
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
					fill_animation(root, loaded, file);
				}
			}
		}

		const auto animations_mixing_inverse_names = config.get(section, "ANIMATIONS_MIXING_INVERSE", std::vector<std::string>());
		animations_mixing_inverse = root->find_meshes(resolve_filter(animations_mixing_inverse_names));
		for (const auto& n : root->find_nodes(resolve_filter(animations_mixing_inverse_names)))
		{
			animations_mixing_inverse += n->get_meshes();
		}
	}

	baked_data bake_scene(const std::shared_ptr<Scene>& scene, const bake_params& config, bool verbose = false)
	{
		return bake_scene(scene, scene->blockers, config, verbose);
	}

	baked_data bake_scene(const std::shared_ptr<Scene>& scene, const SceneBlockers& blockers,
		const bake_params& config, bool verbose = false)
	{
		if (reduced_ground_params.empty())
		{
			return bake_wrap::bake_scene(scene, blockers.full, config, verbose);
		}

		auto scene_adj = std::make_shared<Scene>(*scene);
		for (const auto& r : reduced_ground_params)
		{
			if (r.factor == 1.f)
			{
				scene_adj->receivers -= r.meshes;
			}
		}

		auto result = bake_wrap::bake_scene(scene_adj, blockers.full, config, verbose);
		auto config_noground = config;
		config_noground.use_ground_plane_blocker = false;

		for (const auto& r : reduced_ground_params)
		{
			scene_adj->receivers = scene->receivers & r.meshes;
			if (!scene_adj->receivers.empty())
			{
				result.brighten(bake_wrap::bake_scene(scene_adj, blockers.cut, config_noground, verbose), r.factor);
			}
		}

		return result;
	}

	static utils::path tmp_dir()
	{
		WCHAR result[MAX_PATH] = {};
		SHGetFolderPathW(nullptr, CSIDL_LOCAL_APPDATA, nullptr, SHGFP_TYPE_CURRENT, result);
		return utils::path(result) / "Temp";
	}

	static void* get_env_data(const std::unordered_map<std::string, std::string>& values)
	{
		if (values.empty()) return nullptr;

		auto& env_data = *new std::string;
		for (const auto& i : values)
		{
			env_data += i.first;
			env_data.push_back('=');
			env_data += i.second;
			env_data.push_back('\0');
		}
		return (void*)env_data.data();
	}

	static utils::path find_ac_root_path(const utils::path& input_file)
	{
		for (auto parent = input_file.parent_path(); !parent.empty() && parent.string().size() > 4; parent = parent.parent_path())
		{
			if (exists(parent / "acs.exe"))
			{
				return parent;
			}
		}
		return "";
	}

	struct tree_to_bake
	{
		Vec3 pos;
		Vec2 half_size;
	};

	std::vector<tree_to_bake> trees;
	uint64_t trees_key;

	static void set_race_ini(const std::string& track_id, const std::string& layout_id)
	{
		auto data = std::string(R"([BENCHMARK]
ACTIVE=0

[REPLAY]
ACTIVE=0

[REMOTE]
ACTIVE=0

[RESTART]
ACTIVE=0

[HEADER]
VERSION=2

[LAP_INVALIDATOR]
ALLOWED_TYRES_OUT=-1

[RACE]
MODEL=abarth500
MODEL_CONFIG=
SKIN=0_white_scorpion
TRACK={TRACK}
CONFIG_TRACK={TRACK_LAYOUT}
AI_LEVEL=100
CARS=1
DRIFT_MODE=0
FIXED_SETUP=0
PENALTIES=0
JUMP_START_PENALTY=0

[CAR_0]
SETUP=
SKIN=0_white_scorpion
MODEL=-
MODEL_CONFIG=
BALLAST=0
RESTRICTOR=0
DRIVER_NAME=Player
NATIONALITY=Canada
NATION_CODE=CAN

[OPTIONS]
USE_MPH=0

[GHOST_CAR]
RECORDING=0
PLAYING=0
LOAD=0
FILE=
ENABLED=0

[GROOVE]
VIRTUAL_LAPS=10
MAX_LAPS=30
STARTING_LAPS=0

[SESSION_0]
NAME=Practice
TYPE=1
DURATION_MINUTES=0
SPAWN_SET=HOTLAP_START

[TEMPERATURE]
AMBIENT=10
ROAD=9

[LIGHTING]
SUN_ANGLE=-12.11
TIME_MULT=0.0
CLOUD_SPEED=0.200

[WEATHER]
NAME=0_clear

[WIND]
SPEED_KMH_MIN=0
SPEED_KMH_MAX=0
DIRECTION_DEG=1

[DYNAMIC_TRACK]
SESSION_START=100
RANDOMNESS=1
LAP_GAIN=30
SESSION_TRANSFER=50)");
		std_ext::replace(data, "{TRACK}", track_id.c_str());
		std_ext::replace(data, "{TRACK_LAYOUT}", layout_id.c_str());
		WCHAR result[MAX_PATH] = {};
		SHGetFolderPathW(nullptr, CSIDL_PERSONAL, nullptr, SHGFP_TYPE_CURRENT, result);
		write_file((utils::path(result) / "Assetto Corsa/cfg/race_vao.ini").wstring(), data);
	}

	static bool run_ac(const utils::path& ac_root, const wchar_t* key, const utils::path& output_file)
	{
		if (exists(output_file))
		{
			_unlink(output_file.string().c_str());
		}

		STARTUPINFO start_info{};
		start_info.cb = sizeof(STARTUPINFO);

		SetEnvironmentVariableW(L"AC_TOOL_RUN", L"1");
		SetEnvironmentVariableW(L"AC_CFG_RACE_INI", L"race_vao.ini");
		SetEnvironmentVariableW(key, output_file.wstring().c_str());

		PROCESS_INFORMATION proc_info{};
		std::wstring args;
		if (!CreateProcessW((ac_root / "acs.exe").wstring().c_str(), &args[0],
			nullptr, nullptr, TRUE, 0, nullptr,
			ac_root.wstring().c_str(), &start_info, &proc_info))
		{
			throw std::runtime_error("Failed to start AC process");
		}

		WaitForSingleObject(proc_info.hProcess, INFINITE);
		SetEnvironmentVariableW(key, nullptr);
		return exists(output_file);
	}

	void run()
	{
		const auto ac_root = find_ac_root_path(input_file);

		// Procedural trees
		if ((FEATURE_ACTIVE("PROCEDURAL_TREES_OCCLUDE") || FEATURE_ACTIVE("PROCEDURAL_TREES_BAKE")) && !ac_root.empty())
		{
			const auto track_id = input_file.parent_path().filename().string();
			std::string track_layout;
			if (input_file.extension() == ".ini")
			{
				track_layout = input_file.filename_without_extension().string();
				if (strncmp(track_layout.c_str(), "models_", 7) == 0) track_layout = track_layout.substr(7);
				else track_layout.clear();
			}
			else
			{
				const auto list = list_files(input_file.parent_path(), "models_*.ini");
				if (!list.empty())
				{
					track_layout = list[0].filename_without_extension().string().substr(7);
				}
			}
			std::cout << "Track ID: " << track_id << ", layout: " << (track_layout.empty() ? "<none>" : track_layout) << std::endl;
			set_race_ini(track_id, track_layout);

			PERF("Running AC in background to get list of procedural trees ready");
			const auto tmp_path = tmp_dir() / "ac_trees.bin";
			if (!run_ac(ac_root, L"AC_PREPARE_TREES_LIST", tmp_path))
			{
				throw std::runtime_error("Failed to extract the list of procedural trees");
			}

			std::vector<MeshVertex> vertices;
			std::vector<Triangle> triangles;
			auto add_quad = [&](const Vec3& p, const Vec3& l, const Vec3& t)
			{
				const auto i = uint32_t(vertices.size());
				vertices.push_back({p - l - t, {0.f, 0.f}});
				vertices.push_back({p - l + t, {0.f, 1.f}});
				vertices.push_back({p + l + t, {1.f, 1.f}});
				vertices.push_back({p + l - t, {1.f, 0.f}});
				triangles.push_back({i, i + 1, i + 2});
				triangles.push_back({i, i + 2, i + 3});
				triangles.push_back({i, i + 2, i + 1});
				triangles.push_back({i, i + 3, i + 2});
			};

			utils::binary_reader trees_reader(tmp_path);
			if (trees_reader.read_uint() != 1U)
			{
				throw std::runtime_error("Failed to get the list of procedural trees: unsupported version (bakery needs an update)");
			}

			trees_key = trees_reader.read_uint64();
			for (auto i = trees_reader.read_uint(); i > 0; --i)
			{
				auto pos = trees_reader.read_f3();
				auto hsize = trees_reader.read_f2();
				trees.emplace_back(tree_to_bake{Vec3{pos.x, pos.y, pos.z}, Vec2{hsize.x, hsize.y}});
				const auto mult = std::max(std::min(hsize.y * 1.8f - 0.2f, 1.f), 0.f);
				if (mult > 0.f)
				{
					hsize.x *= 0.7f * mult;
					add_quad(Vec3{pos.x, pos.y - hsize.y * 0.5f, pos.z}, Vec3{hsize.x * 0.65f, 0.f, 0.f}, Vec3{0.f, 0.f, hsize.x * 0.65f});
					add_quad(Vec3{pos.x, pos.y + hsize.y * 0.5f, pos.z}, Vec3{hsize.x * 0.65f, 0.f, 0.f}, Vec3{0.f, 0.f, hsize.x * 0.65f});
					add_quad(Vec3{pos.x, pos.y, pos.z}, Vec3{hsize.x, 0.f, 0.f}, Vec3{0.f, 0.f, hsize.x});
					add_quad(Vec3{pos.x, pos.y, pos.z}, Vec3{hsize.x * 0.9f, 0.f, 0.f}, Vec3{0.f, hsize.y * 0.7f, 0.f});
					add_quad(Vec3{pos.x, pos.y, pos.z}, Vec3{0.f, 0.f, hsize.x * 0.9f}, Vec3{0.f, hsize.y * 0.7f, 0.f});
				}
			}

			if (FEATURE_ACTIVE("PROCEDURAL_TREES_OCCLUDE") && !vertices.empty())
			{
				auto mesh = std::make_shared<bake::Mesh>();
				mesh->name = "trees";
				mesh->matrix = NodeTransformation::identity();
				mesh->cast_shadows = true;
				mesh->is_visible = true;
				mesh->is_transparent = false;
				mesh->vertices = std::move(vertices);
				mesh->triangles = std::move(triangles);
				mesh->layer = 0;
				mesh->lod_in = 0.f;
				mesh->lod_out = FLT_MAX;
				mesh->receive_shadows = false;
				mesh->material = nullptr;
				mesh->signature_point = mesh->vertices[0].pos;
				mesh->is_renderable = true;
				root->add_child(mesh);
			}
		}

		// Optional dumping
		std::string dump_as;
		if (config.try_get(section, "DUMP_INSTEAD_OF_BAKING", dump_as) && !dump_as.empty())
		{
			Animation::apply_all(root, {
				load_ksanim(input_file.parent_path() / "animations" / "car_DOOR_L.ksanim"),
				load_ksanim(input_file.parent_path() / "animations" / "car_DOOR_R.ksanim"),
			}, 1.f);
			root->update_matrix();
			root->resolve_skinned();
			if (driver_root)
			{
				driver_root->update_matrix();
				driver_root->resolve_skinned();
				dump_obj(dump_as, root->get_meshes() + driver_root->get_meshes());
			}
			else
			{
				dump_obj(dump_as, root->get_meshes());
			}
			return;
		}

		auto ao = bake_stuff(root);

		if (cfg_save.average_ao_in_same_pos)
		{
			PERF("Overall averaging");
			ao.smooth_ao();
		}

		// Saving result
		const auto destination = input_file.parent_path() / resulting_name;
		{
			if (!exterior_materials.empty())
			{
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_EXT", "MATERIALS", to_vector(exterior_materials));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_EXT", "PROP_0", std::vector<std::string>{"ksAmbient", "*" + std::to_string(exterior_mult)});
			}

			if (!interior_materials.empty())
			{
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_INT", "MATERIALS", to_vector(interior_materials));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_INT", "PROP_0", std::vector<std::string>{"ksAmbient", "*" + std::to_string(interior_mult)});
			}

			if (!driver_materials.empty())
			{
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_DRIVER", "MATERIALS", to_vector(driver_materials));
				cfg_save.extra_config.set("SHADER_REPLACEMENT_0_FIX_DRIVER", "PROP_0", std::vector<std::string>{"ksAmbient", "*" + std::to_string(driver_mult)});
			}

			PERF("Saving to `" + destination.relative_ac() + "`");
			ao.save(destination, cfg_save, FEATURE_ACTIVE("SPLIT_AO"));
		}

		if (FEATURE_ACTIVE("PROCEDURAL_TREES_FINALIZE"))
		{
			{
				PERF("Running AC in background to compile procedural trees");
				if (!run_ac(ac_root, L"AC_FINALIZE_TREES_LIST", input_file.parent_path() / "compiled_trees.bin"))
				{
					throw std::runtime_error("Failed to compile the list of procedural trees");
				}
			}

			PERF("Resaving to `" + destination.relative_ac() + "` without trees AO");
			ao.extra_entries.erase("TreeSamples.data");
			ao.save(destination, cfg_save, FEATURE_ACTIVE("SPLIT_AO"));
		}
	}

	void bake_animations(const std::shared_ptr<Node>& moving_root, const std::vector<std::shared_ptr<Node>>& targets,
		const SceneBlockers& blockers, const bake_params& cfg_bake, baked_data& ao, const std::string& comment = "",
		bool verbose = true, bool apply_to_both_sets = false)
	{
		bake_animations(moving_root, targets, blockers, animation_instances, animations_steps, animations_mixing, animations_mixing_inverse,
			cfg_bake, ao, comment, verbose, apply_to_both_sets);
	}

	void bake_animations(const std::shared_ptr<Node>& moving_root, const std::vector<std::shared_ptr<Node>>& targets,
		const SceneBlockers& blockers,
		std::vector<std::shared_ptr<Animation>>& animation_instances, const std::vector<float>& animations_steps, const mixing_params& animations_mixing,
		const std::vector<std::shared_ptr<Mesh>>& animations_mixing_inverse, const bake_params& cfg_bake, baked_data& ao, const std::string& comment = "",
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
			if (Animation::apply_all(moving_root, animation_instances, pos))
			{
				const auto baked = bake_scene(std::make_shared<Scene>(targets), blockers, cfg_bake);
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
		Animation::apply_all(moving_root, animation_instances, 0.f);
		if (verbose)
		{
			std::cout << std::endl;
		}
	}

	void bake_rotating(const std::shared_ptr<Node>& root, baked_data& ao, const bake_params& params, const mixing_params& mixing, const std::string& comment = "",
		const bool verbose = true)
	{
		const auto root_scene = std::make_shared<Scene>(root);
		const auto nodes_x = root->find_nodes(resolve_filter(rotating_x));
		const auto nodes_y = root->find_nodes(resolve_filter(rotating_y));
		const auto nodes_z = root->find_nodes(resolve_filter(rotating_z));
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
				n->matrix_local = n->matrix_local_orig * NodeTransformation::rotation(deg, &axis.x);
			}
			for (const auto& n : nodes_y)
			{
				const auto axis = float3{0.f, 1.f, 0.f};
				n->matrix_local = n->matrix_local_orig * NodeTransformation::rotation(deg, &axis.x);
			}
			for (const auto& n : nodes_z)
			{
				const auto axis = float3{0.f, 0.f, 1.f};
				n->matrix_local = n->matrix_local_orig * NodeTransformation::rotation(deg, &axis.x);
			}
			const auto baked = bake_scene(std::make_shared<Scene>(rotating), root_scene->blockers, params);
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
		const auto scene_with_seatbelts = std::make_shared<Scene>(root);
		root->set_active(resolve_filter({"CINTURE_ON"}), true);

		{
			PERF("Special: baking AO for driver model")
			ret = bake_scene(std::make_shared<Scene>(driver_root), scene_with_seatbelts->blockers + driver_root_scene->blockers, cfg_bake);
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
			lodb_cfg_bake.filter_mode = VERTEX_FILTER_AREA_BASED;
			lodb_cfg_bake.ground_offset_factor = 0.2f;

			const auto lodb_scene = std::make_shared<Scene>(driver_root_lodb);
			ret.extend(bake_scene(std::make_shared<Scene>(driver_root_lodb), scene_with_seatbelts->blockers + lodb_scene->blockers, lodb_cfg_bake));
			bake_animations(driver_root_lodb, {driver_root_lodb}, scene_with_seatbelts->blockers + lodb_scene->blockers,
				driver_steer_animations, driver_animations_steer_steps, driver_animations_mixing, {},
				lodb_cfg_bake, ret, "", false, true);
			bake_animations(driver_root_lodb, {driver_root_lodb}, scene_with_seatbelts->blockers + lodb_scene->blockers,
				driver_animations, driver_animations_steps, driver_animations_mixing, {},
				lodb_cfg_bake, ret, "", false, true);
		}

		root->set_active(resolve_filter({"CINTURE_ON"}), false);
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

	void bake_extra_samples_old(baked_data& ao, const std::shared_ptr<Node>& root, bool ao_samples_pits, bool ao_samples_ai_lanes)
	{
		if (!ao_samples_pits && !ao_samples_ai_lanes) return;
		const auto ai_lanes_y_offset = config.get(section, "AO_SAMPLES_AI_LANES_Y_OFFSET", std::vector<float>{0.5f});
		const auto ai_lanes_sides = config.get(section, "AO_SAMPLES_AI_LANES_SIDES", 0.67f) / 2.f;
		const auto ai_lanes_pitlane = config.get(section, "AO_SAMPLES_AI_LANES_PITLINE_WIDTH", 4.f) / 2.f;
		const auto ai_lanes_min_distance = config.get(section, "AO_SAMPLES_AI_LANES_MIN_DISTANCE", 4.f);

		PERF("Special: baking extra AO samples")

		auto mesh = std::make_shared<Mesh>();
		mesh->name = "@@__EXTRA_AO@";
		mesh->cast_shadows = false;
		mesh->receive_shadows = true;
		mesh->is_visible = true;
		mesh->material = std::make_shared<Material>();
		mesh->signature_point = Vec3{};

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
			for (const auto& n : root->find_nodes(resolve_filter({"AC_PIT_?"})))
			{
				mesh->vertices.push_back({{n->matrix[3], n->matrix[7], n->matrix[11]}, {}});
				for (auto p : poisson_disk)
				{
					mesh->vertices.push_back({{n->matrix[3] + p.x * 4.f, n->matrix[7], n->matrix[11] + p.y * 4.f}, {}});
				}
			}
		}

		if (ao_samples_ai_lanes)
		{
			const auto ai_fast = load_ailane(get_ai_lane_filename("fast_lane.ai"));
			const auto ai_pits = load_ailane(get_ai_lane_filename("pit_lane.ai"));

			for (const auto& ai : {ai_fast, ai_pits})
			{
				auto last_length = -ai_lanes_min_distance;
				for (const auto& v : ai)
				{
					if (v.length - last_length >= ai_lanes_min_distance)
					{
						mesh->vertices.push_back({v.point, {}});
						last_length = v.length;
					}
				}
			}

			if (ai_lanes_sides > 0.f)
			{
				for (const auto& ai : {ai_fast, ai_pits})
				{
					auto last_length = -ai_lanes_min_distance;
					for (auto i = 1U; i < ai.size(); i++)
					{
						if (ai[i - 1].length - last_length >= ai_lanes_min_distance)
						{
							const auto& v0 = *(float3*)&ai[i - 1].point;
							const auto& v1 = *(float3*)&ai[i].point;
							auto side = optix::normalize(optix::cross(v1 - v0, float3{0.f, 1.f, 0.f}));
							auto s0 = (ai[i - 1].side_left + ai[i - 1].side_left) * ai_lanes_sides;
							auto s1 = (ai[i - 1].side_right + ai[i - 1].side_right) * ai_lanes_sides;
							if (s0 == 0) s0 = ai_lanes_pitlane;
							if (s1 == 0) s1 = ai_lanes_pitlane;
							if (s0 > 0.f)
							{
								const auto p0 = (v0 + v1) / 2.f - side * s0;
								mesh->vertices.push_back({*(Vec3*)&p0, {}});
							}
							if (s1 > 0.f)
							{
								const auto p1 = (v0 + v1) / 2.f + side * s1;
								mesh->vertices.push_back({*(Vec3*)&p1, {}});
							}
							last_length = ai[i - 1].length;
						}
					}
				}
			}
		}

		auto cfg_extra_ao = cfg_bake;
		cfg_extra_ao.ground_offset_factor = 2.f;
		cfg_extra_ao.disable_normals = true;
		cfg_extra_ao.sample_on_points = true;
		cfg_extra_ao.use_ground_plane_blocker = false;
		cfg_extra_ao.filter_mode = VERTEX_FILTER_AREA_BASED;

		const auto extra_scene = std::make_shared<Scene>(root);
		extra_scene->receivers = {mesh};

		baked_data extra_ao;
		for (auto offset : ai_lanes_y_offset)
		{
			cfg_extra_ao.sample_offset = {0.f, offset, 0.f};
			if (extra_ao.entries.empty())
			{
				extra_ao = bake_scene(extra_scene, cfg_extra_ao);
			}
			else
			{
				extra_ao.max(bake_scene(extra_scene, cfg_extra_ao));
			}
		}
		ao.extend(extra_ao);
	}

	static uint64_t vec3_key(const Vec3& p)
	{
		const auto c = [](float v) -> uint64_t
		{
			auto i = int64_t(v);
			return *(uint64_t*)&i & 0xffffffff;
		};
		uint64_t r = c(p.x);
		r |= c(p.z) << 32;
		r ^= c(p.y) << 16;
		return r;
	}

	static bool is_physics_surface(const std::string& name)
	{
		auto any = false;
		for (const auto& c : name)
		{
			if (c == ' ' || c == '-' || isdigit(c))
			{
				any = any || isdigit(c);
			}
			else
			{
				return any && isalpha(c) && (strncmp(&c, "WALL", 4) != 0 || isalpha((&c)[4]));
			}
		}
		return false;
	}

	static void write_file(const std::wstring& filename, const std::string& data)
	{
		auto s = std::ofstream(filename, std::ios::binary);
		std::copy(data.begin(), data.end(), std::ostream_iterator<char>(s));
	}

	void bake_extra_samples_new(baked_data& ao, const std::shared_ptr<Node>& root)
	{
		std::cout << "Extra AO samples:" << std::endl;

		static float large_grid_size = 40.f;
		static float large_grid_height = 20.f;
		static uint32_t small_grid_size = 10U;
		static uint32_t small_grid_height = 3U;
		static float base_vertical_offset = 0.1f;

		struct mesh_info
		{
			std::shared_ptr<Mesh> mesh;
			std::vector<Vec3> vertices;
			nanort::BVHAccel<float> accel;
			Vec2 bb_min;
			Vec2 bb_max;

			static float distance_sqr(const Vec3& v0, const Vec3& v1)
			{
				const auto dx = v0.x - v1.x;
				const auto dy = v0.z - v1.z;
				return dx * dx + dy * dy;
			}

			void get_y(Vec3 v, float& y, float& y_fault)
			{
				if (v.x > bb_min.x && v.z > bb_min.y
					&& v.x < bb_max.x && v.z < bb_max.y)
				{
					nanort::Ray<float> r{};
					r.dir[0] = 0.f;
					r.dir[1] = -1.f;
					r.dir[2] = 0.f;
					r.org[0] = v.x;
					r.org[1] = v.y + 10.f;
					r.org[2] = v.z;
					r.max_t = 20.f;
					r.min_t = 0.f;

					nanort::TriangleIntersector<> triangle_intersecter(&vertices[0].x, &mesh->triangles[0].a);
					nanort::TriangleIntersection<> isect{};
					if (accel.Traverse<true>(r, triangle_intersecter, &isect))
					{
						const auto new_y = r.org[1] - isect.t;
						if (new_y < y || y_fault >= 0.f)
						{
							y = new_y;
							y_fault = -1.f;
						}
					}
				}
			}

			void get_y_fallback(Vec3 v, float& y, float& y_fault)
			{
				if (y_fault > 0.f
					&& v.x > bb_min.x - 60.f && v.z > bb_min.y - 60.f
					&& v.x < bb_max.x + 60.f && v.z < bb_max.y + 60.f)
				{
					for (const auto& p : vertices)
					{
						const auto d = distance_sqr(p, v);
						if (d < y_fault)
						{
							y_fault = d;
							y = p.y;
						}
					}
				}
			}

			void finalize(std::shared_ptr<Mesh>&& mesh_ptr)
			{
				bb_min = {FLT_MAX, FLT_MAX};
				bb_max = {-FLT_MAX, -FLT_MAX};
				for (auto p : vertices)
				{
					bb_min.x = std::min(bb_min.x, p.x);
					bb_min.y = std::min(bb_min.y, p.z);
					bb_max.x = std::max(bb_max.x, p.x);
					bb_max.y = std::max(bb_max.y, p.z);
				}

				nanort::BVHBuildOptions<float> build_options;
				build_options.cache_bbox = true;
				const nanort::TriangleMesh<float> rt_mesh(&vertices[0].x, &mesh_ptr->triangles[0].a);
				accel.Build(uint32_t(mesh_ptr->triangles.size()), rt_mesh, nanort::TriangleSAHPred<float>(&vertices[0].x, &mesh_ptr->triangles[0].a), build_options);
				mesh = std::move(mesh_ptr);
			}
		};

		std::vector<mesh_info> infos;
		std::unordered_map<size_t, Vec3> points;

		{
			PERF("\tCalculating chunks")
			auto l = root->get_meshes();
			infos.reserve(l.size());
			cout_progress g{l.size()};
			for (auto& m : l)
			{
				g.report();
				if (is_physics_surface(m->name) && !m->vertices.empty())
				{
					infos.push_back(mesh_info());
					auto& mi = *infos.rbegin();
					mi.vertices.reserve(m->vertices.size());
					for (const auto& v : m->vertices)
					{
						const auto p = Vec3{roundf(v.pos.x / large_grid_size), roundf(v.pos.y / large_grid_height), roundf(v.pos.z / large_grid_size)};
						points[vec3_key(p)] = {p.x * large_grid_size, p.y * large_grid_height, p.z * large_grid_size};
						mi.vertices.push_back(v.pos);
					}
					mi.finalize(std::move(m));
				}
			}
		}

		std::cout << "\tSurface meshes: " << infos.size() << std::endl;
		std::cout << "\tLarge grid chunks to prepare: " << points.size() << std::endl;

		if (points.size() > 20000)
		{
			if (points.size() > 90000)
			{
				small_grid_size = 4U;
			}
			else if (points.size() > 60000)
			{
				small_grid_size = 6U;
			}
			else if (points.size() > 30000)
			{
				small_grid_size = 8U;
			}
			small_grid_height = 2U;
			std::cout << "\tToo many chunks, lowering resolution to " << small_grid_size << "x" << small_grid_height << std::endl;
		}

		struct grid_data
		{
			Vec3 center;
			std::vector<float> rows_surface_y;

			grid_data(const Vec3& center)
				: center(center)
			{
				rows_surface_y.resize(small_grid_size * small_grid_size);
			}

			Vec3 calculate_base_pos(uint32_t x, uint32_t z) const
			{
				auto c = center;
				c.x += ((float(x) + 0.5f) / float(small_grid_size) - 0.5f) * large_grid_size;
				c.z += ((float(z) + 0.5f) / float(small_grid_size) - 0.5f) * large_grid_size;
				return c;
			}

			Vec3 calculate_pos(uint32_t x, uint32_t y, uint32_t z) const
			{
				auto c = calculate_base_pos(x, z);
				c.y = rows_surface_y[z * small_grid_size + x] + (float(y) + base_vertical_offset) / float(small_grid_height) * large_grid_height;
				return c;
			}

			void set_ground_pos(uint32_t x, uint32_t z, float surface_y)
			{
				rows_surface_y[z * small_grid_size + x] = surface_y;
			}
		};
		std::vector<grid_data> samples;
		auto cast_hits = 0U;
		auto cast_misses = 0U;

		{
			PERF("\tDistributing samples")
			cout_progress g{points.size()};
			samples.reserve(points.size());
			for (const auto& p : points)
			{
				g.report();
				samples.push_back(grid_data(p.second));
				auto& gr = *samples.rbegin();

				std::vector<mesh_info*> infos_local;
				const auto& p_center = p.second;
				for (auto& i : infos)
				{
					if (p_center.x > i.bb_min.x - large_grid_size / 2.f
						&& p_center.z > i.bb_min.y - large_grid_size / 2.f
						&& p_center.x < i.bb_max.x + large_grid_size / 2.f
						&& p_center.z < i.bb_max.y + large_grid_size / 2.f)
					{
						infos_local.push_back(&i);
					}
				}

				for (auto x = 0U; x < small_grid_size; ++x)
				{
					for (auto z = 0U; z < small_grid_size; ++z)
					{
						auto c = gr.calculate_base_pos(x, z);
						auto surface_y = c.y - large_grid_height / 2.f;
						auto surface_y_fault = FLT_MAX;
						for (auto& mi : infos_local)
						{
							mi->get_y(c, surface_y, surface_y_fault);
						}
						if (surface_y_fault > 0.f)
						{
							++cast_misses;
							/*for (auto& mi : infos_local)
							{
								mi->get_y_fallback(c, surface_y, surface_y_fault);
							}*/
						}
						else ++cast_hits;
						gr.set_ground_pos(x, z, surface_y);
					}
				}
			}
		}
		std::cout << "\tAligned samples: " << cast_hits << " out of " << cast_hits + cast_misses
			<< " (" << roundf(100.f * float(cast_hits) / float(cast_hits + cast_misses)) << "%)" << std::endl;

		const auto extra_scene = std::make_shared<Scene>(root);
		extra_scene->extra_receive_points.reserve(samples.size() * (small_grid_size * small_grid_size * small_grid_height));
		for (const auto& p : samples)
		{
			for (auto x = 0U; x < small_grid_size; ++x)
			{
				for (auto z = 0U; z < small_grid_size; ++z)
				{
					for (auto y = 0U; y < small_grid_height; ++y)
					{
						extra_scene->extra_receive_points.push_back(p.calculate_pos(x, y, z));
					}
				}
			}
		}

		auto cfg_extra_ao = cfg_bake;
		cfg_extra_ao.ground_offset_factor = 1.f;
		cfg_extra_ao.disable_normals = true;
		cfg_extra_ao.sample_on_points = true;
		cfg_extra_ao.use_ground_plane_blocker = false;
		cfg_extra_ao.filter_mode = VERTEX_FILTER_AREA_BASED;
		cfg_extra_ao.sample_offset = {0.f, 0.f, 0.f};

		std::cout << "\tTotal number of AO grid samples: " << extra_scene->extra_receive_points.size() << std::endl;
		std::vector<float> baked;
		{
			PERF("\tActual baking")
			baked = std::move(bake_scene(extra_scene, cfg_extra_ao).extra_points_ao);
		}

		utils::blob data;
		{
			PERF("\tEncoding")
			cout_progress g{samples.size()};

			data << 1U;
			data << large_grid_size;
			data << large_grid_height;
			data << uint16_t(small_grid_size);
			data << uint16_t(small_grid_height);
			data << base_vertical_offset;
			data << uint32_t(samples.size());

			auto j = 0U;
			for (const auto& i : samples)
			{
				g.report();
				data << i.center;
				data.append(i.rows_surface_y.data(), sizeof(float) * i.rows_surface_y.size());
				for (auto x = 0U; x < small_grid_size; ++x)
				{
					for (auto z = 0U; z < small_grid_size; ++z)
					{
						for (auto y = 0U; y < small_grid_height; ++y)
						{
							data << uint8_t(powf(std::min(std::max(baked[j++], 0.f), 1.f), VAO_ENCODE_POW) * 255.f);
						}
					}
				}
			}
		}

		ao.extra_entries["ExtraSamples.data"] = std::move(data);
	}

	void bake_extra_samples(baked_data& ao, const std::shared_ptr<Node>& root)
	{
		if (config.get(section, "AO_SAMPLES_PHYSICS_SURFACES", false))
		{
			bake_extra_samples_new(ao, root);
		}
		else
		{
			bake_extra_samples_old(ao, root,
				config.get(section, "AO_SAMPLES_PIT_POINTS", false),
				config.get(section, "AO_SAMPLES_AI_LANES", false));
		}
	}

	#define M_PI 3.14159265358979323846f
	#define AXIS_SAMPLE_COUNT 8

	static Vec4 sh_evaluate(const Vec3& dir)
	{
		Vec4 result;
		result.x = 0.28209479177387814347403972578039f;
		result.y = -0.48860251190291992158638462283836f * dir.y;
		result.z = 0.48860251190291992158638462283836f * dir.z;
		result.w = -0.48860251190291992158638462283836f * dir.x;
		return result;
	}

	static Vec3 sh_get_uniform_sphere_sample(float azimuth_x, float zenith_y)
	{
		const float phi = 2.f * M_PI * azimuth_x;
		const float z = 1.f - 2.f * zenith_y;
		const float r = sqrt(std::max(0.f, 1.f - z * z));
		return Vec3{r * cos(phi), z, r * sin(phi)};
	}

	void bake_procedural_trees(baked_data& ao, const std::shared_ptr<Node>& root, const std::vector<tree_to_bake>& trees, uint64_t trees_key)
	{
		std::cout << "Baking procedural trees:" << std::endl;
		const auto extra_scene = std::make_shared<Scene>(root);

		{
			PERF("\tGenerating samples")
			extra_scene->extra_receive_directed_points.reserve(trees.size() * (AXIS_SAMPLE_COUNT * AXIS_SAMPLE_COUNT));
			for (const auto& p : trees)
			{
				for (float az = 0.5f; az < AXIS_SAMPLE_COUNT; az += 1.0f)
				{
					for (float ze = 0.5f; ze < AXIS_SAMPLE_COUNT; ze += 1.0f)
					{
						auto dir = sh_get_uniform_sphere_sample(az / AXIS_SAMPLE_COUNT, ze / AXIS_SAMPLE_COUNT);
						dir.y = dir.y * 0.45f + 0.55f;
						dir = dir.normalize();
						extra_scene->extra_receive_directed_points.emplace_back(p.pos + Vec3{dir.x * p.half_size.x, p.half_size.y * dir.y, dir.z * p.half_size.x}, dir);
					}
				}
			}
		}

		auto cfg_extra_ao = cfg_bake;
		cfg_extra_ao.num_rays = 8;
		cfg_extra_ao.bounce_counts = 2;
		cfg_extra_ao.disable_normals = true;
		cfg_extra_ao.sample_on_points = true;
		cfg_extra_ao.use_ground_plane_blocker = false;
		cfg_extra_ao.filter_mode = VERTEX_FILTER_AREA_BASED;
		cfg_extra_ao.sample_offset = {0.f, 0.f, 0.f};

		std::cout << "\tTotal number of AO tree samples: " << extra_scene->extra_receive_directed_points.size() << std::endl;
		std::vector<float> baked;
		{
			PERF("\tActual baking")
			baked = std::move(bake_scene(extra_scene, cfg_extra_ao).extra_points_ao);
		}

		utils::blob data;
		data << 1U;
		data << trees_key;
		auto i = 0;
		for (const auto& p : trees)
		{
			Vec4 sh{};
			for (float az = 0.5f; az < AXIS_SAMPLE_COUNT; az += 1.f)
			{
				for (float ze = 0.5f; ze < AXIS_SAMPLE_COUNT; ze += 1.f)
				{
					const auto dir = sh_get_uniform_sphere_sample(az / AXIS_SAMPLE_COUNT, ze / AXIS_SAMPLE_COUNT);
					sh = sh + sh_evaluate(dir) * baked[i++];
				}
			}
			sh = sh * (4.f * M_PI / (AXIS_SAMPLE_COUNT * AXIS_SAMPLE_COUNT) * 0.2821f);
			const auto mult = std::max(std::min(p.half_size.y * 1.8f - 0.2f, 1.f), 0.f);
			data << p.pos.x;
			data << math::half(sh.x);
			data << math::half(mult * sh.y);
			data << math::half(mult * sh.z);
			data << math::half(mult * sh.w);
		}
		ao.extra_entries["TreeSamples.data"] = std::move(data);
	}

	baked_data bake_stuff(const std::shared_ptr<Node>& root)
	{
		// A quick check beforehand
		if (FEATURE_ACTIVE("REQUIRE_INTERIOR") && root->find_nodes(resolve_filter({"@COCKPIT_HR"})).empty())
		{
			std::cout << "Warning: interior nodes are missing.\n\t"
				"Baking AO without interior adjustments might result in interior being too dark.\n\t"
				"To fix issue, either add COCKPIT_HR or modify config." << std::endl;
		}

		// Generating AO
		std::cout << "Main pass:" << std::endl;
		const auto root_scene = std::make_shared<Scene>(root);
		root_scene->receivers -= root->find_meshes(resolve_filter(bake_as_trees));
		root_scene->receivers -= root->find_meshes(resolve_filter(bake_as_grass));

		if (FEATURE_ACTIVE("EXCLUDE_POBJECTS"))
		{
			root_scene->receivers -= root->find_any_meshes(resolve_filter({"AC_POBJECT?"}));
		}

		auto ao = bake_scene(root_scene, cfg_bake, true);

		if (FEATURE_ACTIVE("REBAKE_DARK_OBJECTS"))
		{
			const auto dark_ao_threshold = config.get(section, "DARK_AO_THRESHOLD", 0.01f);
			const auto dark_vertices_share_threshold = config.get(section, "DARK_VERTICES_SHARE_THRESHOLD", 0.2f);

			std::vector<std::shared_ptr<Mesh>> receivers;
			for (const auto& p : ao.entries)
			{
				auto dark = 0U;
				for (auto i : p.second.main_set)
				{
					if (i.x < dark_ao_threshold)
					{
						++dark;
					}
				}
				if (float(dark) > float(p.second.main_set.size()) * dark_vertices_share_threshold && dark > 20)
				{
					receivers.push_back(p.first);
				}
			}

			if (!receivers.empty())
			{
				std::cout << "Rebaking " << receivers.size() << " dark " << (receivers.size() > 1 ? "objects:" : "object:") << std::endl;
				auto scene_r = std::make_shared<Scene>(std::move(receivers), root_scene->blockers);
				auto cfg_bake_dark = cfg_bake;
				cfg_bake_dark.num_rays = config.get(section, "DARK_RAYS_PER_SAMPLE", cfg_bake_dark.num_rays);
				cfg_bake_dark.min_samples_per_face = config.get(section, "DARK_MIN_SAMPLES_PER_FACE", cfg_bake_dark.min_samples_per_face);
				ao.replace(bake_wrap::bake_scene(scene_r, root_scene->blockers.full, cfg_bake_dark, true));
			}
		}

		// Applying trees shadow factor
		const auto trees_shadow_factor = config.get(section, "TREES_SHADOW_FACTOR", 1.f);
		if (!bake_as_trees.empty() && trees_shadow_factor != 1.f)
		{
			PERF("Special: applying trees shadow factor")
			const auto scene_without_trees = std::make_shared<Scene>(*root_scene);
			scene_without_trees->blockers -= root->find_meshes(resolve_filter(bake_as_trees));
			ao.max(bake_scene(scene_without_trees, cfg_bake), 1.f - trees_shadow_factor);
		}

		// Generating AO for trees
		if (!bake_as_trees.empty())
		{
			PERF("Special: baking trees")
			const auto trees_scene = std::make_shared<Scene>(root);
			auto trees_meshes = root->find_meshes(resolve_filter(bake_as_trees));
			trees_meshes -= root->find_meshes(resolve_filter(bake_as_grass));
			auto trees_cfg_bake = cfg_bake;
			trees_cfg_bake.scene_offset_scale_horizontal = 2.f;
			trees_cfg_bake.scene_offset_scale_vertical = 0.5f;
			trees_cfg_bake.disable_normals = true;
			trees_cfg_bake.missing_normals_up = true;
			trees_cfg_bake.use_ground_plane_blocker = false;
			trees_cfg_bake.filter_mode = VERTEX_FILTER_AREA_BASED;
			trees_scene->receivers = trees_meshes;
			// trees_scene->blockers -= trees_meshes;
			ao.replace(bake_scene(trees_scene, trees_cfg_bake));
		}

		// Generating AO for grass
		if (!bake_as_grass.empty())
		{
			baked_data grass_ao;
			const auto grass_scene = std::make_shared<Scene>(root);
			auto grass_cfg_bake = cfg_bake;
			{
				PERF("Special: baking grass")
				const auto grass_meshes = root->find_meshes(resolve_filter(bake_as_grass));
				grass_cfg_bake.scene_offset_scale_horizontal = 0.5f;
				grass_cfg_bake.scene_offset_scale_vertical = 0.5f;
				grass_cfg_bake.disable_normals = true;
				grass_cfg_bake.missing_normals_up = true;
				grass_cfg_bake.use_ground_plane_blocker = false;
				grass_cfg_bake.filter_mode = VERTEX_FILTER_AREA_BASED;
				grass_scene->receivers = grass_meshes;
				grass_ao = bake_scene(grass_scene, grass_cfg_bake);
			}
			if (!bake_as_trees.empty() && trees_shadow_factor != 1.f)
			{
				PERF("Special: baking grass, applying trees shadow factor")
				grass_scene->blockers -= root->find_meshes(resolve_filter(bake_as_trees));
				grass_ao.max(bake_scene(grass_scene, grass_cfg_bake), 1.f - trees_shadow_factor);
			}
			ao.replace(grass_ao);
		}

		// Generating AO for AI lane
		bake_extra_samples(ao, root);

		// Generating AO for procedural trees
		if (!trees.empty() && FEATURE_ACTIVE("PROCEDURAL_TREES_BAKE"))
		{
			bake_procedural_trees(ao, root, trees, trees_key);
		}

		// Generating AO for driver
		if (FEATURE_ACTIVE("DRIVER_RECEIVES_SHADOWS") && driver_root)
		{
			ao.extend(bake_driver_shadows());
		}

		// Special options		
		if (FEATURE_ACTIVE("BAKE_BLURRED") && blurred_objects.any(input_file))
		{
			auto blurred = root->find_nodes(resolve_filter(blurred_objects.blurred_names));
			PERF("Special: blurred objects: " + std::to_string(blurred.size()));
			root->set_active(resolve_filter(blurred_objects.blurred_names), true);
			root->set_active(resolve_filter(blurred_objects.static_names), false);
			ao.extend(bake_scene(std::make_shared<Scene>(blurred), root_scene->blockers, cfg_bake));
			root->set_active(resolve_filter(blurred_objects.blurred_names), false);
			root->set_active(resolve_filter(blurred_objects.static_names), true);
		}

		if (FEATURE_ACTIVE("BAKE_COCKPIT_LR") && !root->find_nodes(resolve_filter(names_lr)).empty())
		{
			PERF("Special: baking low-res cockpit");
			root->set_active(resolve_filter(names_lr), true);
			root->set_active(resolve_filter(names_hr), false);
			ao.extend(bake_scene(std::make_shared<Scene>(root->find_nodes(resolve_filter(names_lr))), root_scene->blockers, cfg_bake));
			root->set_active(resolve_filter(names_lr), false);
			root->set_active(resolve_filter(names_hr), true);
		}

		if (FEATURE_ACTIVE("BAKE_SEATBELT"))
		{
			if (root->find_node(resolve_filter({"CINTURE_ON"})))
			{
				PERF("Special: baking seatbelt (on)")
				root->set_active(resolve_filter({"CINTURE_ON"}), true);
				auto seatbelt_on_blockers = root_scene->blockers;
				if (FEATURE_ACTIVE("DRIVER_CASTS_SHADOWS") && driver_root_scene)
				{
					seatbelt_on_blockers += driver_root_scene->blockers;
				}
				ao.extend(bake_scene(std::make_shared<Scene>(root->find_node(resolve_filter({"CINTURE_ON"}))), seatbelt_on_blockers, cfg_bake));
				root->set_active(resolve_filter({"CINTURE_ON"}), false);
			}
			if (root->find_node(resolve_filter({"CINTURE_OFF"})))
			{
				PERF("Special: baking seatbelt (off)")
				root->set_active(resolve_filter({"CINTURE_OFF"}), true);
				ao.extend(bake_scene(std::make_shared<Scene>(root->find_node(resolve_filter({"CINTURE_OFF"}))), root_scene->blockers, cfg_bake));
				root->set_active(resolve_filter({"CINTURE_OFF"}), false);
			}
		}

		// Applying animations
		if (!animation_instances.empty())
		{
			bake_animations(root, {root}, root_scene->blockers, cfg_bake, ao);

			if (FEATURE_ACTIVE("DRIVER_RECEIVES_SHADOWS") && driver_root)
			{
				root->set_active(resolve_filter({"CINTURE_ON"}), true);
				auto scene_with_seatbelts = std::make_shared<Scene>(root);
				bake_animations(root, {driver_root}, scene_with_seatbelts->blockers + driver_root_scene->blockers, cfg_bake, ao);
				root->set_active(resolve_filter({"CINTURE_ON"}), false);
			}

			if (FEATURE_ACTIVE("BAKE_COCKPIT_LR") && root->find_node(resolve_filter({"@COCKPIT_LR"})))
			{
				root->set_active(resolve_filter({"@COCKPIT_LR"}), true);
				root->set_active(resolve_filter({"@COCKPIT_HR"}), false);
				bake_animations(root, {root->find_node(resolve_filter({"@COCKPIT_LR"}))}, root_scene->blockers, cfg_bake, ao, "low-res cockpit");
				root->set_active(resolve_filter({"@COCKPIT_LR"}), false);
				root->set_active(resolve_filter({"@COCKPIT_HR"}), true);
			}

			if (FEATURE_ACTIVE("BAKE_SEATBELT"))
			{
				if (root->find_node(resolve_filter({"CINTURE_ON"})))
				{
					root->set_active(resolve_filter({"CINTURE_ON"}), true);
					bake_animations(root, {root->find_node(resolve_filter({"CINTURE_ON"}))}, root_scene->blockers, cfg_bake, ao, "seatbelt on");
					root->set_active(resolve_filter({"CINTURE_ON"}), false);
				}
				if (root->find_node(resolve_filter({"CINTURE_OFF"})))
				{
					root->set_active(resolve_filter({"CINTURE_OFF"}), true);
					bake_animations(root, {root->find_node(resolve_filter({"CINTURE_OFF"}))}, root_scene->blockers, cfg_bake, ao, "seatbelt off");
					root->set_active(resolve_filter({"CINTURE_OFF"}), false);
				}
			}
		}

		// Special case for rotating meshes
		if (!rotating_x.empty() || !rotating_y.empty() || !rotating_z.empty())
		{
			bake_rotating(root, ao, cfg_bake, rotating_mixing);
			if (FEATURE_ACTIVE("BAKE_BLURRED") && blurred_objects.any(input_file))
			{
				root->set_active(resolve_filter(blurred_objects.blurred_names), true);
				root->set_active(resolve_filter(blurred_objects.static_names), false);
				bake_rotating(root, ao, cfg_bake, rotating_mixing, "blurred meshes");
				root->set_active(resolve_filter(blurred_objects.blurred_names), false);
				root->set_active(resolve_filter(blurred_objects.static_names), true);
			}
		}

		if (FEATURE_ACTIVE("SPLIT_AO"))
		{
			const auto steering_wheel = root->find_nodes(resolve_filter({"STEER_HR", "STEER_LR"}));
			if (!steering_wheel.empty())
			{
				PERF("Special: baking steering wheel for AO splitting")

				auto steering_wheel_scene = std::make_shared<Scene>(steering_wheel);
				steering_wheel_scene->receivers = root->find_any_meshes(resolve_filter({"STEER_HR", "STEER_LR"}));

				// Baking turned by 180° and setting it as primary AO:
				for (const auto& n : steering_wheel)
				{
					const auto axis = float3{0.f, 0.f, 1.f};
					n->matrix_local = n->matrix_local_orig * NodeTransformation::rotation(180.f, &axis.x);
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
				for (const auto& n : steering_wheel)
				{
					n->update_matrix();
					n->resolve_skinned();
				}
				ao.max(bake_scene(steering_wheel_scene, root_scene->blockers, cfg_bake), 0.67f);
			}
		}

		// Optionally include LODs
		std::vector<std::shared_ptr<Node>> lod_roots{root};
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
					add_materials(exterior_materials, {lod_root}, lod_root->find_nodes(resolve_filter(names_hr + names_lr)));
				}
				if (interior_mult != 1.f)
				{
					add_materials(interior_materials, lod_root->find_nodes(resolve_filter(names_hr + names_lr)), {});
				}

				auto lod_cfg_bake = cfg_bake;
				if (lod_in > 30.f)
				{
					if (FEATURE_ACTIVE("DISTANT_LODS_INCREASE_OFFSET"))
					{
						lod_cfg_bake.scene_offset_scale_vertical = 0.2f;
						lod_cfg_bake.scene_offset_scale_horizontal = 0.2f;
					}

					if (FEATURE_ACTIVE("DISTANT_LODS_AREA_BASED"))
					{
						lod_cfg_bake.filter_mode = VERTEX_FILTER_AREA_BASED;
					}

					if (FEATURE_ACTIVE("DISTANT_LODS_SKIP_WHEELS"))
					{
						lod_root->set_active(resolve_filter({"WHEEL_LF", "WHEEL_LR", "WHEEL_RF", "WHEEL_RR"}), false);
						for (auto w : {"WHEEL_LF", "WHEEL_LR", "WHEEL_RF", "WHEEL_RR"})
						{
							const auto& node = lod_root->find_node(resolve_filter({w}));
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

				const auto lod_scene = std::make_shared<Scene>(lod_root);

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
						lod_root->set_active(resolve_filter(blurred_objects.blurred_names), true);
						lod_root->set_active(resolve_filter(blurred_objects.static_names), false);
						bake_rotating(lod_root, ao, lod_cfg_bake, rotating_mixing, "", false);
						lod_root->set_active(resolve_filter(blurred_objects.blurred_names), false);
						lod_root->set_active(resolve_filter(blurred_objects.static_names), true);
					}
				}

				resulting_name = "main_geometry.vao-patch";
			}
		}

		// Generating alternative AO with driver shadows
		if (FEATURE_ACTIVE("DRIVER_CASTS_SHADOWS") && driver_root)
		{
			Animation::apply_all(driver_root, driver_animations, 0.f);
			Animation::apply_all(driver_root, driver_steer_animations, 0.5f);
			driver_root->update_matrix();
			driver_root->resolve_skinned();

			const auto driver_shadow_targets = root->find_nodes(resolve_filter({"@COCKPIT_HR", "@COCKPIT_LR"}));
			const auto driver_shadow_blockers = std::make_shared<Scene>(root)->blockers + std::make_shared<Scene>(driver_root)->blockers;
			const auto driver_shadow_targets_scene = std::make_shared<Scene>(driver_shadow_targets);
			for (const auto& n : root->find_nodes(resolve_filter({"PAD_UP", "PAD_DOWN", "STEER_HR", "STEER_LR"})))
			{
				driver_shadow_targets_scene->receivers -= n->get_meshes();
			}

			auto ao_alt = bake_scene(driver_shadow_targets_scene, driver_shadow_blockers, cfg_bake, true);
			bake_animations(root, driver_shadow_targets, driver_shadow_blockers, cfg_bake, ao_alt);
			ao.set_alternative_set(ao_alt);
		}

		const auto car_mode = is_car(input_file);
		for (auto i = 0, j = 100; i < j; i++)
		{
			std::vector<std::string> names;
			auto mult = config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_MULT", 1.f);
			auto offset = config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_OFFSET", 0.f);
			auto exp = config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_GAMMA",
				config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_EXP", 1.f));
			auto primary_only = config.get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_PRIMARY_ONLY", false);
			if (config.try_get(section, "EXTRA_ADJUSTMENT_" + std::to_string(i) + "_NAMES", names)
				&& (mult != 1.f || offset != 0.f || exp != 1.f))
			{
				j = std::min(i + 100, 10000);

				std::vector<std::shared_ptr<Mesh>> dimmed_meshes;
				for (const auto& r : lod_roots)
				{
					dimmed_meshes += r->find_meshes(resolve_filter(names));
					if (car_mode)
					{
						for (const auto& node : r->find_nodes(resolve_filter(names)))
						{
							dimmed_meshes += node->get_meshes();
						}
					}
				}

				{
					dimmed_meshes += driver_root->find_meshes(resolve_filter(names));
					if (car_mode)
					{
						for (const auto& node : driver_root->find_nodes(resolve_filter(names)))
						{
							dimmed_meshes += node->get_meshes();
						}
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
						for (auto& e : f->second.alternative_set)
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

static LONG CALLBACK unhandled_handler(EXCEPTION_POINTERS* e)
{
	utils::make_minidump(e);
	return EXCEPTION_EXECUTE_HANDLER;
}

int main(int argc, const char** argv)
{
	#ifndef REPLACEMENT_OPTIMIZATION_MODE
	SetUnhandledExceptionFilter(unhandled_handler);
	#endif

	#ifdef USE_TRYCATCH
	try
	{
		#endif
		SetConsoleOutputCP(CP_UTF8);

		#ifdef REPLACEMENT_OPTIMIZATION_MODE

		for (auto i = 1; i < argc; i++)
		{
			replacement_optimization(argv[i]);
		}
		
		#else

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
				file_processor(utils::path(filename)).run();
			}
		}
		else
		{
			for (auto i = 1; i < argc; i++)
			{
				file_processor(argv[i]).run();
			}
		}
		#endif

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
