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

void bake_rotating(std::shared_ptr<bake::Node> root, baked_data& ao,
	const std::vector<std::string>& names_x, const std::vector<std::string>& names_y, const std::vector<std::string>& names_z,
	const bake_params& params, const float step, const mixing_params& mixing, const std::string& comment = "")
{
	const auto root_scene = std::make_shared<bake::Scene>(root);
	const auto rotating_x = root->find_nodes(names_x);
	const auto rotating_y = root->find_nodes(names_y);
	const auto rotating_z = root->find_nodes(names_z);
	const auto rotating = rotating_x + rotating_y + rotating_z;
	if (rotating.empty()) return;

	std::cout << "Baking rotating objects (" << rotating.size() << (comment.empty() ? " found):" : " found, " + comment + "):") << std::endl;
	const auto iterations = int(ceilf(359.f / step));
	for (auto i = 1; i < iterations; i++)
	{
		const auto deg = i * step;
		std::cout << (i == 1 ? "\t" : "\r\t") << "Degress: " << deg << utf16_to_utf8(L"�");
		for (const auto& n : rotating_x)
		{
			const auto axis = float3{1.f, 0.f, 0.f};
			n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(deg, &axis.x);
		}
		for (const auto& n : rotating_y)
		{
			const auto axis = float3{0.f, 1.f, 0.f};
			n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(deg, &axis.x);
		}
		for (const auto& n : rotating_z)
		{
			const auto axis = float3{0.f, 0.f, 1.f};
			n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(deg, &axis.x);
		}
		const auto baked = bake_wrap::bake_scene(std::make_shared<bake::Scene>(rotating), root_scene->blockers, params);
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
	std::cout << std::endl;
}

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

void fill_animation(utils::ini_file& config, const bake::Animation& animation, const std::string& file)
{
	std::cout << "Animation `" << file << "` affects nodes: " << std_ext::join_to_string(
		apply(animation.entries, [](const bake::NodeTransition& t) { return t.node->name; }), ", ") << "\n";

	auto name_lc = utils::path(file).filename().string();
	std::transform(name_lc.begin(), name_lc.end(), name_lc.begin(), tolower);

	std::string key;
	if (file == "lights.ksanim")
	{
		key = "HEADLIGHTS_NODES";
	}
	else if (file == "car_door_l.ksanim")
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
				list_new += apply(animation.entries, [](const bake::NodeTransition& t) { return t.node->name; });
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
	std::vector<bake::Animation> animation_instances;

	float interior_mult = 1.f;
	float exterior_mult = 1.f;
	float driver_mult = 1.f;
	std::set<std::string> interior_materials;
	std::set<std::string> exterior_materials;
	std::set<std::string> driver_materials;

	std::shared_ptr<bake::Node> root;
	std::shared_ptr<bake::Node> driver_root;
	std::shared_ptr<bake::Scene> driver_root_scene;

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
			for (const auto& s : data_blurred.iterate("OBJECT"))
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

		auto optix_context = config.get("GENERAL", "OPTIX_CONTEXT", std::string("AUTO"));
		if (optix_context == "AUTO")
		{
			optix_context = is_cuda_available ? "CUDA" : "GPU";
		}
		cfg_bake.use_cuda = optix_context != "GPU";

		// Specialized options
		section = "MODE_" + mode;
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

		interior_mult = config.get(section, "BRIGHTER_AMBIENT_INTERIOR", 1.f);
		exterior_mult = config.get(section, "BRIGHTER_AMBIENT_EXTERIOR", 1.f);
		driver_mult = config.get(section, "BRIGHTER_AMBIENT_DRIVER", 1.f);

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
				load_hierarchy(driver_pos)->align(driver_root);
				if (FEATURE_ACTIVE("DRIVER_POSITION_WITH_STEER_ANIM"))
				{
					load_ksanim(input_file.parent_path() / "animations" / "steer.ksanim", driver_root, true).apply(0.5f);
				}
				driver_root_scene = std::make_shared<bake::Scene>(driver_root);
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
			const auto loaded = load_ksanim(input_file.parent_path() / "animations" / file, root);
			if (!loaded.entries.empty())
			{
				animation_instances.push_back(loaded);
				fill_animation(cfg_save.extra_config, loaded, file);
			}
		}

		if (FEATURE_ACTIVE("LOAD_WING_ANIMATIONS"))
		{
			const auto data_wing_animations = utils::ini_file(input_file.parent_path() / "data" / "wing_animations.ini");
			for (const auto& s : data_wing_animations.iterate("ANIMATION"))
			{
				const auto file = data_wing_animations.get(s, "FILE", std::string());
				const auto loaded = load_ksanim(input_file.parent_path() / "animations" / file, root);
				if (!loaded.entries.empty())
				{
					animation_instances.push_back(loaded);
					fill_animation(cfg_save.extra_config, loaded, file);
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

	void run()
	{
		// Optional dumping
		std::string dump_as;
		if (config.try_get(section, "DUMP_INSTEAD_OF_BAKING", dump_as) && !dump_as.empty())
		{
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
			auto ao_alt = bake_wrap::bake_scene(root_scene, driver_shadow_blockers, cfg_bake, true);
			bake_animations(driver_shadow_targets, driver_shadow_blockers, ao_alt);
			ao.set_alternative_set(ao_alt);
		}

		// Saving result
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

			const auto destination = input_file.parent_path() / resulting_name;
			PERF("Saving to `" + destination.relative_ac() + "`");
			ao.save(destination, cfg_save, FEATURE_ACTIVE("SPLIT_AO"));
		}
	}

	void bake_animations(const std::vector<std::shared_ptr<bake::Node>>& targets,
		const std::vector<std::shared_ptr<bake::Mesh>>& blockers,
		baked_data& ao, const std::string& comment = "")
	{
		if (animations.empty()) return;
		std::cout << "Baking animations (" << animations.size() << (comment.empty() ? " found):" : " found, " + comment + "):") << std::endl;
		for (auto i = 0U; i < animations_steps.size(); i++)
		{
			const auto pos = animations_steps[i];
			std::cout << (i == 0 ? "\t" : "\r\t") << "Position: " << round(pos * 100.f) << "%";
			for (const auto& a : animation_instances)
			{
				a.apply(pos);
			}

			root->update_matrix();
			root->resolve_skinned();

			const auto baked = bake_wrap::bake_scene(std::make_shared<bake::Scene>(targets), blockers, cfg_bake);
			if (animations_mixing.use_max)
			{
				ao.max(baked, animations_mixing.max_mult, animations_mixing_inverse);
			}
			else
			{
				const auto avg_mult = 1.f / float(animations_steps.size() + 1);
				ao.average(baked, avg_mult, i == 0 ? avg_mult : 1.f, animations_mixing_inverse);
			}
		}
		for (const auto& a : animation_instances)
		{
			a.apply(0.f);
		}
		std::cout << std::endl;
	}

	baked_data bake_driver_shadows()
	{
		root->set_active({"CINTURE_ON"}, true);
		const auto scene_with_seatbelts = std::make_shared<bake::Scene>(root);
		const auto result = bake_wrap::bake_scene(std::make_shared<bake::Scene>(driver_root), scene_with_seatbelts->blockers + driver_root_scene->blockers, cfg_bake);
		root->set_active({"CINTURE_ON"}, false);
		resulting_name = "main_geometry.vao-patch";
		return result;
	}

	baked_data bake_stuff(const std::shared_ptr<bake::Node>& root)
	{
		// Generating AO
		std::cout << "Main pass:" << std::endl;
		const auto root_scene = std::make_shared<bake::Scene>(root);

		auto ao = bake_wrap::bake_scene(root_scene, cfg_bake, true);

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
			ao.extend(bake_wrap::bake_scene(std::make_shared<bake::Scene>(blurred), root_scene->blockers, cfg_bake));
			root->set_active(blurred_objects.blurred_names, false);
			root->set_active(blurred_objects.static_names, true);
		}

		if (FEATURE_ACTIVE("BAKE_COCKPIT_LR") && !root->find_nodes(names_lr).empty())
		{
			PERF("Special: baking low-res cockpit");
			root->set_active(names_lr, true);
			root->set_active(names_hr, false);
			ao.extend(bake_wrap::bake_scene(std::make_shared<bake::Scene>(root->find_nodes(names_lr)), root_scene->blockers, cfg_bake));
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
				ao.extend(bake_wrap::bake_scene(std::make_shared<bake::Scene>(root->find_node("CINTURE_ON")), seatbelt_on_blockers, cfg_bake));
				root->set_active({"CINTURE_ON"}, false);
			}
			if (root->find_node("CINTURE_OFF"))
			{
				PERF("Special: baking seatbelt (off)")
				root->set_active({"CINTURE_OFF"}, true);
				ao.extend(bake_wrap::bake_scene(std::make_shared<bake::Scene>(root->find_node("CINTURE_OFF")), root_scene->blockers, cfg_bake));
				root->set_active({"CINTURE_OFF"}, false);
			}
		}

		// Applying animations
		if (!animation_instances.empty())
		{
			bake_animations({root}, root_scene->blockers, ao);

			if (FEATURE_ACTIVE("DRIVER_RECEIVES_SHADOWS"))
			{
				root->set_active({"CINTURE_ON"}, true);
				auto scene_with_seatbelts = std::make_shared<bake::Scene>(root);
				bake_animations({driver_root}, scene_with_seatbelts->blockers + driver_root_scene->blockers, ao);
				root->set_active({"CINTURE_ON"}, false);
			}

			if (FEATURE_ACTIVE("BAKE_COCKPIT_LR") && root->find_node("COCKPIT_LR"))
			{
				root->set_active({"COCKPIT_LR"}, true);
				root->set_active({"COCKPIT_HR"}, false);
				bake_animations({root->find_node("COCKPIT_LR")}, root_scene->blockers, ao, "low-res cockpit");
				root->set_active({"COCKPIT_LR"}, false);
				root->set_active({"COCKPIT_HR"}, true);
			}

			if (FEATURE_ACTIVE("BAKE_SEATBELT"))
			{
				if (root->find_node("CINTURE_ON"))
				{
					root->set_active({"CINTURE_ON"}, true);
					bake_animations({root->find_node("CINTURE_ON")}, root_scene->blockers, ao, "seatbelt on");
					root->set_active({"CINTURE_ON"}, false);
				}
				if (root->find_node("CINTURE_OFF"))
				{
					root->set_active({"CINTURE_OFF"}, true);
					bake_animations({root->find_node("CINTURE_OFF")}, root_scene->blockers, ao, "seatbelt off");
					root->set_active({"CINTURE_OFF"}, false);
				}
			}
		}

		// Special case for rotating meshes
		if (!rotating_x.empty() || !rotating_y.empty() || !rotating_z.empty())
		{
			bake_rotating(root, ao, rotating_x, rotating_y, rotating_z,
				cfg_bake, rotating_step, rotating_mixing);
			if (FEATURE_ACTIVE("BAKE_BLURRED") && blurred_objects.any(input_file))
			{
				root->set_active(blurred_objects.blurred_names, true);
				root->set_active(blurred_objects.static_names, false);
				bake_rotating(root, ao, rotating_x, rotating_y, rotating_z,
					cfg_bake, rotating_step, rotating_mixing, "blurred meshes");
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

				// Baking turned by 180� and setting it as primary AO:
				for (const auto& n : steering_wheel)
				{
					const auto axis = float3{0.f, 0.f, 1.f};
					n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(180.f, &axis.x);
				}
				ao.replace_primary(bake_wrap::bake_scene(std::make_shared<bake::Scene>(steering_wheel), root_scene->blockers, cfg_bake));

				// Turning wheel back to 0�, baking and mixing it in primary AO with multiplier set to 0.67:
				for (const auto& n : steering_wheel)
				{
					n->matrix_local = n->matrix_local_orig;
				}
				ao.max(bake_wrap::bake_scene(std::make_shared<bake::Scene>(steering_wheel), root_scene->blockers, cfg_bake), 0.67f);
			}
		}

		// Optionally include LODs
		std::vector<std::shared_ptr<bake::Node>> lod_roots{root};
		if (FEATURE_ACTIVE("INCLUDE_LODS"))
		{
			auto data_lods = utils::ini_file(input_file.parent_path() / "data" / "lods.ini");
			for (const auto& s : data_lods.iterate("LOD"))
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
					if (FEATURE_ACTIVE("DISTANT_LOGS_INCREASE_OFFSET"))
					{
						lod_cfg_bake.scene_offset_scale = 0.2f;
					}

					if (FEATURE_ACTIVE("DISTANT_LOGS_AREA_BASED"))
					{
						lod_cfg_bake.filter_mode = bake::VERTEX_FILTER_AREA_BASED;
					}

					if (FEATURE_ACTIVE("DISTANT_LOGS_SKIP_WHEELS"))
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
					ao.extend(bake_wrap::bake_scene(lod_scene, lod_cfg_bake));
				}

				if (!rotating_x.empty() || !rotating_y.empty() || !rotating_z.empty())
				{
					bake_rotating(lod_root, ao, rotating_x, rotating_y, rotating_z,
						lod_cfg_bake, rotating_step, rotating_mixing, "LOD `" + name + "`");
					if (FEATURE_ACTIVE("BAKE_BLURRED") && blurred_objects.any(input_file))
					{
						lod_root->set_active(blurred_objects.blurred_names, true);
						lod_root->set_active(blurred_objects.static_names, false);
						bake_rotating(lod_root, ao, rotating_x, rotating_y, rotating_z,
							lod_cfg_bake, rotating_step, rotating_mixing, "LOD `" + name + "`, blurred meshes");
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

int main(int argc, const char** argv)
{
	#ifdef USE_TRYCATCH
	try
	{
		#endif
		SetConsoleOutputCP(CP_UTF8);

		int cuda_version;
		const auto cuda_result = cudaRuntimeGetVersion(&cuda_version);
		if (cuda_result == cudaSuccess && cuda_version)
		{
			std::cout << "Installed CUDA: " << cuda_version / 1000 << "." << cuda_version / 10 % 100 << "\n";
		}
		else
		{
			cuda_version = 0;
			std::cout << "CUDA toolkit is not installed, consider reducing quality settings or expect baking to be orders of magnitude slower\n";
		}

		for (auto i = 1; i < argc; i++)
		{
			file_processor(argv[i], cuda_version > 0).run();
		}
		return 0;
		#ifdef USE_TRYCATCH
	}
	catch (const std::exception& e)
	{
		std::cerr << "Exception: " << e.what();
		std::getchar();
		return 1;
	}
	#endif
}
