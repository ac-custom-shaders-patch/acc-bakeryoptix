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

// #define USE_TRYCATCH

#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <Windows.h>

#include <app_config.h>
#include <utils/std_ext.h>
#include <utils/filesystem.h>
#include <utils/perf_moment.h>
#include <utils/load_scene.h>

#include <bake_api.h>
#include <bake_util.h>

#pragma comment(lib, "Shlwapi.lib")
#pragma comment(lib, "WinMM.lib")
#pragma comment(lib, "cudart_static.lib")
#pragma comment(lib, "optix_prime.6.0.0.lib")

void bake_animations(std::shared_ptr<bake::Node> root, baked_data& ao, const utils::path& dir, const std::vector<std::string>& names,
	const bake_params& params, int steps, float mix_multiplier)
{
	std::vector<bake::Animation> animations;
	for (const auto& name : names)
	{
		const auto loaded = load_ksanim(dir / name, root);
		if (!loaded.entries.empty())
		{
			animations.push_back(loaded);
		}
	}
	if (animations.empty()) return;
	std::cout << "Baking animations (" << animations.size() << " found):" << std::endl;
	for (auto i = 0; i < steps; i++)
	{
		const auto pos = float(i + 1) / float(steps);
		std::cout << (i == 0 ? "\t" : "\r\t") << "Position: " << round(pos * 100.f) << "%";
		for (const auto& a : animations)
		{
			a.apply(1.f);
		}
		ao.max(bake_wrap::bake_scene(std::make_shared<bake::Scene>(root), params), mix_multiplier);
	}
	std::cout << std::endl;
}

void bake_rotating(std::shared_ptr<bake::Node> root, baked_data& ao, const std::vector<std::string>& names,
	const bake_params& params, float step, bool average_ao)
{
	const auto root_scene = std::make_shared<bake::Scene>(root);
	std::vector<std::shared_ptr<bake::Node>> rotating;
	for (const auto& name : names)
	{
		auto node = root->find_node(name);
		if (node != nullptr)
		{
			rotating.push_back(node);
		}
	}
	if (rotating.empty()) return;
	std::cout << "Baking rotating objects (" << rotating.size() << " found):" << std::endl;
	const auto iterations = int(ceilf(359.f / step));
	for (auto i = 1; i < iterations; i++)
	{
		const auto deg = i * step;
		std::cout << (i == 1 ? "\t" : "\r\t") << "Degress: " << deg << utf16_to_utf8(L"°");
		for (const auto& n : rotating)
		{
			const auto is_wheel = n->name[0] == 'W';
			const auto axis = is_wheel ? float3{1.f, 0.f, 0.f} : float3{0.f, 0.f, 1.f};
			n->matrix_local = n->matrix_local_orig * bake::NodeTransformation::rotation(deg, &axis.x);
		}

		if (average_ao)
		{
			ao.average(bake_wrap::bake_scene(std::make_shared<bake::Scene>(rotating), root_scene->blockers, params),
				1.f / float(iterations), i == 1 ? 1.f / float(iterations) : 1.f);
		}
		else
		{
			ao.max(bake_wrap::bake_scene(std::make_shared<bake::Scene>(rotating), root_scene->blockers, params));
		}
	}
	std::cout << std::endl;
}

int main(int argc, const char** argv)
{
#ifdef USE_TRYCATCH
	try
	{
#endif
		SetConsoleOutputCP(CP_UTF8);
		const app_config config(argc, argv);

		// Load scene
		std::shared_ptr<bake::Node> root;
		{
			{
				PERF("Loading model `" + config.scene_filename.relative_ac() + "`");
				root = load_scene(config.scene_filename.string().c_str(), config.load);
			}

			// Print scene stats
			auto meshes = root->get_meshes();
			{
				size_t num_vertices = 0;
				size_t num_triangles = 0;
				for (const auto& mesh : meshes)
				{
					num_vertices += mesh->num_vertices;
					num_triangles += mesh->num_triangles;
				}
				std::cout << "\t" << meshes.size() << " meshes" << std::endl;
				std::cout << "\t" << num_vertices / 1000 << "K vertices, " << num_triangles / 1000 << "K triangles" << std::endl;
			}
		}

		// Generating AO
		std::cout << "Main pass:" << std::endl;
		auto ao = bake_wrap::bake_scene(std::make_shared<bake::Scene>(root), config.bake, true);

		// Applying animations
		bake_animations(root, ao, config.scene_filename.parent_path() / "animations", {"lights.ksanim", "car_door_L.ksanim", "car_door_R.ksanim"},
			config.bake, 2, 0.8f);

		// Special case for rotating meshes
		bake_rotating(root, ao, {"WHEEL_LF", "WHEEL_LR", "WHEEL_RF", "WHEEL_RR", "STEER_HR", "STEER_LR"},
			config.bake, 30.f, true);

		// Saving result
		{
			auto destination = utils::path(std_ext::replace_to(
				std_ext::replace_to(config.scene_filename.string(), ".ini", ".vao-patch"), ".kn5", ".vao-patch"));
			PERF("Saving to `" + destination.relative_ac() + "`");
			ao.save(destination, config.save);
		}

		return 0;
#ifdef USE_TRYCATCH
	}
	catch (const std::exception& e)
	{
		std::cerr << "Exception: " << e.what();
		return 1;
	}
#endif
}
