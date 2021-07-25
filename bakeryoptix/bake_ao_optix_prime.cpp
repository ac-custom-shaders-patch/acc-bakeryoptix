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

#include <algorithm>
#include <map>

#include <bake_ao_optix_prime.h>
#include <bake_kernels.h>
#include <bake_util.h>

#include <cuda/buffer.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixpp_namespace.h>
#include <optix.h>
#include <bake_kernels__cpu.h>
#include <chrono>
#include <ptx_programs.h>
#include <utils/cout_progress.h>
#include <utils/string_operations.h>

using namespace optix;

inline size_t idiv_ceil(const size_t x, const size_t y)
{
	return (x + y - 1) / y;
}

Geometry ao_optix_geometry(Handle<ContextObj>& ctx, bake::Mesh* mesh, const Program& bb, const Program& intersection)
{
	optix::Geometry geometry(nullptr);

	try
	{
		geometry = ctx->createGeometry();

		optix::Buffer attributesBuffer = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		attributesBuffer->setElementSize(sizeof(bake::MeshVertex));
		attributesBuffer->setSize(mesh->vertices.size());

		void* dst = attributesBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
		memcpy(dst, mesh->vertices.data(), sizeof(bake::MeshVertex) * mesh->vertices.size());
		attributesBuffer->unmap();

		optix::Buffer indicesBuffer = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, mesh->triangles.size());
		dst = indicesBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
		memcpy(dst, mesh->triangles.data(), sizeof(optix::uint3) * mesh->triangles.size());
		indicesBuffer->unmap();

		geometry->setBoundingBoxProgram(bb);
		geometry->setIntersectionProgram(intersection);

		geometry["attributesBuffer"]->setBuffer(attributesBuffer);
		geometry["indicesBuffer"]->setBuffer(indicesBuffer);
		geometry->setPrimitiveCount(uint32_t(mesh->triangles.size()));
	}
	catch (optix::Exception& e)
	{
		std::cerr << e.getErrorString() << std::endl;
	}
	return geometry;
}

optix::Geometry create_plane(Handle<ContextObj>& ctx, const Program& bb, const Program& intersection)
{
	static bake::Mesh mesh;
	mesh.triangles.push_back(bake::Triangle{0, 1, 2});
	mesh.triangles.push_back(bake::Triangle{0, 2, 3});
	mesh.vertices.push_back({bake::Vec3{-1.f, 0.f, -1.f}, bake::Vec2{}});
	mesh.vertices.push_back({bake::Vec3{1.f, 0.f, -1.f}, bake::Vec2{}});
	mesh.vertices.push_back({bake::Vec3{1.f, 0.f, 1.f}, bake::Vec2{}});
	mesh.vertices.push_back({bake::Vec3{-1.f, 0.f, 1.f}, bake::Vec2{}});
	return ao_optix_geometry(ctx, &mesh, bb, intersection);
}

static std::string m_builder = std::string("Trbvh");

void set_acceleration_properties(optix::Acceleration acceleration)
{
	// To speed up the acceleration structure build for triangles, skip calls to the bounding box program and
	// invoke the special splitting BVH builder for indexed triangles by setting the necessary acceleration properties.
	// Using the fast Trbvh builder which does splitting has a positive effect on the rendering performanc as well.
	if (m_builder == std::string("Trbvh") || m_builder == std::string("Sbvh"))
	{
		// This requires that the position is the first element and it must be float x, y, z.
		acceleration->setProperty("vertex_buffer_name", "attributesBuffer");
		acceleration->setProperty("vertex_buffer_stride", "24");

		acceleration->setProperty("index_buffer_name", "indicesBuffer");
		acceleration->setProperty("index_buffer_stride", "12");
	}
}

Program load_program(Handle<ContextObj>& ctx, const char* name, const std::string& program)
{
	// std::cout << "Loading shader: " << name << std::endl;
	// return ctx->createProgramFromPTXString(program, name);
	return ctx->createProgramFromPTXFile(std::string("C:/Development/acc-bakeryoptix/x64/Release/") + name + ".cu.ptx", name);
}

struct cuda_textures_loader
{
	Context& ctx;
	std::map<const dds_loader*, TextureSampler> loaded;

	cuda_textures_loader(Context& ctx) : ctx(ctx) { }

	TextureSampler load_texture(const dds_loader& loader)
	{
		const auto f = loaded.find(&loader);
		if (f != loaded.end())
		{
			return f->second;
		}

		optix::Buffer pixel_buffer = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, loader.width, loader.height);
		void* buffer_data = pixel_buffer->map();
		std::memcpy(buffer_data, (*loader.data).data(), loader.width * loader.height * 4);
		pixel_buffer->unmap();

		// create texture sampler
		TextureSampler sampler = ctx->createTextureSampler();
		sampler->setWrapMode(0, RT_WRAP_REPEAT);
		sampler->setWrapMode(1, RT_WRAP_REPEAT);
		sampler->setWrapMode(2, RT_WRAP_REPEAT);
		sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
		sampler->setMaxAnisotropy(1.0f);
		sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
		sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NEAREST);
		sampler->setBuffer(pixel_buffer);
		return loaded[&loader] = sampler;
	}
};

void bake::ao_optix_prime(const std::vector<Mesh*>& blockers, const AOSamples& ao_samples, const int rays_per_sample, const float albedo, const uint bounce_counts,
	const float scene_offset_scale_horizontal, const float scene_offset_scale_vertical, const float trees_light_pass_chance, float* ao_values)
{
	auto ctx = Context::create();
	// ctx->setDevices(devices.begin(), devices.end())

	ctx->setEntryPointCount(1); // 0 = render
	ctx->setRayTypeCount(1);    // 0 = radiance;
	ctx->setStackSize(1024);
	//ctx->setPrintEnabled(true);
	//ctx->setExceptionEnabled(RT_EXCEPTION_ALL, true);

	cuda_textures_loader textures_loader(ctx);
	const auto bb = load_program(ctx, "raybb", ptx_program_raybb());
	const auto intersection = load_program(ctx, "rayintersection", ptx_program_rayintersection());
	const auto hit = load_program(ctx, "rayhit", ptx_program_rayhit());
	const auto anyhit = load_program(ctx, "rayanyhit", ptx_program_rayanyhit());
	const auto anyhit_tree = load_program(ctx, "rayanyhit_tree", ptx_program_rayanyhit_tree());

	auto mat_opaque = ctx->createMaterial();
	mat_opaque->setClosestHitProgram(0, hit);

	auto mat_alphatest = ctx->createMaterial();
	mat_alphatest->setAnyHitProgram(0, anyhit);
	mat_alphatest->setClosestHitProgram(0, hit);

	auto mat_foliage = ctx->createMaterial();
	mat_foliage->setAnyHitProgram(0, anyhit_tree);
	mat_foliage->setClosestHitProgram(0, hit);

	auto scene_accell = ctx->createAcceleration(m_builder); // No need to set acceleration properties on the top level Acceleration.
	auto scene_root = ctx->createGroup();                   // The scene's root group nodes becomes the sysTopObject.
	scene_root->setAcceleration(scene_accell);
	ctx["sysTopObject"]->set(scene_root); // This is where the rtTrace calls start the BVH traversal. (Same for radiance and shadow rays.)

	optix::Geometry plane_geometry = create_plane(ctx, bb, intersection);
	optix::GeometryInstance plane_instance = ctx->createGeometryInstance(); // This connects Geometries with Materials.
	plane_instance->setGeometry(plane_geometry);
	plane_instance->setMaterialCount(1);
	plane_instance->setMaterial(0, mat_opaque);
	plane_instance["parMaterialAlbedo"]->setFloat(0.2f);
	optix::Acceleration plane_accel = ctx->createAcceleration(m_builder);
	set_acceleration_properties(plane_accel);
	// This connects GeometryInstances with Acceleration structures. (All OptiX nodes with "Group" in the name hold an Acceleration.)
	optix::GeometryGroup plane_group = ctx->createGeometryGroup();
	plane_group->setAcceleration(plane_accel);
	plane_group->setChildCount(1);
	plane_group->setChild(0, plane_instance);

	// The original object coordinates of the plane have unit size, from -1.0f to 1.0f in x-axis and z-axis.
	// Scale the plane to go from -5 to 5.
	float plane_matrix_data[16] = {
		5.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 5.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 5.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	optix::Matrix4x4 plane_matrix(plane_matrix_data);
	optix::Transform plane_transform = ctx->createTransform();
	plane_transform->setChild(plane_group);
	plane_transform->setMatrix(false, plane_matrix.getData(), plane_matrix.inverse().getData());

	// Add the transform node placeing the plane to the scene's root Group node.
	auto children_added = 0U;
	scene_root->setChildCount(uint32_t(blockers.size()) + 1U);
	scene_root->setChild(children_added++, plane_transform);
	// m_rootGroup->setChildCount(blockers.size());

	for (auto m : blockers)
	{
		optix::Acceleration mesh_accel = ctx->createAcceleration(m_builder);
		set_acceleration_properties(mesh_accel);

		optix::GeometryInstance mesh_instance = ctx->createGeometryInstance(); // This connects Geometries with Materials.
		mesh_instance->setGeometry(ao_optix_geometry(ctx, m, bb, intersection));
		mesh_instance->setMaterialCount(1);

		if (m->material && m->material->texture && m->material->texture->data)
		{
			const auto is_foliage = utils::starts_with(m->material->shader, "ksTree");
			mesh_instance->setMaterial(0, is_foliage ? mat_foliage : mat_alphatest);
			mesh_instance["parMaterialTexture"]->setTextureSampler(textures_loader.load_texture(*m->material->texture));

			const auto var = m->material->get_var_or_null("ksAlphaRef");
			mesh_instance["parMaterialAlphaRef"]->setFloat((var ? var->v1 : 0.5f) * 255.f);

			if (is_foliage)
			{
				mesh_instance["parMaterialPassChange"]->setFloat(trees_light_pass_chance);
			}

			//std::cout << (is_foliage ? "Foliage: " : "Alpha-test: ") << m->name << ", material: " << m->material->name << ", shader: " << m->material->shader
			//	<< ", alpha ref.: " << (var ? var->v1 : 0.5f) << "\n";
		}
		else
		{
			mesh_instance->setMaterial(0, mat_opaque);
		}
		mesh_instance["parMaterialAlbedo"]->setFloat(albedo);

		optix::GeometryGroup mesh_group = ctx->createGeometryGroup();
		// This connects GeometryInstances with Acceleration structures. (All OptiX nodes with "Group" in the name hold an Acceleration.)
		mesh_group->setAcceleration(mesh_accel);
		mesh_group->setChildCount(1);
		mesh_group->setChild(0, mesh_instance);

		optix::Matrix4x4 mesh_transform(m->matrix.data());
		optix::Transform mesh_parent = ctx->createTransform();
		mesh_parent->setChild(mesh_group);
		mesh_parent->setMatrix(false, mesh_transform.getData(), mesh_transform.inverse().getData());
		scene_root->setChild(children_added++, mesh_parent);
	}

	const auto sqrt_rays_per_sample = static_cast<int>(lroundf(sqrtf(static_cast<float>(rays_per_sample))));
	unsigned seed = 0;

	// Split sample points into batches
	const size_t batch_size = 2000000; // Note: fits on GTX 750 (1 GB) along with Hunter model
	const auto num_batches = std::max(idiv_ceil(ao_samples.num_samples, batch_size), size_t(1));

	cout_progress progress{num_batches * sqrt_rays_per_sample * sqrt_rays_per_sample};
	progress.report();

	ctx->setRayGenerationProgram(0, load_program(ctx, "raygeneration", ptx_program_raygeneration()));
	ctx->setMissProgram(0, load_program(ctx, "raymiss", ptx_program_raymiss()));

	for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++, seed++)
	{
		const auto sample_offset = batch_idx * batch_size;
		const auto num_samples = std::min(batch_size, ao_samples.num_samples - sample_offset);
		if (num_samples == 0) continue;

		optix::Buffer buffer_normals = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3);
		{
			// inNormalsBuffer->setElementSize(sizeof(float3));
			buffer_normals->setSize(num_samples);
			void* dst = buffer_normals->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
			memcpy(dst, &ao_samples.sample_normals[3 * sample_offset], sizeof(float3) * num_samples);
			buffer_normals->unmap();
			ctx["inNormalsBuffer"]->setBuffer(buffer_normals);
		}

		optix::Buffer buffer_face_normals = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3);
		{
			// inFaceNormalsBuffer->setElementSize(sizeof(float3));
			buffer_face_normals->setSize(num_samples);
			void* dst = buffer_face_normals->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
			memcpy(dst, &ao_samples.sample_face_normals[3 * sample_offset], sizeof(float3) * num_samples);
			buffer_face_normals->unmap();
			ctx["inFaceNormalsBuffer"]->setBuffer(buffer_face_normals);
		}

		optix::Buffer buffer_positions = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3);
		{
			// inPositionsBuffer->setElementSize(sizeof(float3));
			buffer_positions->setSize(num_samples);
			void* dst = buffer_positions->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
			memcpy(dst, &ao_samples.sample_positions[3 * sample_offset], sizeof(float3) * num_samples);
			buffer_positions->unmap();
			ctx["inPositionsBuffer"]->setBuffer(buffer_positions);
		}

		ctx["numSamples"]->setUint(uint32_t(num_samples));
		ctx["bounceCounts"]->setUint(bounce_counts);
		ctx["sceneOffsetHorizontal"]->setFloat(scene_offset_scale_horizontal);
		ctx["sceneOffsetVertical"]->setFloat(scene_offset_scale_vertical);
		ctx["baseSeed"]->setUint(seed);
		ctx["sqrtPasses"]->setInt(sqrt_rays_per_sample);

		auto m_bufferOutput = ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT);
		m_bufferOutput->setFormat(RT_FORMAT_FLOAT);
		m_bufferOutput->setSize(num_samples, 1);
		ctx["sysOutputBuffer"]->set(m_bufferOutput);

		for (auto i = 0; i < sqrt_rays_per_sample; ++i)
			for (auto j = 0; j < sqrt_rays_per_sample; ++j)
			{
				// auto i2 = int(float(sqrt_rays_per_sample) * powf(float(i) / float(sqrt_rays_per_sample), 0.4f));
				// generate_rays(use_cuda, seed, i, j, sqrt_rays_per_sample, scene_offset_scale, ao_samples_device, rays);

				ctx["px"]->setInt(i);
				ctx["py"]->setInt(j);
				ctx->launch(0, num_samples, 1);

				// update_ao(use_cuda, num_samples, scene_maxdistance_scale * scene_scale, hits, ao);
				// update_ao(use_cuda, num_samples, 1.f, hits, ao);
				progress.report();
			}

		void* dst = m_bufferOutput->map(0, RT_BUFFER_MAP_READ);
		memcpy(&ao_values[sample_offset], dst, sizeof(float) * num_samples);
		m_bufferOutput->unmap();
	}

	// Normalize
	for (size_t i = 0; i < ao_samples.num_samples; ++i)
	{
		ao_values[i] = ao_values[i] / float(rays_per_sample);
		// ao_values[i] = optix::clamp((1.0f - ao_values[i] / float(rays_per_sample)) * 1.2f, 0.f, 1.f);
	}

	ctx->destroy();
}
