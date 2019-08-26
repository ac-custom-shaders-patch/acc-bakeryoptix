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
#include <optix_prime/optix_primepp.h>
#include <bake_kernels__cpu.h>
#include <chrono>
#include <utils/cout_progress.h>

using namespace optix;
using namespace prime;

namespace
{
	void create_instances(Context& context,
		const std::vector<bake::Mesh*>& meshes,
		const bool use_cuda,
		// output, to keep allocations around
		std::vector<Buffer<float3>*>& allocated_vertex_buffers, std::vector<Buffer<int3>*>& allocated_index_buffers,
		std::vector<Model>& models, std::vector<RTPmodel>& prime_instances, std::vector<Matrix4x4>& transforms)
	{
		const auto buffer_type = use_cuda ? RTP_BUFFER_TYPE_CUDA_LINEAR : RTP_BUFFER_TYPE_HOST;
		const auto copy_type = use_cuda ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;

		// For sharing identical buffers between Models
		std::map<float*, Buffer<float3>*> unique_vertex_buffers;
		std::map<unsigned int*, Buffer<int3>*> unique_index_buffers;

		const auto model_offset = models.size();
		models.reserve(models.size() + meshes.size());
		for (auto mesh : meshes)
		{
			auto model = context->createModel();

			// Allocate or reuse vertex buffer and index buffer
			Buffer<float3>* vertex_buffer = nullptr;
			if (unique_vertex_buffers.find(&mesh->vertices[0].x) != unique_vertex_buffers.end())
			{
				vertex_buffer = unique_vertex_buffers.find(&mesh->vertices[0].x)->second;
			}
			else
			{
				// Note: copy disabled for Buffer, so need pointer here
				vertex_buffer = new Buffer<float3>(mesh->vertices.size(), buffer_type, UNLOCKED);
				cudaMemcpy(vertex_buffer->ptr(), &mesh->vertices[0].x, vertex_buffer->size_in_bytes(), copy_type);
				unique_vertex_buffers[&mesh->vertices[0].x] = vertex_buffer;
				allocated_vertex_buffers.push_back(vertex_buffer);
			}

			Buffer<int3>* index_buffer = nullptr;
			if (unique_index_buffers.find(&mesh->triangles[0].a) != unique_index_buffers.end())
			{
				index_buffer = unique_index_buffers.find(&mesh->triangles[0].a)->second;
			}
			else
			{
				index_buffer = new Buffer<int3>(mesh->triangles.size(), buffer_type);
				cudaMemcpy(index_buffer->ptr(), &mesh->triangles[0].a, index_buffer->size_in_bytes(), copy_type);
				unique_index_buffers[&mesh->triangles[0].a] = index_buffer;
				allocated_index_buffers.push_back(index_buffer);
			}

			model->setTriangles(
				index_buffer->count(), index_buffer->type(), index_buffer->ptr(),
				vertex_buffer->count(), vertex_buffer->type(), vertex_buffer->ptr(), vertex_buffer->stride()
			);
			model->update(0);
			models.push_back(model); // Model is ref counted, so need to return it to prevent destruction
		}

		prime_instances.reserve(prime_instances.size() + meshes.size());
		for (auto i = 0U; i < meshes.size(); ++i)
		{
			const auto index = model_offset + i;
			auto rtp_model = models[index]->getRTPmodel();
			prime_instances.push_back(rtp_model);
			transforms.emplace_back(meshes[i]->matrix._Elems);
		}
	}

	inline size_t idiv_ceil(const size_t x, const size_t y)
	{
		return (x + y - 1) / y;
	}

	void generate_rays(bool use_cuda, unsigned seed, int i, int j, int sqrt_passes, float scene_offset, const bake::AOSamples& ao_samples, Buffer<bake::Ray>& rays)
	{
		if (use_cuda)
		{
			generate_rays_device(seed, i, j, sqrt_passes, scene_offset, ao_samples, rays.ptr());
		}
		else
		{
			generate_rays_host(seed, i, j, sqrt_passes, scene_offset, ao_samples, rays.ptr());
		}
	}

	void update_ao(bool use_cuda, size_t num_samples, float max_distance, Buffer<float>& hits, Buffer<float>& ao)
	{
		if (use_cuda)
		{
			bake::update_ao_device(num_samples, max_distance, hits.ptr(), ao.ptr());
		}
		else
		{
			bake::update_ao_host(num_samples, max_distance, hits.ptr(), ao.ptr());
		}
	}
}

void bake::ao_optix_prime(
	const std::vector<Mesh*>& blockers,
	const AOSamples& ao_samples,
	const int rays_per_sample,
	const float scene_offset_scale,
	const float scene_maxdistance_scale,
	const float* bbox_min,
	const float* bbox_max,
	float* ao_values,
	bool use_cuda)
{
	auto ctx = Context::create(use_cuda ? RTP_CONTEXT_TYPE_CUDA : RTP_CONTEXT_TYPE_CPU);

	std::vector<Model> models;
	std::vector<RTPmodel> prime_instances;
	std::vector<Matrix4x4> transforms;
	std::vector<Buffer<float3>*> allocated_vertex_buffers;
	std::vector<Buffer<int3>*> allocated_index_buffers;
	create_instances(ctx, blockers, use_cuda, allocated_vertex_buffers, allocated_index_buffers, models, prime_instances, transforms);

	auto scene_model = ctx->createModel();
	scene_model->setInstances(prime_instances.size(), RTP_BUFFER_TYPE_HOST, &prime_instances[0],
		RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4, RTP_BUFFER_TYPE_HOST, &transforms[0]);
	scene_model->update(0);

	auto query = scene_model->createQuery(RTP_QUERY_TYPE_ANY);
	const auto sqrt_rays_per_sample = static_cast<int>(lroundf(sqrtf(static_cast<float>(rays_per_sample))));
	unsigned seed = 0;

	// Split sample points into batches
	const size_t batch_size = 2000000; // Note: fits on GTX 750 (1 GB) along with Hunter model
	const auto num_batches = std::max(idiv_ceil(ao_samples.num_samples, batch_size), size_t(1));
	const auto scene_scale = std::max(std::max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]), bbox_max[2] - bbox_min[2]);

	cout_progress progress{num_batches * sqrt_rays_per_sample * sqrt_rays_per_sample};
	progress.report();

	for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++, seed++)
	{
		const auto sample_offset = batch_idx * batch_size;
		const auto num_samples = std::min(batch_size, ao_samples.num_samples - sample_offset);
		if (num_samples == 0) continue;

		// Copy all necessary data to device
		const auto buffer_type = use_cuda ? RTP_BUFFER_TYPE_CUDA_LINEAR : RTP_BUFFER_TYPE_HOST;
		const auto copy_type = use_cuda ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
		Buffer<float3> sample_normals(num_samples, buffer_type);
		Buffer<float3> sample_face_normals(num_samples, buffer_type);
		Buffer<float3> sample_positions(num_samples, buffer_type);

		cudaMemcpy(sample_normals.ptr(), ao_samples.sample_normals + 3 * sample_offset, sample_normals.size_in_bytes(), copy_type);
		cudaMemcpy(sample_face_normals.ptr(), ao_samples.sample_face_normals + 3 * sample_offset, sample_face_normals.size_in_bytes(), copy_type);
		cudaMemcpy(sample_positions.ptr(), ao_samples.sample_positions + 3 * sample_offset, sample_positions.size_in_bytes(), copy_type);
		AOSamples ao_samples_device{};
		ao_samples_device.num_samples = num_samples;
		ao_samples_device.sample_normals = reinterpret_cast<float*>(sample_normals.ptr());
		ao_samples_device.sample_face_normals = reinterpret_cast<float*>(sample_face_normals.ptr());
		ao_samples_device.sample_positions = reinterpret_cast<float*>(sample_positions.ptr());
		ao_samples_device.sample_infos = nullptr;

		Buffer<float> hits(num_samples, buffer_type);
		Buffer<Ray> rays(num_samples, buffer_type);
		Buffer<float> ao(num_samples, buffer_type);
		cudaMemset(ao.ptr(), 0, ao.size_in_bytes());

		query->setRays(rays.count(), Ray::format, rays.type(), rays.ptr());
		query->setHits(hits.count(), RTP_BUFFER_FORMAT_HIT_T, hits.type(), hits.ptr());

		for (auto i = 0; i < sqrt_rays_per_sample; ++i)
			for (auto j = 0; j < sqrt_rays_per_sample; ++j)
			{
				generate_rays(use_cuda, seed, i, j, sqrt_rays_per_sample, scene_offset_scale, ao_samples_device, rays);
				query->execute(0);
				update_ao(use_cuda, num_samples, scene_maxdistance_scale * scene_scale, hits, ao);
				progress.report();
			}

		// Copy AO to ao_values
		cudaMemcpy(&ao_values[sample_offset], ao.ptr(), ao.size_in_bytes(), use_cuda ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
	}

	// Normalize
	for (size_t i = 0; i < ao_samples.num_samples; ++i)
	{
		ao_values[i] = 1.0f - ao_values[i] / float(rays_per_sample);
	}

	// Clean up Buffer pointers. Could be avoided with unique_ptr.
	for (auto& allocated_vertex_buffer : allocated_vertex_buffers)
	{
		delete allocated_vertex_buffer;
	}
	allocated_vertex_buffers.clear();
	for (auto& allocated_index_buffer : allocated_index_buffers)
	{
		delete allocated_index_buffer;
	}
	allocated_index_buffers.clear();
}
