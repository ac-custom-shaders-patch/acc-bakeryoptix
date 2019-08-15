#include "bake_kernels__cpu.h"

#include <bake_kernels.h>
#include <bake_api.h>
#include <cuda/random.h>
#include <optixu/optixu_math_namespace.h>

namespace bake
{
	using optix::float3;

	inline int idiv_ceil(const int x, const int y)
	{
		return (x + y - 1) / y;
	}

	// Ray generation kernel
	void generate_rays_kernel(
		const size_t idx,
		const unsigned int base_seed,
		const int px,
		const int py,
		const int sqrt_passes,
		const float scene_offset,
		const float3* sample_normals,
		const float3* sample_face_normals,
		const float3* sample_positions,
		bake::Ray* rays)
	{
		const unsigned int tea_seed = (base_seed << 16) | (px * sqrt_passes + py);
		unsigned seed = tea<2>(tea_seed, unsigned(idx));

		const float3 sample_norm = sample_normals[idx];
		const float3 sample_face_norm = sample_face_normals[idx];
		const float3 sample_pos = sample_positions[idx];
		const float3 ray_origin = sample_pos + scene_offset * sample_norm;
		optix::Onb onb(sample_norm);

		float3 ray_dir;
		float u0 = (static_cast<float>(px) + rnd(seed)) / static_cast<float>(sqrt_passes);
		float u1 = (static_cast<float>(py) + rnd(seed)) / static_cast<float>(sqrt_passes);
		int j = 0;
		do
		{
			optix::cosine_sample_hemisphere(u0, u1, ray_dir);

			onb.inverse_transform(ray_dir);
			++j;
			u0 = rnd(seed);
			u1 = rnd(seed);
		}
		while (j < 5 && optix::dot(ray_dir, sample_face_norm) <= 0.0f);

		rays[idx].origin = ray_origin;
		rays[idx].direction = ray_dir;
	}

	void generate_rays_host(unsigned seed, int px, int py, int sqrt_passes, float scene_offset, const AOSamples& ao_samples, Ray* rays)
	{
		#pragma omp parallel for
		for (auto i = 0U; i < ao_samples.num_samples; i++)
		{
			generate_rays_kernel(i, seed, px, py, sqrt_passes, scene_offset, (float3*)ao_samples.sample_normals, 
				(float3*)ao_samples.sample_face_normals, (float3*)ao_samples.sample_positions, rays);
		}
	}

	void update_ao_kernel(size_t idx, float max_distance, const float* hit_data, float* ao_data)
	{
		ao_data[idx] += hit_data[idx] > 0.0 && hit_data[idx] < max_distance ? 1.0f : 0.0f;
	}

	void update_ao_host(size_t num_samples, float max_distance, const float* hits, float* ao)
	{
		#pragma omp parallel for
		for (auto i = 0U; i < num_samples; i++)
		{
			update_ao_kernel(i, max_distance, hits, ao);
		}
	}
}
