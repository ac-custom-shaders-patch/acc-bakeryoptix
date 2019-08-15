#pragma once
#include <bake_util.h>

namespace bake
{
	struct AOSamples;
	void generate_rays_host(unsigned int seed, int px, int py, int sqrt_passes, float scene_offset, const AOSamples& ao_samples, Ray* rays);
	void update_ao_host(size_t num_samples, float max_distance, const float* hits, float* ao);
}
