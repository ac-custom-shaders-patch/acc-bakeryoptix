#pragma once
#include <bake_api.h>

namespace bake
{
	// Samples coming out darker than dark_threshold are traced again with dark_rays_per_sample rays (if it’s
	// higher than rays_per_sample). Returns the number of samples refined that way.
	size_t ao_embree(const std::vector<Mesh*>& blockers,
		const AOSamples& ao_samples, int rays_per_sample, float albedo, uint32_t bounce_counts,
		float scene_offset_scale_horizontal, float scene_offset_scale_vertical, float trees_light_pass_chance,
		int dark_rays_per_sample, float dark_threshold, bool debug_mode, float* ao_values);
}
