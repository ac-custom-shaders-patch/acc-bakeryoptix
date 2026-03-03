#pragma once
#include <bake_api.h>

namespace bake
{
	void ao_embree(const std::vector<Mesh*>& blockers,
		const AOSamples& ao_samples, int rays_per_sample, float albedo, uint32_t bounce_counts,
		float scene_offset_scale_horizontal, float scene_offset_scale_vertical, float trees_light_pass_chance,
		bool debug_mode, float* ao_values);
}
