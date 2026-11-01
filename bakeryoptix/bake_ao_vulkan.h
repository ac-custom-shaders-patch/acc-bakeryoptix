#pragma once
#include <bake_api.h>

namespace bake
{
	const char* vulkan_configure();
	bool vulkan_available();

	// GPU-accelerated AO baking using Vulkan ray tracing (VK_KHR_ray_query).
	// Falls back to ao_embree() automatically if Vulkan initialization fails at runtime.
	// Dark samples refinement and return value are the same as with ao_embree().
	size_t ao_vulkan(const std::vector<Mesh*>& blockers,
		const AOSamples& ao_samples, int rays_per_sample, float albedo, uint32_t bounce_counts,
		float scene_offset_scale_horizontal, float scene_offset_scale_vertical, float trees_light_pass_chance,
		int dark_rays_per_sample, float dark_threshold, bool debug_mode, float* ao_values);
}
