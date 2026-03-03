#pragma once
#include <bake_api.h>

namespace bake
{
	const char* vulkan_configure();
	bool vulkan_available();

	// GPU-accelerated AO baking using Vulkan ray tracing (VK_KHR_ray_query).
	// Falls back to ao_embree() automatically if Vulkan initialization fails at runtime.
	void ao_vulkan(const std::vector<Mesh*>& blockers,
		const AOSamples& ao_samples, int rays_per_sample, float albedo, uint32_t bounce_counts,
		float scene_offset_scale_horizontal, float scene_offset_scale_vertical, float trees_light_pass_chance,
		bool debug_mode, float* ao_values);
}
