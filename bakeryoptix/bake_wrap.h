#pragma once
#include <vector>
#include <bake_api.h>
#include <baked_data.h>

namespace bake {
	struct Scene;
}

struct bake_params {
	int num_samples;
	int min_samples_per_face;
	bool disable_normals;
	bool missing_normals_up;
	bool fix_incorrect_normals;
	bool debug_mode;

	bool sample_on_points;
	bake::Vec3 sample_offset;

	bool use_ground_plane_blocker;
	int ground_upaxis;
	float ground_scale_factor;
	float ground_offset_factor;

	int num_rays;
	int bounce_counts;
	float scene_offset_scale_horizontal;
	float scene_offset_scale_vertical;
	float scene_albedo;
	float trees_light_pass_chance;

	bake::VertexFilterMode filter_mode;
	float regularization_weight;
};

struct bake_wrap final
{
	static baked_data bake_scene(const std::shared_ptr<bake::Scene>& scene, const std::vector<std::shared_ptr<bake::Mesh>>& blockers, 
		const bake_params& config, bool verbose = false);
};
