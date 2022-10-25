#pragma once

#include <algorithm>
#include <memory>
#include <bake_api.h>
#include <utils/filesystem.h>
#include <utils/ini_file.h>

inline void expand_bbox(float bbox_min[3], float bbox_max[3], const float v[3])
{
	for (size_t k = 0; k < 3; ++k)
	{
		bbox_min[k] = std::min(bbox_min[k], v[k]);
		bbox_max[k] = std::max(bbox_max[k], v[k]);
	}
}

namespace bake
{
	struct Mesh;
}

namespace utils {
	class path;
}

struct load_params
{
	float normals_bias{};
	std::vector<std::string> exclude_patch;
	std::vector<std::string> exclude_blockers;
	bool exclude_blockers_alpha_test{};
	bool exclude_blockers_alpha_blend{};

	utils::path car_configs_dir;
	utils::path track_configs_dir;
};

std::vector<utils::ini_file> collect_configs(const utils::path& filename, const load_params& params);
std::shared_ptr<bake::HierarchyNode> load_hierarchy(const utils::path& filename);
std::shared_ptr<bake::Node> load_model(const utils::path& filename, const load_params& params);
std::shared_ptr<bake::Animation> load_ksanim(const utils::path& filename, bool include_static = false);
std::vector<bake::AILanePoint> load_ailane(const utils::path& filename);
void replacement_optimization(const utils::path& filename);

