#pragma once

#include <algorithm>
#include <memory>
#include <bake_api.h>

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
};

std::shared_ptr<bake::HierarchyNode> load_hierarchy(const utils::path& filename);
std::shared_ptr<bake::Node> load_model(const utils::path& filename, const load_params& params);
bake::Animation load_ksanim(const utils::path& filename, const std::shared_ptr<bake::Node>& root, bool include_static = false);

