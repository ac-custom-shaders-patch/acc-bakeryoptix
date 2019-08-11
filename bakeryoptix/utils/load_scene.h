#pragma once

#include <memory>
#include <bake_api.h>

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
std::shared_ptr<bake::Node> load_scene(const utils::path& filename, const load_params& params);
bake::Animation load_ksanim(const utils::path& filename, const std::shared_ptr<bake::Node>& root, bool include_static = false);

