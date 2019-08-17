#pragma once
#include <map>
#include <vector>
#include <memory>
#include <utils/ini_file.h>

namespace utils {
	class path;
}

namespace bake {
	struct Mesh;
}

typedef std::vector<float2> baked_data_mesh_set;

struct baked_data_mesh
{
	// And each set consists of primary and secondary AO for smooth transition.
	baked_data_mesh_set main_set;
	baked_data_mesh_set alternative_set;
};

struct save_params
{
	float averaging_threshold;
	float averaging_cos_threshold;
	float brightness;
	float gamma;
	float opacity;
	utils::ini_file extra_config;
	bool use_v4;
};

struct baked_data
{
	std::map<std::shared_ptr<bake::Mesh>, baked_data_mesh> entries;
	void save(const utils::path& destination, const save_params& params, bool store_secondary_set) const;
	void replace(const baked_data& b);
	void replace_primary(const baked_data& b);
	void max(const baked_data& b, float mult_b = 1.f, const std::vector<std::shared_ptr<bake::Mesh>>& inverse = {}, bool apply_to_both_sets = false);
	void average(const baked_data& b, float mult_b, float mult_base, const std::vector<std::shared_ptr<bake::Mesh>>& inverse = {}, bool apply_to_both_sets = false);
	void extend(const baked_data& b);
	void set_alternative_set(const baked_data& b);
	void fill(const std::shared_ptr<bake::Mesh>& mesh, float x);
};

