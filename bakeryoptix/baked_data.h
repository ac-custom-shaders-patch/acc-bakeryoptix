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

typedef std::vector<float2> baked_data_mesh;

struct save_params
{
	float averaging_threshold;
	float averaging_cos_threshold;
	float brightness;
	float gamma;
	float opacity;
	utils::ini_file extra_config;
};

struct baked_data
{
	std::map<std::shared_ptr<bake::Mesh>, baked_data_mesh> entries;
	void save(const utils::path& destination, const save_params& params, bool store_secondary_set) const;
	void replace(const baked_data& b);
	void replace_primary(const baked_data& b);
	void max(const baked_data& b, float mult_b = 1.f);
	void average(const baked_data& b, float mult_b, float mult_base);
	void extend(const baked_data& b);
	void fill(const std::shared_ptr<bake::Mesh>& mesh, float x);
};

