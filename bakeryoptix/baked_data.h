#pragma once
#include <map>
#include <vector>
#include <memory>

namespace utils {
	class path;
}

namespace bake {
	struct Mesh;
}

typedef std::vector<float> baked_data_mesh;

struct save_params
{
	float averaging_threshold;
	float brightness;
	float gamma;
	float opacity;
};

struct baked_data
{
	std::map<std::shared_ptr<bake::Mesh>, baked_data_mesh> entries;
	void save(const utils::path& destination, const save_params& params) const;
	void replace(const baked_data& b);
	void max(const baked_data& b, float mult_b = 1.f);
	void average(const baked_data& b, float mult_b, float mult_base);
};

