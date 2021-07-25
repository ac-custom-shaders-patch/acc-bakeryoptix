#pragma once
#include <map>
#include <vector>
#include <memory>
#include <utils/blob.h>
#include <utils/ini_file.h>

#define VAO_DECODE_POW 2.f
#define VAO_ENCODE_POW (1.f / VAO_DECODE_POW)

namespace utils
{
	class path;
}

namespace bake
{
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
	bool average_ao_in_same_pos;
};

struct baked_data
{
	std::map<std::shared_ptr<bake::Mesh>, baked_data_mesh> entries;
	std::vector<float> extra_points_ao;
	std::map<std::string, utils::blob> extra_entries;

	void smooth_ao();
	void save(const utils::path& destination, const save_params& params, bool store_secondary_set);
	void replace(const baked_data& b);
	void replace_primary(const baked_data& b);
	void brighten(const baked_data& b, float brighten_k);
	void max(const baked_data& b, float mult_b = 1.f, const std::vector<std::shared_ptr<bake::Mesh>>& inverse = {}, bool apply_to_both_sets = false);
	void average(const baked_data& b, float mult_b, float mult_base, const std::vector<std::shared_ptr<bake::Mesh>>& inverse = {}, bool apply_to_both_sets = false);
	void extend(const baked_data& b);
	void set_alternative_set(const baked_data& b);
	void fill(const std::shared_ptr<bake::Mesh>& mesh, float x);
};
