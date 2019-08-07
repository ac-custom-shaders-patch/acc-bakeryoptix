#include "load_scene.h"
#include "../bake_api.h"

#include <vector_types.h>
#include <optixu/optixu_math_namespace.h>

#include <iostream>
#include <utils/binary_reader.h>
#include <utils/ini_file.h>
#include <iomanip>
#include <utils/std_ext.h>

// #pragma optimize("", off)

using namespace optix;

struct kn5material_property
{
	std::string name;
	float v1{};
	float2 v2{};
	float3 v3{};
	float4 v4{};
};

struct kn5material_resource
{
	std::string name;
	uint32_t slot{};
	std::string texture;
};

enum class kn5material_blend : byte
{
	opaque = 0,
	blend = 1,
	coverage = 2
};

struct kn5material
{
	std::string name;
	std::string shader;
	kn5material_blend blend{};
	bool alpha_tested{};
	uint32_t depth_mode{};
	std::vector<kn5material_property> properties;
	std::vector<kn5material_resource> resources;
};

struct load_data
{
	std::vector<kn5material> materials;
};

static std::shared_ptr<bake::Mesh> load_kn5__read_mesh(load_data& target, const load_params& params, utils::binary_reader& reader, bool skip)
{
	const auto cast_shadows = reader.read_bool();
	const auto is_visible = reader.read_bool();
	const auto is_transparent = reader.read_bool();
	if (!is_visible) skip = true;

	if (skip)
	{
		const auto vs = sizeof(float3) + sizeof(float3) + sizeof(float2) + sizeof(float3);
		reader.skip(reader.read_uint() * vs);
		reader.skip(reader.read_uint() * sizeof(uint16_t) + 33);
		return nullptr;
	}

	auto mesh = std::make_shared<bake::Mesh>();
	mesh->cast_shadows = cast_shadows && !is_transparent;
	mesh->vertex_stride_bytes = 0;
	mesh->normal_stride_bytes = 0;

	const auto vertices = reader.read_uint();
	mesh->num_vertices = vertices;
	mesh->vertices = new float[vertices * 3];
	mesh->normals = new float[vertices * 3];
	for (auto i = 0U; i < vertices; i++)
	{
		const auto pos = reader.read_f3();
		auto normal = reader.read_f3();
		reader.skip(sizeof(float2) + sizeof(float3));
		normal.y += params.normals_bias;
		*(float3*)&mesh->vertices[i * 3] = pos;
		*(float3*)&mesh->normals[i * 3] = optix::normalize(normal);
	}

	const auto indices = reader.read_uint();
	mesh->num_triangles = indices / 3;
	mesh->tri_vertex_indices = new unsigned[indices];
	for (auto i = 0U; i < indices; i++)
	{
		mesh->tri_vertex_indices[i] = reader.read_ushort();
	}

	const auto material = target.materials[reader.read_uint()];
	const auto layer = reader.read_uint();
	const auto lod_in = reader.read_float();
	const auto lod_out = reader.read_float();
	reader.skip(sizeof(float3) + 4);

	if (material.alpha_tested || material.blend != kn5material_blend::opaque)
	{
		mesh->cast_shadows = false;
	}

	mesh->receive_shadows = true;

	const auto is_renderable = reader.read_bool();
	if (!is_renderable || lod_in > 0.f)
	{
		return nullptr;
	}

	return mesh;
}

static std::shared_ptr<bake::NodeBase> load_kn5__read_node(load_data& target, const load_params& params, utils::binary_reader& reader, bool skip)
{
	const auto node_class = reader.read_uint();
	const auto name = reader.read_string();
	const auto node_children = reader.read_uint();
	skip = !reader.read_bool() || skip; // is node active

	switch (node_class)
	{
	case 0:
		{
			auto result = std::make_shared<bake::Node>(name, bake::NodeTransformation::identity());
			for (auto i = node_children; i > 0; i--)
			{
				result->add_child(load_kn5__read_node(target, params, reader, skip));
			}
			return result;
		}

	case 1:
		{
			auto result = std::make_shared<bake::Node>(name, reader.read_f4x4().transpose());
			for (auto i = node_children; i > 0; i--)
			{
				result->add_child(load_kn5__read_node(target, params, reader, skip));
			}
			return result;
		}

	case 2:
		{
			auto result = load_kn5__read_mesh(target, params, reader, skip);
			if (result)
			{
				result->name = name;
				for (const auto& e : params.exclude_blockers)
				{
					if (std_ext::match(e, name))
					{
						result->cast_shadows = false;
						break;
					}
	
				}
				for (const auto& e : params.exclude_patch)
				{
					if (std_ext::match(e, name))
					{
						result->receive_shadows = false;
						break;
					}
	
				}
			}
			return result;
		}

	case 3:
		{
			reader.skip(3); // cast shadows + is visible + is transparent
			for (auto i = reader.read_uint(); i > 0; i--)
			{
				reader.skip_string();
				reader.skip(sizeof(bake::NodeTransformation));
			}
			// pos + normal + uv + tangent + bone weights + bone indices
			const auto vs = sizeof(float3) + sizeof(float3) + sizeof(float2) + sizeof(float3) + sizeof(float4) + sizeof(float4);
			reader.skip(reader.read_uint() * vs);
			reader.skip(reader.read_uint() * sizeof(uint16_t));
			reader.skip(4 + 4); // material ID + layer
			reader.skip(8); // mystery bytes
			return nullptr;
		}

	default: std::cerr << "Unexpected node class: " << node_class;
		return nullptr;
	}
}

std::shared_ptr<bake::NodeBase> load_kn5_file(load_data& target, const utils::path& filename, const load_params& params)
{
	if (!exists(filename)) return nullptr;

	utils::binary_reader reader(filename);
	if (!reader.match("sc6969")) return nullptr;

	// skipping header
	auto version = reader.read_uint();
	// std::cout << "version: " << version << "\n";

	if (version == 6) reader.read_uint();
	if (version > 6) return nullptr;

	// skipping textures
	for (auto i = reader.read_uint(); i > 0; i--)
	{
		if (reader.read_uint() == 0) continue;
		auto name = reader.read_string(); // name
		auto size = reader.read_uint();
		// std::cout << "texture: " << name << ", size: " << size << "\n";
		reader.skip(size); // length+data
	}

	// skipping materials
	for (auto i = reader.read_uint(); i > 0; i--)
	{
		kn5material mat;
		mat.name = reader.read_string();
		mat.shader = reader.read_string();
		mat.blend = reader.read_ref<kn5material_blend>();
		mat.alpha_tested = reader.read_bool();
		mat.depth_mode = reader.read_uint();

		for (auto j = reader.read_int(); j > 0; j--)
		{
			kn5material_property prop;
			prop.name = reader.read_string();
			prop.v1 = reader.read_float();
			reader.skip(4 * (2 + 3 + 4));
			mat.properties.push_back(prop);
		}

		for (auto j = reader.read_int(); j > 0; j--)
		{
			kn5material_resource res;
			res.name = reader.read_string();
			res.slot = reader.read_uint();
			res.texture = reader.read_string();
			mat.resources.push_back(res);
		}

		target.materials.push_back(mat);
	}

	return load_kn5__read_node(target, params, reader, false);
}

std::shared_ptr<bake::Node> load_scene(const utils::path& filename, const load_params& params)
{
	std::string errs;
	load_data target{};

	auto p = utils::path(filename);
	auto result = std::make_shared<bake::Node>(bake::Node(""));

	if (p.string().find(".kn5") != std::string::npos)
	{
		result->add_child(load_kn5_file(target, p, params));
	}
	else
	{
		auto ini = utils::ini_file(filename);
		for (const auto& s : ini.iterate("MODEL"))
		{
			result->add_child(load_kn5_file(target, p.parent_path() / ini.get(s, "FILE", std::string()), params));
		}
	}

	return result;
}

Matrix4x4 rotation_quaternion(const float4& rotation)
{
	Matrix4x4 matrix;
	const auto xx = rotation.x * rotation.x;
	const auto yy = rotation.y * rotation.y;
	const auto zz = rotation.z * rotation.z;
	const auto yx = rotation.y * rotation.x;
	const auto wz = rotation.w * rotation.z;
	const auto zx = rotation.z * rotation.x;
	const auto wy = rotation.w * rotation.y;
	const auto zy = rotation.z * rotation.y;
	const auto wx = rotation.w * rotation.x;
	matrix[0] = 1.f - (zz + yy) * 2.f;
	matrix[1] = (wz + yx) * 2.f;
	matrix[2] = (zx - wy) * 2.f;
	matrix[3] = 0.f;
	matrix[4] = (yx - wz) * 2.f;
	matrix[5] = 1.f - (zz + xx) * 2.f;
	matrix[6] = (wx + zy) * 2.f;
	matrix[7] = 0.f;
	matrix[8] = (wy + zx) * 2.f;
	matrix[9] = (zy - wx) * 2.f;
	matrix[10] = 1.f - (yy + xx) * 2.f;
	matrix[11] = 0.f;
	matrix[12] = 0.f;
	matrix[13] = 0.f;
	matrix[14] = 0.f;
	matrix[15] = 1.f;
	return matrix.transpose();
}

bake::Animation load_ksanim(const utils::path& filename, const std::shared_ptr<bake::Node>& root)
{
	bake::Animation result;
	if (!exists(filename)) return result;

	utils::binary_reader reader(filename);
	const auto version = reader.read_uint();
	if (version > 2)
	{
		std::cerr << "Unsupported animation version: " << version << " (`" << filename.relative_ac() << "`)";
		return result;
	}

	const auto count = reader.read_uint();
	for (auto i = 0U; i < count; i++)
	{
		const auto name = reader.read_string();
		const auto frames_count = reader.read_uint();
		bake::NodeTransition entry;
		entry.node = root->find_node(name);
		entry.frames.resize(frames_count);
		auto dif = false;

		for (auto j = 0U; j < frames_count; j++)
		{
			if (version == 1)
			{
				entry.frames[j] = reader.read_f4x4();
			}
			else
			{
				const auto rotation = rotation_quaternion(reader.read_f4());
				const auto translation = Matrix4x4::translate(reader.read_f3());
				const auto scale = Matrix4x4::scale(reader.read_f3());
				const auto final_matrix = translation * rotation * scale;
				entry.frames[j] = *(bake::NodeTransformation*)&final_matrix;
			}

			if (!dif && j > 0 && entry.frames[j - 1] != entry.frames[j])
			{
				dif = true;
			}
		}

		if (entry.node && dif)
		{
			result.entries.push_back(entry);
		}
	}

	return result;
}
