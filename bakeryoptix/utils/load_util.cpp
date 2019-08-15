#include "load_util.h"
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

struct load_data
{
	std::vector<std::shared_ptr<bake::Material>> materials;
};

static std::shared_ptr<bake::HierarchyNode> load_knh__read_node(utils::binary_reader& reader)
{
	const auto name = reader.read_string();
	const auto transformation = reader.read_f4x4().transpose();
	const auto result = std::make_shared<bake::HierarchyNode>(name, transformation);
	result->children.resize(reader.read_uint());
	for (auto& i : result->children)
	{
		i = load_knh__read_node(reader);
	}
	return result;
}

std::shared_ptr<bake::HierarchyNode> load_hierarchy(const utils::path& filename)
{
	if (!exists(filename)) return nullptr;
	utils::binary_reader reader(filename);
	return load_knh__read_node(reader);
}

static std::shared_ptr<bake::Mesh> load_kn5__read_mesh(load_data& target, const load_params& params, utils::binary_reader& reader)
{
	auto mesh = std::make_shared<bake::Mesh>();

	const auto cast_shadows = reader.read_bool();
	const auto is_visible = reader.read_bool();
	const auto is_transparent = reader.read_bool();
	mesh->cast_shadows = cast_shadows && !is_transparent;

	const auto vertices = reader.read_uint();
	mesh->vertices.resize(vertices);
	mesh->normals.resize(vertices);

	for (auto i = 0U; i < vertices; i++)
	{
		const auto pos = reader.read_ref<bake::Vec3>();
		auto normal = reader.read_ref<bake::Vec3>();
		reader.skip(sizeof(float2) + sizeof(float3));
		normal.y += params.normals_bias;
		mesh->vertices[i] = pos;
		*(float3*)&mesh->normals[i] = normalize(*(float3*)&normal);
	}

	const auto indices = reader.read_uint();
	mesh->triangles.resize(indices / 3);
	for (auto& t : mesh->triangles)
	{
		t = {reader.read_ushort(), reader.read_ushort(), reader.read_ushort()};
	}

	const auto material = target.materials[reader.read_uint()];
	const auto layer = reader.read_uint();
	const auto lod_in = reader.read_float();
	const auto lod_out = reader.read_float();
	reader.skip(sizeof(float3) + 4);

	if (params.exclude_blockers_alpha_test && (material->alpha_tested || material->blend == bake::MaterialBlendMode::coverage)
		|| material->blend == bake::MaterialBlendMode::blend)
	{
		mesh->cast_shadows = false;
	}

	mesh->receive_shadows = true;
	mesh->visible = is_visible;
	mesh->material = material;
	mesh->signature_point = mesh->vertices[0];

	const auto is_renderable = reader.read_bool();
	if (!is_renderable || lod_in > 0.f)
	{
		mesh->receive_shadows = false;
		mesh->cast_shadows = false;
	}

	return mesh;
}

static std::shared_ptr<bake::Mesh> load_kn5__read_skinned_mesh(load_data& target, const load_params& params, utils::binary_reader& reader)
{
	auto mesh = std::make_shared<bake::SkinnedMesh>();

	const auto cast_shadows = reader.read_bool();
	const auto is_visible = reader.read_bool();
	const auto is_transparent = reader.read_bool();
	mesh->cast_shadows = cast_shadows && !is_transparent;

	mesh->bones.resize(reader.read_uint());
	for (auto& b : mesh->bones)
	{
		b = bake::Bone(reader.read_string(), reader.read_f4x4().transpose());
	}

	const auto vertices = reader.read_uint();
	mesh->vertices.resize(vertices);
	mesh->normals.resize(vertices);
	mesh->weights.resize(vertices);
	mesh->bone_ids.resize(vertices);

	for (auto i = 0U; i < vertices; i++)
	{
		const auto pos = reader.read_ref<bake::Vec3>();
		auto normal = reader.read_ref<bake::Vec3>();
		reader.skip(sizeof(float2) + sizeof(float3));
		const auto weights = reader.read_ref<bake::Vec4>();
		const auto bone_ids = reader.read_ref<bake::Vec4>();
		normal.y += params.normals_bias;
		mesh->vertices[i] = pos;
		*(float3*)&mesh->normals[i] = normalize(*(float3*)&normal);
		mesh->weights[i] = weights;
		mesh->bone_ids[i] = bone_ids;
	}

	const auto indices = reader.read_uint();
	mesh->triangles.resize(indices / 3);
	for (auto& t : mesh->triangles)
	{
		t = {reader.read_ushort(), reader.read_ushort(), reader.read_ushort()};
	}

	const auto material = target.materials[reader.read_uint()];
	const auto layer = reader.read_uint();
	mesh->lod_in = reader.read_float();
	mesh->lod_out = reader.read_float();

	if (params.exclude_blockers_alpha_test && (material->alpha_tested || material->blend == bake::MaterialBlendMode::coverage)
		|| material->blend == bake::MaterialBlendMode::blend)
	{
		mesh->cast_shadows = false;
	}

	mesh->receive_shadows = true;
	mesh->visible = is_visible;
	mesh->material = material;
	mesh->signature_point = mesh->vertices[0];

	if (mesh->lod_in > 0.f)
	{
		mesh->cast_shadows = false;
	}

	return mesh;
}

static std::shared_ptr<bake::Mesh> proc_mesh(const std::shared_ptr<bake::Mesh>& result, const load_params& params, const std::string& name, bool active)
{
	if (result)
	{
		result->name = name;
		result->active_local = active;
		for (const auto& e : params.exclude_blockers)
		{
			if (result->matches(e))
			{
				// std::cout << "DOES NOT CAST SHADOWS: " << result->name << "\n";
				result->cast_shadows = false;
				break;
			}
		}
		for (const auto& e : params.exclude_patch)
		{
			if (result->matches(e))
			{
				// std::cout << "DOES NOT RECEIVE SHADOWS: " << result->name << "\n";
				result->receive_shadows = false;
				break;
			}
		}
		// DBG(result->name, result->material->name, result->material->shader, result->cast_shadows, result->receive_shadows)
	}
	return result;
}

static std::shared_ptr<bake::NodeBase> load_kn5__read_node(load_data& target, const load_params& params, utils::binary_reader& reader)
{
	const auto node_class = reader.read_uint();
	const auto name = reader.read_string();
	const auto node_children = reader.read_uint();
	const auto active = reader.read_bool();
	// skip = !reader.read_bool() || skip; // is node active

	switch (node_class)
	{
		case 0:
		{
			auto result = std::make_shared<bake::Node>(name, bake::NodeTransformation::identity());
			for (auto i = node_children; i > 0; i--)
			{
				result->add_child(load_kn5__read_node(target, params, reader));
			}
			result->active_local = active;
			return result;
		}

		case 1:
		{
			auto result = std::make_shared<bake::Node>(name, reader.read_f4x4().transpose());
			for (auto i = node_children; i > 0; i--)
			{
				result->add_child(load_kn5__read_node(target, params, reader));
			}
			result->active_local = active;
			return result;
		}

		case 2:
		{
			return proc_mesh(load_kn5__read_mesh(target, params, reader), params, name, active);
		}

		case 3:
		{
			return proc_mesh(load_kn5__read_skinned_mesh(target, params, reader), params, name, active);
		}

		default: std::cerr << "Unexpected node class: " << node_class;
			return nullptr;
	}
}

std::shared_ptr<bake::NodeBase> load_kn5_file(const utils::path& filename, const load_params& params)
{
	if (!exists(filename)) return nullptr;

	utils::binary_reader reader(filename);
	if (!reader.match("sc6969")) return nullptr;

	const auto version = reader.read_uint();
	if (version == 6) reader.read_uint();
	if (version > 6) return nullptr;

	for (auto i = reader.read_uint(); i > 0; i--)
	{
		if (reader.read_uint() == 0) continue;
		auto name = reader.read_string(); // name
		const auto size = reader.read_uint();
		// std::cout << "texture: " << name << ", size: " << size << "\n";
		reader.skip(size); // length+data
	}

	load_data target{};
	for (auto i = reader.read_uint(); i > 0; i--)
	{
		auto mat = std::make_shared<bake::Material>();
		mat->name = reader.read_string();
		mat->shader = reader.read_string();
		mat->blend = reader.read_ref<bake::MaterialBlendMode>();
		mat->alpha_tested = reader.read_bool();
		mat->depth_mode = reader.read_uint();

		for (auto j = reader.read_int(); j > 0; j--)
		{
			bake::MaterialProperty prop;
			prop.name = reader.read_string();
			prop.v1 = reader.read_float();
			reader.skip(4 * (2 + 3 + 4));
			mat->vars.push_back(prop);
		}

		for (auto j = reader.read_int(); j > 0; j--)
		{
			bake::MaterialResource res;
			res.name = reader.read_string();
			res.slot = reader.read_uint();
			res.texture = reader.read_string();
			mat->resources.push_back(res);
		}

		target.materials.push_back(mat);
	}

	return load_kn5__read_node(target, params, reader);
}

std::shared_ptr<bake::Node> load_model(const utils::path& filename, const load_params& params)
{
	std::string errs;

	auto p = utils::path(filename);
	auto result = std::make_shared<bake::Node>(bake::Node(""));

	if (p.string().find(".kn5") != std::string::npos)
	{
		result->add_child(load_kn5_file(p, params));
	}
	else
	{
		const auto ini = utils::ini_file(filename);
		for (const auto& s : ini.iterate_break("MODEL"))
		{
			result->add_child(load_kn5_file(p.parent_path() / ini.get(s, "FILE", std::string()), params));
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

std::shared_ptr<bake::Animation> load_ksanim(const utils::path& filename, bool include_static)
{
	std::shared_ptr<bake::Animation> result = std::make_shared<bake::Animation>();
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
		entry.name = name;
		entry.frames.resize(frames_count);
		auto dif = include_static;

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

		if (dif)
		{
			result->entries.push_back(entry);
		}
	}

	return result;
}

std::vector<bake::AILanePoint> load_ailane(const utils::path& filename)
{
	std::vector<bake::AILanePoint> result;
	if (!exists(filename)) return result;

	utils::binary_reader reader(filename);
	const auto version = reader.read_uint();
	if (version != 7)
	{
		std::cerr << "Unsupported AI lane version: " << version << " (`" << filename.relative_ac() << "`)";
		return result;
	}

	const auto count = reader.read_uint();
	result.resize(count);

	const auto lap_time = reader.read_uint();
	const auto sample_count = reader.read_uint();

	for (auto i = 0U; i < count; i++)
	{
		result[i].point = reader.read_ref<bake::Vec3>();
		result[i].length = reader.read_float();
		result[i].index = reader.read_uint();
		result[i].side_left = 4.f;
		result[i].side_right = 4.f;
	}

	const auto count_2 = reader.read_uint();
	if (count_2 == count)
	{
		for (auto i = 0U; i < count; i++)
		{
			const auto speed = reader.read_float();
			const auto gas = reader.read_float();
			const auto brake = reader.read_float();
			const auto obsoletelatg = reader.read_float();
			const auto radius = reader.read_float();
			result[i].side_left = reader.read_float();
			result[i].side_right = reader.read_float();
			const auto camber = reader.read_float();
			const auto direction = reader.read_float();
			const auto normal = reader.read_f3();
			const auto length = reader.read_float();
			const auto forwardvector = reader.read_f3();
			const auto tag = reader.read_float();
			const auto grade = reader.read_float();
		}
	}
	else
	{
		std::cerr << "Trying to load AI lane: count is not equal to count_2\n";
	}

	return result;
}
