#include "load_util.h"

#include <fstream>
#include <functional>

#include "../bake_api.h"

#include <vector_types.h>
#include <optixu/optixu_math_namespace.h>

#include <iostream>
#include <utils/binary_reader.h>
#include <utils/ini_file.h>
#include <iomanip>
#include <map>
#include <dds/dds_loader.h>
#include <utils/blob.h>
#include <utils/custom_crash_handler.h>
#include <utils/std_ext.h>
#include <utils/string_operations.h>

// #pragma optimize("", off)

using namespace optix;

struct load_data
{
	std::vector<std::shared_ptr<bake::Material>> materials;
	std::map<std::string, std::string> textures;
	std::map<std::string, std::shared_ptr<dds_loader>> loaded_textures;

	std::shared_ptr<dds_loader> get_texture(const std::string& name, bool require)
	{
		const auto lf = loaded_textures.find(name);
		if (lf != loaded_textures.end())
		{
			return lf->second;
		}

		const auto tf = textures.find(name);
		if (tf == textures.end())
		{
			return loaded_textures[name] = require ? std::make_shared<dds_loader>() : nullptr;
		}

		return loaded_textures[name] = std::make_shared<dds_loader>(tf->second.c_str(), tf->second.size());
	}
};

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef M_TAU
#define M_TAU (M_PI * 2)
#endif

#ifndef TORAD
#define TORAD(x) ((x) * M_PI / 180.0f)
#endif

#ifndef TODEG
#define TODEG(x) ((x) * 180.0f / M_PI)
#endif

inline Matrix4x4 euler(const float head, const float pitch, const float roll)
{
	return Matrix4x4::rotate(head, float3{0, 1, 0}) * Matrix4x4::rotate(pitch, float3{1, 0, 0}) * Matrix4x4::rotate(roll, float3{0, 0, 1});
}

std::vector<utils::ini_file> collect_configs(const utils::path& filename, const load_params& params)
{
	std::vector<utils::ini_file> ret;
	const auto load = [&](const utils::path& arg)
	{
		if (exists(arg))
		{
			ret.emplace_back(arg);
		}
	};

	auto relative_path = filename.relative_ac();
	const auto ac_root = utils::path(filename.string().substr(0, filename.string().size() - relative_path.size() - 1));
	for (auto& c : relative_path)
	{
		if (c == '\\') c = '/';
	}

	if (utils::starts_with(relative_path, "content/tracks/"))
	{
		auto track_config_dir = params.track_configs_dir;
		if (!track_config_dir.is_absolute())
		{
			track_config_dir = ac_root / track_config_dir;
		}

		auto track_id = relative_path.substr(utils::case_str("content/tracks/"));
		const auto track_id_end = track_id.find('/');
		if (track_id_end != std::string::npos)
		{
			track_id = track_id.substr(0, track_id_end);
		}

		const auto name = filename.filename().string();
		std::string layout_id;
		if (utils::starts_with(name, "models_") && utils::ends_with(name, ".ini"))
		{
			layout_id = name.substr(utils::case_str("models_"), name.length() - utils::case_str("models_") - utils::case_str(".ini"));
		}

		load(track_config_dir / track_id + ".ini");
		if (!layout_id.empty())
		{
			load(track_config_dir / track_id + "__" + layout_id + ".ini");
		}
	}
	return ret;
}

struct model_replacement_params
{
	struct node_filter
	{
		std::vector<std::string> i;
		node_filter() = default;
		node_filter(std::vector<std::string> i) : i(std::move(i)) {}

		bool matches(const bake::NodeBase& node) const
		{
			for (const auto& h : i)
			{
				if (std_ext::match(h, node.name)) return true;
			}
			return false;
		}
	};

	struct record
	{
		utils::path config_dir;
		std::string file;
		std::string insert_kn5;
		node_filter insert_in;
		node_filter insert_after;
		node_filter hide;
		bool multiple{};
		bool inserted{};

		float3 rotation;
		float3 scale{1.f, 1.f, 1.f};
		float3 offset;

		void tweak(const std::shared_ptr<bake::NodeBase>& node) const
		{
			/*auto m = *(Matrix4x4*)&node->matrix;
			m *= Matrix4x4::scale(scale).transpose();
			m *= euler(TORAD(rotation.x), TORAD(rotation.y), TORAD(rotation.z)).transpose();
			m *= Matrix4x4::translate(offset).transpose();
			*(Matrix4x4*)&node->matrix = m;*/
		}
	};

	std::vector<record> records;

	model_replacement_params() = default;

	model_replacement_params(const utils::path& filename)
	{
		load(utils::ini_file(filename.filename() + ".ini"));
	}

	model_replacement_params(const utils::path& filename, const load_params& params)
	{
		for (const auto& cfg : collect_configs(filename, params))
		{
			load(cfg);
		}
	}

	void load(const utils::ini_file& cfg)
	{
		const auto sections = cfg.iterate_nobreak("MODEL_REPLACEMENT");
		for (const auto& s : sections)
		{
			if (cfg.get(s, "ACTIVE", true))
			{
				record r;
				r.config_dir = cfg.filename.parent_path();
				r.file = cfg.get(s, "FILE", std::string());
				r.insert_kn5 = cfg.get(s, "INSERT", std::string());
				r.insert_in = cfg.get(s, "INSERT_IN", std::vector<std::string>());
				r.insert_after = cfg.get(s, "INSERT_AFTER", std::vector<std::string>());
				r.hide = cfg.get(s, "HIDE", std::vector<std::string>());
				r.multiple = cfg.get(s, "MULTIPLE", false);
				r.rotation = cfg.get(s, "ROTATION", float3());
				r.scale = cfg.get(s, "SCALE", float3{1.f, 1.f, 1.f});
				r.offset = cfg.get(s, "OFFSET", float3());
				records.push_back(std::move(r));
			}
		}
	}

	model_replacement_params filter(const utils::path& path) const
	{
		model_replacement_params ret;
		auto name = path.filename().string();
		std::transform(name.begin(), name.end(), name.begin(), tolower);
		for (auto& r : records)
		{
			if (std_ext::match(r.file, name))
			{
				ret.records.push_back(r);
			}
		}
		return ret;
	}

	bool to_remove(const std::shared_ptr<bake::NodeBase>& shared) const
	{
		for (const auto& r : records)
		{
			if (r.hide.matches(*shared)) return true;
		}
		return false;
	}

	void insert_in(const std::shared_ptr<bake::NodeBase>& shared, std::function<void(const utils::path& p, record& r)> callback)
	{
		for (auto& r : records)
		{
			if (!r.multiple && r.inserted || r.insert_kn5.empty()) continue;
			if (r.insert_in.matches(*shared))
			{
				r.inserted = true;
				callback(r.config_dir / r.insert_kn5, r);
			}
		}
	}

	void insert_after(const std::shared_ptr<bake::NodeBase>& shared, std::function<void(const utils::path& p, record& r)> callback)
	{
		for (auto& r : records)
		{
			if (!r.multiple && r.inserted || r.insert_kn5.empty()) continue;
			if (r.insert_after.matches(*shared))
			{
				r.inserted = true;
				callback(r.config_dir / r.insert_kn5, r);
			}
		}
	}
};

static std::shared_ptr<bake::HierarchyNode> load_knh__read_node(utils::binary_reader& reader)
{
	const auto name = reader.read_string();
	const auto transformation = reader.read_f4x4().transpose();
	auto result = std::make_shared<bake::HierarchyNode>(name, transformation);
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
	mesh->cast_shadows = reader.read_bool();
	mesh->is_visible = reader.read_bool();
	mesh->is_transparent = reader.read_bool();

	const auto vertices = reader.read_uint();
	mesh->vertices.resize(vertices);
	mesh->normals.resize(vertices);

	for (auto i = 0U; i < vertices; i++)
	{
		const auto pos = reader.read_ref<bake::Vec3>();
		auto normal = reader.read_ref<bake::Vec3>();
		const auto tex = reader.read_ref<bake::Vec2>();
		reader.skip(sizeof(float3));
		normal.y += params.normals_bias;
		mesh->vertices[i] = {pos, tex};
		*(float3*)&mesh->normals[i] = normalize(*(float3*)&normal);
	}

	const auto indices = reader.read_uint();
	mesh->triangles.resize(indices / 3);
	for (auto& t : mesh->triangles)
	{
		t = {reader.read_ushort(), reader.read_ushort(), reader.read_ushort()};
	}

	const auto material = target.materials[reader.read_uint()];
	mesh->layer = reader.read_int();
	mesh->lod_in = reader.read_float();
	mesh->lod_out = reader.read_float();
	reader.skip(sizeof(float3) + 4);

	if (params.exclude_blockers_alpha_test && (material->alpha_tested || material->blend == bake::MaterialBlendMode::coverage)
		|| params.exclude_blockers_alpha_blend && material->blend == bake::MaterialBlendMode::blend)
	{
		mesh->cast_shadows = false;
	}

	mesh->receive_shadows = true;
	mesh->material = material;
	mesh->signature_point = mesh->vertices[0].pos;

	if (mesh->lod_in > 0.f)
	{
		mesh->cast_shadows = false;
	}

	mesh->is_renderable = reader.read_bool();
	if (!mesh->is_renderable)
	{
		mesh->receive_shadows = false;
		mesh->cast_shadows = false;
	}

	return mesh;
}

static std::shared_ptr<bake::Mesh> load_kn5__read_skinned_mesh(load_data& target, const load_params& params, utils::binary_reader& reader)
{
	auto mesh = std::make_shared<bake::SkinnedMesh>();
	mesh->cast_shadows = reader.read_bool();
	mesh->is_visible = reader.read_bool();
	mesh->is_transparent = reader.read_bool();

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
		const auto tex = reader.read_ref<bake::Vec2>();
		reader.skip(sizeof(float3));
		const auto weights = reader.read_ref<bake::Vec4>();
		const auto bone_ids = reader.read_ref<bake::Vec4>();
		normal.y += params.normals_bias;
		mesh->vertices[i] = {pos, tex};
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
	mesh->layer = reader.read_int();
	mesh->lod_in = reader.read_float();
	mesh->lod_out = reader.read_float();

	if (params.exclude_blockers_alpha_test && (material->alpha_tested || material->blend == bake::MaterialBlendMode::coverage)
		|| params.exclude_blockers_alpha_blend && material->blend == bake::MaterialBlendMode::blend)
	{
		mesh->cast_shadows = false;
	}

	mesh->receive_shadows = true;
	mesh->material = material;
	mesh->signature_point = mesh->vertices[0].pos;

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

static std::shared_ptr<bake::NodeBase> load_kn5_file(const utils::path& filename, const load_params& params, const model_replacement_params& replacement_params,
	load_data* parent_data, load_data* own_data);

static std::shared_ptr<bake::NodeBase> load_model_replacement(const utils::path& filename, const load_params& params,
	model_replacement_params::record* replacement_record, load_data* parent_data)
{
	load_data own_data;
	auto r = load_kn5_file(filename, params, {}, parent_data, &own_data);
	if (replacement_record) replacement_record->tweak(r);
	return r;
}

static std::shared_ptr<bake::NodeBase> load_kn5__read_node(load_data& target, const load_params& params, model_replacement_params& replacement_params,
	utils::binary_reader& reader)
{
	const auto node_class = reader.read_uint();
	const auto name = reader.read_string();
	const auto node_children = reader.read_uint();
	const auto active = reader.read_bool();
	// skip = !reader.read_bool() || skip; // is node active

	switch (node_class)
	{
		case 0:
		case 1:
		{
			auto result = std::make_shared<bake::Node>(name, node_class == 1 ? reader.read_f4x4().transpose() : bake::NodeTransformation::identity());
			for (auto i = node_children; i > 0; i--)
			{
				auto n = load_kn5__read_node(target, params, replacement_params, reader);
				if (!replacement_params.to_remove(n))
				{
					result->add_child(n);
				}

				replacement_params.insert_after(n, [&](const utils::path& p, model_replacement_params::record& r)
				{
					result->add_child(load_model_replacement(p, params, &r, &target));
				});

				if (auto n_node = std::dynamic_pointer_cast<bake::Node>(n))
				{
					replacement_params.insert_in(n, [&](const utils::path& p, model_replacement_params::record& r)
					{
						n_node->add_child(load_model_replacement(p, params, &r, &target));
					});
				}
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

inline bool uses_alpha_from_normal(const std::string& shader_name)
{
	return shader_name == "ksPerPixelNM"
		|| shader_name == "ksPerPixelMultiMap_AT"
		|| shader_name == "ksPerPixelMultiMap_AT_NMDetail"
		|| shader_name == "nePerPixelNM_heating";
}

static std::shared_ptr<bake::NodeBase> load_kn5_file(const utils::path& filename, const load_params& params, const model_replacement_params& replacement_params,
	load_data* parent_data, load_data* own_data)
{
	if (!exists(filename)) return nullptr;

	utils::binary_reader reader(filename);
	if (!reader.match("sc6969")) return nullptr;

	const auto version = reader.read_uint();
	if (version == 6) reader.read_uint();
	if (version > 6) return nullptr;

	load_data target{};
	for (auto i = reader.read_uint(); i > 0; i--)
	{
		if (reader.read_uint() == 0) continue;
		auto name = reader.read_string(); // name
		const auto size = reader.read_uint();
		// std::cout << "texture: " << name << ", size: " << size << "\n";
		// reader.skip(size); // length+data

		target.textures[name] = reader.read_data(size);
	}

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

		if (mat->alpha_tested || mat->blend == bake::MaterialBlendMode::coverage || mat->blend == bake::MaterialBlendMode::blend)
		{
			if (const auto tx = mat->get_resource_or_null(uses_alpha_from_normal(mat->shader) ? "txNormal" : "txDiffuse"))
			{
				if (parent_data)
				{
					mat->texture = parent_data->get_texture(tx->texture, false);
				}
				if (!mat->texture)
				{
					mat->texture = target.get_texture(tx->texture, true);
				}
			}
		}

		target.materials.push_back(std::move(mat));
	}

	target.textures.clear();

	auto replacement = replacement_params.filter(filename);
	auto ret = load_kn5__read_node(target, params, replacement, reader);
	if (own_data) *own_data = std::move(target);
	return ret;
}

std::shared_ptr<bake::Node> load_model(const utils::path& filename, const load_params& params)
{
	std::string errs;

	auto p = utils::path(filename);
	auto result = std::make_shared<bake::Node>(bake::Node(""));
	const model_replacement_params model_replacement(filename, params);

	if (p.string().find(".kn5") != std::string::npos)
	{
		result->add_child(load_kn5_file(p, params, model_replacement, nullptr, nullptr));
	}
	else
	{
		const auto ini = utils::ini_file(filename);
		for (const auto& s : ini.iterate_break("MODEL"))
		{
			result->add_child(load_kn5_file(p.parent_path() / ini.get(s, "FILE", std::string()), params, model_replacement, nullptr, nullptr));
		}
	}

	return result;
}

inline void write_file(const utils::path& filename, const utils::blob_view& data)
{
	auto s = std::ofstream(filename.wstring(), std::ios::binary);
	std::copy(data.begin(), data.end(), std::ostream_iterator<char>(s));
}

void replacement_optimization(const utils::path& track_dir)
{
	std::map<std::string, uint64_t> default_textures;
	const model_replacement_params model_replacement(track_dir);
	if (model_replacement.records.empty()) return;

	std::map<std::string, int> models;
	int max_value{};
	for (auto& filename : list_files(track_dir, "models*.ini"))
	{
		utils::ini_file models_ini(filename);
		for (const auto& s : models_ini.iterate_break("MODEL"))
		{
			auto file = models_ini.get(s, "FILE", std::string());
			if (!file.empty())
			{
				max_value = max(max_value, ++models[file]);
			}
		}
	}

	for (auto& filename : list_files(track_dir, "*.kn5"))
	{
		if (!models.empty())
		{
			auto f = models.find(filename.filename().string());
			if (f != models.end() && f->second < max_value)
			{
				std::cout << "Skipping partially used: " << filename.filename() << std::endl;
				continue;
			}
		}

		utils::binary_reader reader(filename);
		if (!reader.match("sc6969")) continue;

		const auto version = reader.read_uint();
		if (version == 6) reader.read_uint();
		if (version > 6) continue;

		for (auto i = reader.read_uint(); i > 0; i--)
		{
			if (reader.read_uint() == 0) continue;
			const auto name = reader.read_string();
			const auto size = reader.read_uint();
			const auto hash_code = utils::hash_code(reader.read_data(size));
			const auto found = default_textures.find(name);
			default_textures[name] = found != default_textures.end() && found->second != hash_code ? 0ULL : hash_code;
		}
	}

	for (const auto& record : model_replacement.records)
	{
		if (record.insert_kn5.empty()) continue;

		const auto replacement_filename = record.config_dir / record.insert_kn5;
		std::cout << "Processing: " << replacement_filename.filename() << std::endl;
		utils::binary_reader reader(replacement_filename);
		if (!reader.match("sc6969")) continue;

		utils::blob fixed_kn5;
		fixed_kn5.append("sc6969", 6);

		const auto version = reader.read_uint();
		fixed_kn5 << version;
		if (version == 6) fixed_kn5 << reader.read_uint();
		if (version > 6) continue;

		struct texture_info
		{
			std::string name;
			std::string data;
		};
		std::vector<texture_info> textures;

		const auto textures_count = reader.read_uint();
		for (auto i = textures_count; i > 0; i--)
		{
			auto state = reader.read_uint();
			if (state == 0) continue;

			const auto name = reader.read_string();
			const auto size = reader.read_uint();
			auto data = reader.read_data(size);
			const auto hash_code = utils::hash_code(data);
			const auto found = default_textures.find(name);
			if (found == default_textures.end() || hash_code != found->second)
			{
				textures.push_back({name, std::move(data)});
			}
			else
			{
				std::cout << "\tRemoving: " << name << std::endl;
			}
		}

		const auto destination = replacement_filename.parent_path() / ".optimized" / replacement_filename.filename();
		if (textures.size() == textures_count)
		{
			std::cout << "\tNothing to remove, leaving KN5 as it is" << std::endl;
			if (exists(destination))
			{
				DeleteFileW(destination.wstring().c_str());
			}
			continue;
		}

		fixed_kn5 << uint32_t(textures.size());
		for (auto& t : textures)
		{
			fixed_kn5 << 1U;
			fixed_kn5 << t.name;
			fixed_kn5 << t.data;
		}

		auto remaining = reader.read_rest();
		fixed_kn5.append(remaining.data(), remaining.size());
		write_file(destination, fixed_kn5);
	}
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
				entry.frames[j] = reader.read_f4x4().transpose();
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
