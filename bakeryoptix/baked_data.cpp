#include "baked_data.h"

#include <iostream>
#include <algorithm>
#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

#include <utils/half.h>
#include <utils/miniz/miniz.h>
#include <utils/filesystem.h>
#include <utils/ini_file.h>
#include <bake_api.h>
#include <utils/dbg_output.h>
#include <utils/cout_progress.h>

template <typename T>
void add_bytes(std::basic_string<uint8_t>& d, const T& value)
{
	for (auto i = 0U; i < sizeof(T); i++)
	{
		d += ((uint8_t*)&value)[i];
	}
}

void add_string(std::basic_string<uint8_t>& d, const std::string& value)
{
	add_bytes(d, int(value.size()));
	for (const auto i : value)
	{
		d += *(uint8_t*)&i;
	}
}

void baked_data::save(const utils::path& destination, const save_params& params, bool store_secondary_set) const
{
	using namespace optix;
	std::basic_string<uint8_t> data;

	std::vector<unsigned> nearbyes;
	std::vector<bool> averaged;

	cout_progress progress{entries.size()};
	for (const auto& entry : entries)
	{
		progress.report();

		const auto m = entry.first;
		auto ao = entry.second;

		if (m->name == "@@__EXTRA_AO@")
		{
			add_string(data, m->name);
			add_bytes(data, int(4));
			add_bytes(data, m->signature_point);

			std::vector<float4> cleaned_up;
			cleaned_up.reserve(m->vertices.size());
			for (auto j = 0U; j < m->vertices.size(); j++)
			{
				cleaned_up.push_back(make_float4(m->vertices[j].x, m->vertices[j].y, m->vertices[j].z, ao.main_set[j].x));
			}

			/*add_bytes(data, int(m->vertices.size()));
			for (auto j = 0U; j < m->vertices.size(); j++)
			{
				add_bytes(data, m->vertices[j]);
				add_bytes(data, half_float::half(ao.main_set[j].x));
			}*/

			add_bytes(data, int(cleaned_up.size()));
			for (const auto& j : cleaned_up)
			{
				add_bytes(data, *(bake::Vec3*)&j);
				add_bytes(data, byte(clamp(powf(j.w * 1.1f, 1.1f) * 255, 0.f, 255.f)));
			}

			continue;
		}

		if (!m->normals.empty() && params.averaging_threshold > 0.f)
		{
			if (m->vertices.size() > nearbyes.size())
			{
				nearbyes.resize(m->vertices.size());
				averaged.resize(m->vertices.size());
			}
			std::fill(averaged.begin(), averaged.end(), false);

			struct vertex
			{
				float pos;
				unsigned index;
			};

			std::vector<vertex> vertices_conv;
			vertices_conv.reserve(m->vertices.size());
			for (auto j = 0U; j < m->vertices.size(); j++)
			{
				vertices_conv.push_back({m->vertices[j].x, j});
			}
			std::sort(vertices_conv.begin(), vertices_conv.end(), [](const vertex& a, const vertex& b) { return a.pos > b.pos; });

			for (auto j2 = 0U; j2 < m->vertices.size(); j2++)
			{
				const auto j = vertices_conv[j2].index;
				if (averaged[j]) continue;
				const auto& jp = *(float3*)&m->vertices[j];
				const auto& jn = *(float3*)&m->normals[j];
				auto t_main = ao.main_set[j];
				auto t_alternative = ao.alternative_set.empty() ? float2{} : ao.alternative_set[j];
				auto nearbyes_size = 0U;
				for (auto k2 = j2 + 1; k2 < m->vertices.size() && k2 - j2 < 100; k2++)
				{
					const auto k = vertices_conv[k2].index;
					const auto& kn = *(float3*)&m->normals[k];
					const auto nd = dot(jn, kn);
					if (nd < params.averaging_cos_threshold) continue;

					const auto& kp = *(float3*)&m->vertices[k];
					const auto dp = jp - kp;
					const auto dq = std::max(abs(dp.x), std::max(abs(dp.y), abs(dp.z)));

					if (dq < params.averaging_threshold)
					{
						nearbyes[nearbyes_size++] = k;
						t_main += ao.main_set[k];
						if (!ao.alternative_set.empty())
						{
							t_alternative += ao.alternative_set[k];
						}
					}
				}
				if (nearbyes_size > 0)
				{
					t_main /= float(nearbyes_size + 1);
					t_alternative /= float(nearbyes_size + 1);
					ao.main_set[j] = t_main;
					if (!ao.alternative_set.empty())
					{
						ao.alternative_set[j] = t_alternative;
					}
					for (auto k = 0U; k < nearbyes_size; k++)
					{
						ao.main_set[nearbyes[k]] = t_main;
						if (!ao.alternative_set.empty())
						{
							ao.alternative_set[nearbyes[k]] = t_alternative;
						}
						averaged[nearbyes[k]] = true;
					}
				}
			}
		}

		add_string(data, m->name);
		add_bytes(data, int(1));
		add_bytes(data, m->signature_point);
		add_bytes(data, int(m->vertices.size()));
		for (auto j = 0U; j < m->vertices.size(); j++)
		{
			add_bytes(data, half_float::half(ao.main_set[j].x));
		}

		if (store_secondary_set)
		{
			// std::cout << "Store secondary set: " << m->name << "\n";
			add_string(data, m->name);
			add_bytes(data, int(3));
			add_bytes(data, m->signature_point);
			add_bytes(data, int(m->vertices.size()));
			for (auto j = 0U; j < m->vertices.size(); j++)
			{
				add_bytes(data, half_float::half(ao.main_set[j].y));
			}
		}

		if (!ao.alternative_set.empty())
		{
			add_string(data, "@@__ALT@:" + m->name);
			add_bytes(data, int(1));
			add_bytes(data, m->signature_point);
			add_bytes(data, int(m->vertices.size()));
			for (auto j = 0U; j < m->vertices.size(); j++)
			{
				add_bytes(data, half_float::half(ao.alternative_set[j].x));
			}

			if (store_secondary_set)
			{
				// std::cout << "Store secondary set: " << m->name << "\n";
				add_string(data, "@@__ALT@:" + m->name);
				add_bytes(data, int(3));
				add_bytes(data, m->signature_point);
				add_bytes(data, int(m->vertices.size()));
				for (auto j = 0U; j < m->vertices.size(); j++)
				{
					add_bytes(data, half_float::half(ao.alternative_set[j].y));
				}
			}
		}
	}

	remove(destination.string().c_str());
	mz_zip_add_mem_to_archive_file_in_place(destination.string().c_str(), "Patch_v3.data",
		data.c_str(), data.size(), nullptr, 0, 0);

	auto config = params.extra_config;
	config.set("LIGHTING", "BRIGHTNESS", params.brightness);
	config.set("LIGHTING", "GAMMA", params.gamma);
	config.set("LIGHTING", "OPACITY", params.opacity);
	const auto config_data = config.to_string();
	mz_zip_add_mem_to_archive_file_in_place(destination.string().c_str(), "Config.ini",
		config_data.c_str(), config_data.size(), nullptr, 0, 0);
}

baked_data_mesh_set replace_primary_mesh_data(const baked_data_mesh_set& b, const baked_data_mesh_set& base)
{
	baked_data_mesh_set result;
	if (b.size() != base.size())
	{
		std::cerr << "\n[ ERROR: Sizes do not match. ]\n";
		exit(1);
	}
	result.resize(b.size());
	for (auto i = 0U; i < b.size(); i++)
	{
		result[i].x = b[i].x;
		result[i].y = base[i].y;
	}
	return result;
}

baked_data_mesh_set merge_mesh_data(const baked_data_mesh_set& b, const baked_data_mesh_set& base, float mult_b, bool use_min, bool merge_x, bool merge_y)
{
	baked_data_mesh_set result;
	if (b.size() != base.size())
	{
		std::cerr << "\n[ ERROR: Sizes do not match. ]\n";
		exit(1);
	}
	result.resize(b.size());
	for (auto i = 0U; i < b.size(); i++)
	{
		result[i].x = merge_x ? (use_min ? std::min(b[i].x * mult_b, base[i].x) : std::max(b[i].x * mult_b, base[i].x)) : base[i].x;
		result[i].y = merge_y ? (use_min ? std::min(b[i].y * mult_b, base[i].y) : std::max(b[i].y * mult_b, base[i].y)) : base[i].y;
	}
	return result;
}

baked_data_mesh_set average_mesh_data(const baked_data_mesh_set& b, const baked_data_mesh_set& base, float mult_b, float mult_base, bool merge_x, bool merge_y)
{
	baked_data_mesh_set result;
	if (b.size() != base.size())
	{
		std::cerr << "\n[ ERROR: Sizes do not match. ]\n";
		exit(1);
	}
	result.resize(b.size());
	for (auto i = 0U; i < b.size(); i++)
	{
		result[i].x = merge_x ? (b[i].x * mult_b + base[i].x * mult_base) : base[i].x;
		result[i].y = merge_y ? (b[i].y * mult_b + base[i].y * mult_base) : base[i].y;
	}
	return result;
}

void baked_data::replace_primary(const baked_data& b)
{
	for (const auto& p : b.entries)
	{
		auto f = entries.find(p.first);
		entries[p.first].main_set = f == entries.end()
			? p.second.main_set
			: replace_primary_mesh_data(p.second.main_set, f->second.main_set);
	}
}

void baked_data::max(const baked_data& b, float mult_b, const std::vector<std::shared_ptr<bake::Mesh>>& inverse, bool apply_to_both_sets)
{
	for (const auto& p : b.entries)
	{
		const auto use_min = std::find(inverse.begin(), inverse.end(), p.first) != inverse.end();
		auto f = entries.find(p.first);
		entries[p.first].main_set = f == entries.end()
			? p.second.main_set
			: merge_mesh_data(p.second.main_set, f->second.main_set, use_min ? 1.f : mult_b, use_min, true, apply_to_both_sets);
	}
}

void baked_data::replace(const baked_data& b)
{
	for (const auto& p : b.entries)
	{
		entries[p.first] = p.second;
	}
}

void baked_data::average(const baked_data& b, float mult_b, float mult_base, const std::vector<std::shared_ptr<bake::Mesh>>& inverse, bool apply_to_both_sets)
{
	for (const auto& p : b.entries)
	{
		const auto use_min = std::find(inverse.begin(), inverse.end(), p.first) != inverse.end();
		auto f = entries.find(p.first);
		entries[p.first].main_set = f == entries.end()
			? p.second.main_set
			: use_min
			? merge_mesh_data(p.second.main_set, f->second.main_set, 1.f, true, true, apply_to_both_sets)
			: average_mesh_data(p.second.main_set, f->second.main_set, mult_b, mult_base, true, apply_to_both_sets);
	}
}

void baked_data::extend(const baked_data& b)
{
	for (const auto& p : b.entries)
	{
		auto f = entries.find(p.first);
		if (f == entries.end())
		{
			entries[p.first] = p.second;
		}
	}
}

void baked_data::set_alternative_set(const baked_data& b)
{
	for (const auto& p : b.entries)
	{
		auto f = entries.find(p.first);
		if (f != entries.end())
		{
			entries[p.first].alternative_set = p.second.main_set;
		}
	}
}

void baked_data::fill(const std::shared_ptr<bake::Mesh>& mesh, float x)
{
	if (!mesh) return;
	auto& entry = entries[mesh];
	entry.main_set.resize(mesh->vertices.size());
	for (auto j = 0U; j < mesh->vertices.size(); j++)
	{
		entry.main_set[j] = float2{x, x};
	}
}
