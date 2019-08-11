﻿#include "baked_data.h"

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

	for (const auto& entry : entries)
	{
		const auto m = entry.first;
		auto ao = entry.second;

		if (params.averaging_threshold > 0.f)
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

			/*for (auto j2 = 0U; j2 < m->vertices.size(); j2++)
			{
				const auto j = vertices_conv[j2].index;
				if (averaged[j]) continue;
				const auto& jp = *(float3*)&m->vertices[j];
				const auto& jn = *(float3*)&m->normals[j];
				auto t = ao[j];
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
						t += ao[k];
					}
				}
				if (nearbyes_size > 0)
				{
					t /= float(nearbyes_size + 1);
					ao[j] = t;
					for (auto k = 0U; k < nearbyes_size; k++)
					{
						ao[nearbyes[k]] = t;
						averaged[nearbyes[k]] = true;
					}
				}
			}*/
		}

		add_string(data, m->name);
		add_bytes(data, int(1));
		add_bytes(data, m->vertices[0]);
		add_bytes(data, int(m->vertices.size()));
		for (auto j = 0U; j < m->vertices.size(); j++)
		{
			add_bytes(data, half_float::half(ao[j].x));
		}

		if (store_secondary_set)
		{
			// std::cout << "Store secondary set: " << m->name << "\n";
			add_string(data, m->name);
			add_bytes(data, int(3));
			add_bytes(data, m->vertices[0]);
			add_bytes(data, int(m->vertices.size()));
			for (auto j = 0U; j < m->vertices.size(); j++)
			{
				add_bytes(data, half_float::half(ao[j].y));
			}
		}
	}

	remove(destination.string().c_str());
	mz_zip_add_mem_to_archive_file_in_place(destination.string().c_str(), "Patch.data",
		data.c_str(), data.size(), nullptr, 0, 0);

	auto config = params.extra_config;
	config.set("LIGHTING", "BRIGHTNESS", params.brightness);
	config.set("LIGHTING", "GAMMA", params.gamma);
	config.set("LIGHTING", "OPACITY", params.opacity);
	const auto config_data = config.to_string();
	mz_zip_add_mem_to_archive_file_in_place(destination.string().c_str(), "Config.ini",
		config_data.c_str(), config_data.size(), nullptr, 0, 0);
}

baked_data_mesh replace_primary_mesh_data(const baked_data_mesh& b, const baked_data_mesh& base)
{
	baked_data_mesh result;
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

baked_data_mesh merge_mesh_data(const baked_data_mesh& b, const baked_data_mesh& base, float mult_b)
{
	baked_data_mesh result;
	if (b.size() != base.size())
	{
		std::cerr << "\n[ ERROR: Sizes do not match. ]\n";
		exit(1);
	}
	result.resize(b.size());
	for (auto i = 0U; i < b.size(); i++)
	{
		result[i].x = std::max(b[i].x * mult_b, base[i].x);
		result[i].y = base[i].y;
	}
	return result;
}

baked_data_mesh average_mesh_data(const baked_data_mesh& b, const baked_data_mesh& base, float mult_b, float mult_base)
{
	baked_data_mesh result;
	if (b.size() != base.size())
	{
		std::cerr << "\n[ ERROR: Sizes do not match. ]\n";
		exit(1);
	}
	result.resize(b.size());
	for (auto i = 0U; i < b.size(); i++)
	{
		result[i].x = b[i].x * mult_b + base[i].x * mult_base;
		result[i].y = base[i].y;
	}
	return result;
}

void baked_data::replace_primary(const baked_data& b)
{
	for (const auto& p : b.entries)
	{
		auto f = entries.find(p.first);
		entries[p.first] = f == entries.end()
			? p.second
			: replace_primary_mesh_data(p.second, f->second);
	}
}

void baked_data::max(const baked_data& b, float mult_b)
{
	for (const auto& p : b.entries)
	{
		auto f = entries.find(p.first);
		entries[p.first] = f == entries.end()
			? p.second
			: merge_mesh_data(p.second, f->second, mult_b);
	}
}

void baked_data::replace(const baked_data& b)
{
	for (const auto& p : b.entries)
	{
		entries[p.first] = p.second;
	}
}

void baked_data::average(const baked_data& b, float mult_b, float mult_base)
{
	for (const auto& p : b.entries)
	{
		auto f = entries.find(p.first);
		entries[p.first] = f == entries.end()
			? p.second
			: average_mesh_data(p.second, f->second, mult_b, mult_base);
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

void baked_data::fill(const std::shared_ptr<bake::Mesh>& mesh, float x)
{
	if (!mesh) return;
	auto& entry = entries[mesh];
	entry.resize(mesh->vertices.size());
	for (auto j = 0U; j < mesh->vertices.size(); j++)
	{
		entry[j] = float2{x, x};
	}
}
