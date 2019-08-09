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
#include <iomanip>

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

void baked_data::save(const utils::path& destination, const save_params& params) const
{
	using namespace optix;
	std::basic_string<uint8_t> data;

	std::vector<unsigned> nearbyes;
	std::vector<bool> averaged;

	for (const auto& entry : entries)
	{
		const auto m = entry.first;
		auto ao = entry.second;

		add_string(data, m->name);
		add_bytes(data, int(1));
		add_bytes(data, m->vertices[0]);
		add_bytes(data, m->vertices[1]);
		add_bytes(data, m->vertices[2]);
		add_bytes(data, int(m->num_vertices));

		if (params.averaging_threshold > 0.f)
		{
			if (m->num_vertices > nearbyes.size())
			{
				nearbyes.resize(m->num_vertices);
				averaged.resize(m->num_vertices);
			}
			std::fill(averaged.begin(), averaged.end(), false);

			struct vertex
			{
				float pos;
				unsigned index;
			};

			std::vector<vertex> vertices_conv;
			vertices_conv.reserve(m->num_vertices);
			for (auto j = 0U; j < m->num_vertices; j++)
			{
				vertices_conv.push_back({m->vertices[j * 3], j});
			}
			std::sort(vertices_conv.begin(), vertices_conv.end(), [](const vertex& a, const vertex& b) { return a.pos > b.pos; });

			for (auto j2 = 0U; j2 < m->num_vertices; j2++)
			{
				const auto j = vertices_conv[j2].index;
				if (averaged[j]) continue;
				const auto& jp = *(float3*)&m->vertices[j * 3];
				const auto& jn = *(float3*)&m->normals[j * 3];
				auto t = ao[j];
				auto nearbyes_size = 0U;
				for (auto k2 = j2 + 1; k2 < m->num_vertices && k2 - j2 < 100; k2++)
				{
					const auto k = vertices_conv[k2].index;
					const auto& kn = *(float3*)&m->normals[k * 3];
					const auto nd = dot(jn, kn);
					if (nd < params.averaging_cos_threshold) continue;

					const auto& kp = *(float3*)&m->vertices[k * 3];
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
			}
		}

		// std::cout << "name=" << m->name << ": ";
		for (auto j = 0U; j < m->num_vertices; j++)
		{
			// std::cout << std::setprecision(2) << ao[j] << ", ";
			add_bytes(data, half_float::half(ao[j]));
		}
		// std::cout << "\n";
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
		result[i] = std::max(b[i] * mult_b, base[i]);
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
		result[i] = b[i] * mult_b + base[i] * mult_base;
	}
	return result;
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
	entry.resize(mesh->num_vertices);
	for (auto j = 0U; j < mesh->num_vertices; j++)
	{
		entry[j] = x;
	}
}
