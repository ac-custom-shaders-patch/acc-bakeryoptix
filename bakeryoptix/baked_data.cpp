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

DEBUGTIME

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

struct values_recorder
{
	std::basic_string<uint8_t>& data;
	const save_params& params;

	enum alt_state : int
	{
		MAIN = 0,
		SECONDARY = 1,
		ALTERNATIVE = 2
	};

	byte adjust(float value) const
	{
		return byte(255.f * powf(powf(optix::clamp((1.f - params.opacity + value * params.opacity) * params.brightness, 0.f, 1.f), params.gamma), VAO_ENCODE_POW));
	}

	template <typename T>
	void add_values(const std::shared_ptr<bake::Mesh>& m, int state, T fn)
	{
		const auto name = (state & ALTERNATIVE) ? "@@__ALT@:" + m->name : m->name;
		data.reserve(data.size() + name.size() + m->vertices.size() + 24);
		add_string(data, name);
		add_bytes(data, int((state & SECONDARY) ? 3 : 1));
		add_bytes(data, m->signature_point);
		add_bytes(data, int(m->vertices.size()));
		for (size_t j = 0U; j < m->vertices.size(); j++)
		{
			add_bytes(data, adjust(fn(j)));
		}
	}

	void add_values(const std::shared_ptr<bake::Mesh>& m, const baked_data_mesh& ao, int state)
	{
		if ((state & SECONDARY) == 0 && (state & ALTERNATIVE) == 0)
		{
			add_values(m, state, [=](size_t j) { return ao.main_set[j].x; });
		}

		if ((state & SECONDARY) != 0 && (state & ALTERNATIVE) == 0)
		{
			add_values(m, state, [=](size_t j) { return ao.main_set[j].y; });
		}

		if ((state & SECONDARY) == 0 && (state & ALTERNATIVE) != 0)
		{
			add_values(m, state, [=](size_t j) { return ao.alternative_set[j].x; });
		}

		if ((state & SECONDARY) != 0 && (state & ALTERNATIVE) != 0)
		{
			add_values(m, state, [=](size_t j) { return ao.alternative_set[j].y; });
		}
	}
};

inline size_t vertex_key(const bake::MeshVertex& t, const bake::Vec3& n, float mult, float normal_contribution)
{
	auto v = bake::Vec3{
		roundf((t.pos.x + n.x * normal_contribution) * mult),
		roundf((t.pos.y + n.y * normal_contribution) * mult),
		roundf((t.pos.z + n.z * normal_contribution) * mult)};
	return *(size_t*)&v.x * 397 ^ size_t(*(uint32_t*)&v.z);
}

void baked_data::smooth_ao()
{
	struct data
	{
		float2 main, alternative;

		void add(const baked_data_mesh& m, uint32_t i)
		{
			main.x = std::max(main.x, m.main_set[i].x);
			main.y = std::max(main.y, m.main_set[i].y);
			if (!m.alternative_set.empty())
			{
				alternative.x = std::max(alternative.x, m.alternative_set[i].x);
				alternative.y = std::max(alternative.y, m.alternative_set[i].y);
			}
		}

		void fill(baked_data_mesh& m, uint32_t i) const
		{
			m.main_set[i] = main;
			if (!m.alternative_set.empty())
			{
				m.alternative_set[i] = alternative;
			}
		}
	};

	std::unordered_map<size_t, data> v;

	const auto mult = 1.f / 0.005f;
	const auto normal_contribution = 0.01f;
	
	cout_progress progress_smooth{entries.size() * 2};
	for (const auto& m : entries)
	{
		progress_smooth.report();
		if (m.first->normals.empty()) continue;
		
		auto i = 0U;
		for (auto& t : m.first->vertices)
		{
			const auto k = vertex_key(t, m.first->normals[i], mult, normal_contribution);
			const auto f = v.find(k);
			if (f != v.end())
			{
				f->second.add(m.second, i);
			}
			else
			{
				(v[k] = data()).add(m.second, i);
			}
			++i;
		}
	}
	for (auto& m : entries)
	{
		progress_smooth.report();
		if (m.first->normals.empty()) continue;
		
		auto i = 0U;
		for (auto& t : m.first->vertices)
		{
			v[vertex_key(t, m.first->normals[i], mult, normal_contribution)].fill(m.second, i);
			++i;
		}
	}
}

void baked_data::save(const utils::path& destination, const save_params& params, bool store_secondary_set)
{
	using namespace optix;
	std::basic_string<uint8_t> data;

	std::vector<unsigned> nearbyes;
	std::vector<bool> averaged;
	
	remove(destination.string().c_str());
	mz_zip_archive zip_archive;
	mz_zip_zero_struct(&zip_archive);
	if (!mz_zip_writer_init_file_v2(&zip_archive, destination.string().c_str(), 0, MZ_BEST_COMPRESSION))
	{
		throw std::runtime_error("Failed to initialize new archive");
	}
	
	auto config = params.extra_config;
	const auto config_data = config.to_string();
	mz_zip_writer_add_mem(&zip_archive, "Config.ini", config_data.c_str(), config_data.size(), MZ_BEST_COMPRESSION);

	for (auto& e : extra_entries)
	{
		mz_zip_writer_add_mem(&zip_archive, e.first.c_str(), e.second.data(), e.second.size(), MZ_BEST_COMPRESSION);
	}

	values_recorder rec{data, params};	
	cout_progress progress_main{entries.size()};
	for (const auto& entry : entries)
	{
		progress_main.report();

		const auto m = entry.first;
		auto ao = entry.second;

		if (m->name == "@@__EXTRA_AO@")
		{
			data.reserve(data.size() + m->name.size() + m->vertices.size() * 11 + 24);
			add_string(data, m->name);
			add_bytes(data, int(4));
			add_bytes(data, m->signature_point);

			std::vector<float4> cleaned_up;
			cleaned_up.reserve(m->vertices.size());
			for (auto j = 0U; j < m->vertices.size(); j++)
			{
				const auto& v = m->vertices[j].pos;
				cleaned_up.push_back(make_float4(v.x, v.y, v.z, ao.main_set[j].x));
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
				add_bytes(data, *(const bake::Vec3*)&j);
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
				vertices_conv.push_back({m->vertices[j].pos.x, j});
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

		rec.add_values(m, ao, values_recorder::MAIN);
		if (store_secondary_set)
		{
			rec.add_values(m, ao, values_recorder::SECONDARY);
		}

		if (!ao.alternative_set.empty())
		{
			rec.add_values(m, ao, values_recorder::ALTERNATIVE);
			if (store_secondary_set)
			{
				rec.add_values(m, ao, values_recorder::ALTERNATIVE | values_recorder::SECONDARY);
			}
		}
	}

	mz_zip_writer_add_mem(&zip_archive, "Patch_v5.data", data.c_str(), data.size(), MZ_BEST_COMPRESSION);
	mz_zip_writer_finalize_archive(&zip_archive);
	mz_zip_writer_end(&zip_archive);
}

template <typename TPrim>
void combine_sets(baked_data_mesh_set& dst, const baked_data_mesh_set& src, const TPrim& fn, bool apply_to_both)
{
	if (src.size() != dst.size())
	{
		std::cerr << "\n[ ERROR: Sizes do not match. ]\n";
		exit(1);
	}
	for (auto i = 0U; i < src.size(); i++)
	{
		dst[i].x = fn(dst[i].x, src[i].x);
		if (apply_to_both) dst[i].y = fn(dst[i].y, src[i].y);
	}
}

template <typename TPrim>
void combine_sets_entries(std::map<std::shared_ptr<bake::Mesh>, baked_data_mesh>& entries, const baked_data& b, const TPrim& fn, bool apply_to_both)
{
	for (const auto& p : b.entries)
	{
		auto f = entries.find(p.first);
		if (f == entries.end())
		{
			entries[p.first].main_set = p.second.main_set;
		}
		else
		{
			combine_sets(f->second.main_set, p.second.main_set, fn, apply_to_both);
		}
	}
}

template <typename TPrim>
void combine_sets_entries(std::map<std::shared_ptr<bake::Mesh>, baked_data_mesh>& entries, const baked_data& b, const TPrim& fn_merge)
{
	for (const auto& p : b.entries)
	{
		auto f = entries.find(p.first);
		if (f == entries.end())
		{
			entries[p.first].main_set = p.second.main_set;
		}
		else
		{
			fn_merge(p.first, f->second.main_set, p.second.main_set);
		}
	}
}

void baked_data::replace_primary(const baked_data& b)
{
	combine_sets_entries(entries, b, [](float d, float s) { return s; }, false);
}

void baked_data::brighten(const baked_data& b, float brighten_k)
{
	combine_sets_entries(entries, b, [=](float d, float s) { return d + std::max(s - d, 0.f) * brighten_k; }, true);
}

void baked_data::max(const baked_data& b, float mult_b, const std::vector<std::shared_ptr<bake::Mesh>>& inverse, bool apply_to_both_sets)
{
	combine_sets_entries(entries, b, [=](const std::shared_ptr<bake::Mesh>& m, baked_data_mesh_set& dst, const baked_data_mesh_set& src)
	{
		if (std::find(inverse.begin(), inverse.end(), m) != inverse.end())
		{
			combine_sets(dst, src, [=](float d, float s) { return std::min(s * mult_b, d); }, apply_to_both_sets);
		}
		else
		{
			combine_sets(dst, src, [=](float d, float s) { return std::max(s * mult_b, d); }, apply_to_both_sets);
		}
	});
}

void baked_data::average(const baked_data& b, float mult_b, float mult_base, const std::vector<std::shared_ptr<bake::Mesh>>& inverse, bool apply_to_both_sets)
{
	combine_sets_entries(entries, b, [=](const std::shared_ptr<bake::Mesh>& m, baked_data_mesh_set& dst, const baked_data_mesh_set& src)
	{
		if (std::find(inverse.begin(), inverse.end(), m) != inverse.end())
		{
			combine_sets(dst, src, [=](float d, float s) { return std::min(s, d); }, apply_to_both_sets);
		}
		else
		{
			combine_sets(dst, src, [=](float d, float s) { return s * mult_b + d * mult_base; }, apply_to_both_sets);
		}
	});
}

void baked_data::replace(const baked_data& b)
{
	for (const auto& p : b.entries)
	{
		entries[p.first] = p.second;
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
