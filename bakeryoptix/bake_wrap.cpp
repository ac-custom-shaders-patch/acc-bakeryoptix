#include "bake_wrap.h"

#include <cassert>

#include <utils/perf_moment.h>
#include <utils/load_util.h>
#include <bake_ao_optix_prime.h>
#include <bake_api.h>
#include <bake_sample.h>

inline void set_vertex_entry(float* vertices, const int idx, const int axis, float* vec)
{
	vertices[3 * idx + axis] = vec[axis];
}

inline void set_vertex_value(float* vertices, const int idx, const int axis, float vec)
{
	vertices[3 * idx + axis] = vec;
}

static void make_ground_plane(const float scene_bbox_min[3], const float scene_bbox_max[3], int upaxis, const float scale_factor,
	const float offset_factor, std::vector<bake::Mesh*>& meshes, bake::Scene& scene)
{
	auto plane_mesh = new bake::Mesh();
	plane_mesh->name = "blocker";
	plane_mesh->vertices.resize(4);
	plane_mesh->triangles = {
		{0, 1, 2},
		{0, 2, 3},
		{2, 1, 0},
		{3, 2, 0}
	};

	float scene_extents[] = {
		scene_bbox_max[0] - scene_bbox_min[0],
		scene_bbox_max[1] - scene_bbox_min[1],
		scene_bbox_max[2] - scene_bbox_min[2]
	};

	float ground_min[] = {
		scene_bbox_max[0] - scale_factor * scene_extents[0],
		scene_bbox_min[1] - scale_factor * scene_extents[1],
		scene_bbox_max[2] - scale_factor * scene_extents[2]
	};
	float ground_max[] = {
		scene_bbox_min[0] + scale_factor * scene_extents[0],
		scene_bbox_min[1] + scale_factor * scene_extents[1],
		scene_bbox_min[2] + scale_factor * scene_extents[2]
	};

	if (upaxis > 2)
	{
		upaxis %= 3;
		ground_min[upaxis] = scene_bbox_max[upaxis] + scene_extents[upaxis] * offset_factor;
		ground_max[upaxis] = scene_bbox_max[upaxis] + scene_extents[upaxis] * offset_factor;
	}
	else
	{
		ground_min[upaxis] = scene_bbox_min[upaxis] - scene_extents[upaxis] * offset_factor;
		ground_max[upaxis] = scene_bbox_min[upaxis] - scene_extents[upaxis] * offset_factor;
	}

	const auto axis0 = (upaxis + 2) % 3;
	const auto axis1 = (upaxis + 1) % 3;

	float vertex_data[4 * 3] = {};
	set_vertex_entry(vertex_data, 0, upaxis, ground_min);
	set_vertex_entry(vertex_data, 0, axis0, ground_min);
	set_vertex_entry(vertex_data, 0, axis1, ground_min);

	set_vertex_entry(vertex_data, 1, upaxis, ground_min);
	set_vertex_entry(vertex_data, 1, axis0, ground_max);
	set_vertex_entry(vertex_data, 1, axis1, ground_min);

	set_vertex_entry(vertex_data, 2, upaxis, ground_min);
	set_vertex_entry(vertex_data, 2, axis0, ground_max);
	set_vertex_entry(vertex_data, 2, axis1, ground_max);

	set_vertex_entry(vertex_data, 3, upaxis, ground_min);
	set_vertex_entry(vertex_data, 3, axis0, ground_min);
	set_vertex_entry(vertex_data, 3, axis1, ground_max);

	for (size_t i = 0; i < 4; ++i)
	{
		plane_mesh->vertices[i].pos.x = vertex_data[3 * i];
		plane_mesh->vertices[i].pos.y = vertex_data[3 * i + 1];
		plane_mesh->vertices[i].pos.z = vertex_data[3 * i + 2];
	}

	plane_mesh->matrix = bake::NodeTransformation::identity();
	expand_bbox(scene.bbox_min, scene.bbox_max, ground_min);
	expand_bbox(scene.bbox_min, scene.bbox_max, ground_max);
	meshes.push_back(plane_mesh);
}

static void make_emissive_plane(const bake_params::light_emitter& emitter, std::vector<bake::Mesh*>& meshes)
{
	auto plane_mesh = new bake::Mesh();
	plane_mesh->name = "emissive";
	plane_mesh->lod_out = emitter.intensity;
	plane_mesh->vertices.resize(4);
	plane_mesh->triangles = {
		{0, 1, 2},
		{0, 2, 3},
		{2, 1, 0},
		{3, 2, 0}
	};
	
	for (size_t i = 0; i < 4; ++i)
	{
		plane_mesh->vertices[i].pos = emitter.pos[i];
	}

	plane_mesh->matrix = bake::NodeTransformation::identity();
	meshes.push_back(plane_mesh);
}

static void allocate_ao_samples(bake::AOSamples& ao_samples, size_t n)
{
	ao_samples.num_samples = n;
	ao_samples.sample_positions = new float[3 * n];
	ao_samples.sample_normals = new float[3 * n];
	ao_samples.sample_face_normals = new float[3 * n];
	ao_samples.sample_infos = new bake::SampleInfo[n];
}

static void destroy_ao_samples(bake::AOSamples& ao_samples)
{
	delete [] ao_samples.sample_positions;
	ao_samples.sample_positions = nullptr;
	delete [] ao_samples.sample_normals;
	ao_samples.sample_normals = nullptr;
	delete [] ao_samples.sample_face_normals;
	ao_samples.sample_face_normals = nullptr;
	delete [] ao_samples.sample_infos;
	ao_samples.sample_infos = nullptr;
	ao_samples.num_samples = 0;
}

baked_data bake_wrap::bake_scene(const std::shared_ptr<bake::Scene>& scene, 
	const std::vector<std::shared_ptr<bake::Mesh>>& blockers, 
	const bake_params& config, bool verbose)
{
	#undef PERF
	#define PERF(NAME)\
		perf_moment PERF_UNIQUE(NAME, verbose);

	if (scene->receivers.empty())
	{
		// std::cerr << "\n[ WARNING: No shadow receivers. ]\n";
		return baked_data{};
	}

	// Generate AO samples
	std::vector<size_t> num_samples_per_instance(scene->receivers.size());
	bake::AOSamples ao_samples{};
	size_t total_samples;
	{
		{
			PERF("\tGenerating sample points")

			if (!scene->extra_receive_points.empty())
			{
				total_samples = scene->extra_receive_points.size();
				allocate_ao_samples(ao_samples, total_samples);

				auto sample_index = 0;
				const auto sample_positions = reinterpret_cast<bake::Vec3*>(ao_samples.sample_positions);
				const auto sample_norms = reinterpret_cast<bake::Vec3*>(ao_samples.sample_normals);
				const auto sample_face_norms = reinterpret_cast<bake::Vec3*>(ao_samples.sample_face_normals);
				for (const auto& p : scene->extra_receive_points)
				{
					sample_positions[sample_index] = {p.x + config.sample_offset.x, p.y + config.sample_offset.y, p.z + config.sample_offset.z};
					sample_norms[sample_index] = {0.f, 1.f, 0.f};
					sample_face_norms[sample_index] = {0.f, 1.f, 0.f};
					sample_index++;
				}
			}
			else if (!scene->extra_receive_directed_points.empty())
			{
				total_samples = scene->extra_receive_directed_points.size();
				allocate_ao_samples(ao_samples, total_samples);
				ao_samples.align_rays = 4.f;

				auto sample_index = 0;
				const auto sample_positions = reinterpret_cast<bake::Vec3*>(ao_samples.sample_positions);
				const auto sample_norms = reinterpret_cast<bake::Vec3*>(ao_samples.sample_normals);
				const auto sample_face_norms = reinterpret_cast<bake::Vec3*>(ao_samples.sample_face_normals);
				for (const auto& p : scene->extra_receive_directed_points)
				{
					sample_positions[sample_index] = {p.first.x + config.sample_offset.x, p.first.y + config.sample_offset.y, p.first.z + config.sample_offset.z};
					sample_norms[sample_index] = p.second;
					sample_face_norms[sample_index] = p.second;
					sample_index++;
				}
			}
			else if (config.sample_on_points)
			{
				total_samples = 0;
				for (const auto& m : scene->receivers)
				{
					total_samples += m->vertices.size();
				}
				allocate_ao_samples(ao_samples, total_samples);
				auto sample_index = 0;
				const auto sample_positions = reinterpret_cast<bake::Vec3*>(ao_samples.sample_positions);
				const auto sample_norms = reinterpret_cast<bake::Vec3*>(ao_samples.sample_normals);
				const auto sample_face_norms = reinterpret_cast<bake::Vec3*>(ao_samples.sample_face_normals);
				for (const auto& m : scene->receivers)
				{
					for (const auto& p : m->vertices)
					{
						sample_positions[sample_index] = {p.pos.x + config.sample_offset.x, p.pos.y + config.sample_offset.y, p.pos.z + config.sample_offset.z};
						sample_norms[sample_index] = {0.f, 1.f, 0.f};
						sample_face_norms[sample_index] = {0.f, 1.f, 0.f};
						sample_index++;
					}
				}
			}
			else
			{
				total_samples = distribute_samples(*scene,
					config.min_samples_per_face, config.num_samples, &num_samples_per_instance[0]);
				allocate_ao_samples(ao_samples, total_samples);
				sample_instances(*scene, &num_samples_per_instance[0], config.min_samples_per_face, 
					config.disable_normals, config.missing_normals_up, config.fix_incorrect_normals, ao_samples);
			}
		}
		if (verbose) std::cout << "\tTotal samples: " << total_samples << std::endl;
	}

	// Computing AO
	std::vector<float> ao_values(total_samples);
	{
		PERF("\tComputing AO");

		std::fill(ao_values.begin(), ao_values.end(), 0.f);
		std::vector<bake::Mesh*> blocker_meshes;
		for (const auto& m : blockers)
		{
			if (m->cast_shadows)
			{
				blocker_meshes.push_back(m.get());
			}
		}

		if (config.use_ground_plane_blocker)
		{
			make_ground_plane(scene->bbox_min, scene->bbox_max, config.ground_upaxis, config.ground_scale_factor, config.ground_offset_factor, blocker_meshes, *scene);
		}

		for (const auto& c : config.light_emitters)
		{
			make_emissive_plane(c, blocker_meshes);
		}

		ao_optix_prime(blocker_meshes, ao_samples, config.num_rays, config.scene_albedo, uint32_t(config.bounce_counts),
			config.scene_offset_scale_horizontal, config.scene_offset_scale_vertical, config.trees_light_pass_chance,
			config.stack_size, config.batch_size, config.debug_mode, &ao_values[0]);
	}

	// Mapping AO to vertices
	baked_data result;

	auto vertex_ao = new float*[scene->receivers.size()];
	{
		PERF("\tMapping AO to vertices");
		for (size_t i = 0U; i < scene->receivers.size(); i++)
		{
			vertex_ao[i] = new float[scene->receivers[i]->vertices.size()];
		}

		if (!scene->extra_receive_points.empty())
		{
			result.extra_points_ao.resize(scene->extra_receive_points.size());
			for (auto i = 0U; i < scene->extra_receive_points.size(); ++i)
			{
				result.extra_points_ao[i] = ao_values[i];
			}
		}
		else if (!scene->extra_receive_directed_points.empty())
		{
			result.extra_points_ao.resize(scene->extra_receive_directed_points.size());
			for (auto i = 0U; i < scene->extra_receive_directed_points.size(); ++i)
			{
				result.extra_points_ao[i] = ao_values[i];
			}
		}
		else if (config.sample_on_points)
		{
			auto index = 0U;
			for (size_t i = 0U; i < scene->receivers.size(); i++)
			{
				const auto& m = scene->receivers[i];
				for (auto j = 0U; j < m->vertices.size(); j++)
				{
					vertex_ao[i][j] = ao_values[index++];
				}
			}
		}
		else
		{
			map_ao_to_vertices(*scene, &num_samples_per_instance[0], ao_samples, &ao_values[0], config.filter_mode, config.regularization_weight, vertex_ao);
		}
	}

	// Rearranging result
	for (auto i = 0U; i < scene->receivers.size(); i++)
	{
		auto& m = scene->receivers[i];
		baked_data_mesh_set v(m->vertices.size());
		for (auto j = 0U; j < m->vertices.size(); j++)
		{
			v[j].x = v[j].y = vertex_ao[i][j];
		}
		result.entries[scene->receivers[i]].main_set = v;
	}

	// Releasing some crap
	for (size_t i = 0; i < scene->receivers.size(); ++i)
	{
		delete [] vertex_ao[i];
	}
	delete [] vertex_ao;
	destroy_ao_samples(ao_samples);

	// Returning result
	return result;
}
