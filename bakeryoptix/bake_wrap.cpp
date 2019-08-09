#include "bake_wrap.h"

#include <cassert>
#include <optixu/optixu_matrix_namespace.h>

#include <utils/perf_moment.h>
#include <utils/load_scene_util.h>
#include <bake_ao_optix_prime.h>
#include <bake_api.h>
#include <bake_sample.h>

void set_vertex_entry(float* vertices, const int idx, const int axis, float* vec)
{
	vertices[3 * idx + axis] = vec[axis];
}

void make_ground_plane(const float scene_bbox_min[3], const float scene_bbox_max[3],
	unsigned upaxis, const float scale_factor, const float offset_factor,
	const unsigned scene_vertex_stride_bytes,
	std::vector<float>& plane_vertices, std::vector<unsigned int>& plane_indices,
	std::vector<bake::Mesh*>& meshes,
	bake::Scene& scene)
{
	const unsigned int index_data[] = {0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0};
	const unsigned int num_indices = sizeof(index_data) / sizeof(index_data[0]);
	plane_indices.resize(num_indices);
	std::copy(index_data, index_data + num_indices, plane_indices.begin());
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

	const int axis0 = (upaxis + 2) % 3;
	const int axis1 = (upaxis + 1) % 3;

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

	// OptiX Prime requires all meshes in the same scene to have the same vertex stride.
	const unsigned vertex_stride_bytes = scene_vertex_stride_bytes > 0 ? scene_vertex_stride_bytes : 3 * sizeof(float);
	assert(vertex_stride_bytes % sizeof(float) == 0);
	const unsigned num_floats_per_vert = vertex_stride_bytes / sizeof(float);
	plane_vertices.resize(4 * (num_floats_per_vert));
	std::fill(plane_vertices.begin(), plane_vertices.end(), 0.f);
	for (size_t i = 0; i < 4; ++i)
	{
		plane_vertices[num_floats_per_vert * i] = vertex_data[3 * i];
		plane_vertices[num_floats_per_vert * i + 1] = vertex_data[3 * i + 1];
		plane_vertices[num_floats_per_vert * i + 2] = vertex_data[3 * i + 2];
	}

	auto plane_mesh = new bake::Mesh();
	plane_mesh->num_vertices = 4;
	plane_mesh->num_triangles = num_indices / 3;
	plane_mesh->vertices = &plane_vertices[0];
	plane_mesh->vertex_stride_bytes = vertex_stride_bytes;
	plane_mesh->normals = nullptr;
	plane_mesh->normal_stride_bytes = 0;
	plane_mesh->tri_vertex_indices = &plane_indices[0];
	plane_mesh->matrix = bake::NodeTransformation::identity();
	expand_bbox(scene.bbox_min, scene.bbox_max, ground_min);
	expand_bbox(scene.bbox_min, scene.bbox_max, ground_max);
	meshes.push_back(plane_mesh);
}

void allocate_ao_samples(bake::AOSamples& ao_samples, size_t n)
{
	ao_samples.num_samples = n;
	ao_samples.sample_positions = new float[3 * n];
	ao_samples.sample_normals = new float[3 * n];
	ao_samples.sample_face_normals = new float[3 * n];
	ao_samples.sample_infos = new bake::SampleInfo[n];
}

void destroy_ao_samples(bake::AOSamples& ao_samples)
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

baked_data bake_wrap::bake_scene(const std::shared_ptr<bake::Scene>& scene, const std::vector<std::shared_ptr<bake::Mesh>> blockers,
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
			total_samples = distribute_samples(*scene,
				config.min_samples_per_face, config.num_samples, &num_samples_per_instance[0]);
			allocate_ao_samples(ao_samples, total_samples);
			sample_instances(*scene, &num_samples_per_instance[0], config.min_samples_per_face, ao_samples);
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

		std::vector<float> plane_vertices;
		std::vector<unsigned int> plane_indices;
		if (config.use_ground_plane_blocker)
		{
			make_ground_plane(scene->bbox_min, scene->bbox_max, config.ground_upaxis, config.ground_scale_factor, config.ground_offset_factor,
				scene->receivers[0]->vertex_stride_bytes, plane_vertices, plane_indices, blocker_meshes, *scene);
		}

		ao_optix_prime(blocker_meshes, ao_samples,
			config.num_rays, config.scene_offset_scale, config.scene_maxdistance_scale,
			scene->bbox_min, scene->bbox_max, &ao_values[0], config.use_cuda);
	}

	// Mapping AO to vertices
	auto vertex_ao = new float*[ scene->receivers.size() ];
	{
		PERF("\tMapping AO to vertices");
		for (size_t i = 0U; i < scene->receivers.size(); i++)
		{
			vertex_ao[i] = new float[ scene->receivers[i]->num_vertices ];
		}
		map_ao_to_vertices(*scene, &num_samples_per_instance[0], ao_samples, &ao_values[0], config.filter_mode, config.regularization_weight, vertex_ao);
	}

	// Rearranging result
	baked_data result;
	for (auto i = 0U; i < scene->receivers.size(); i++)
	{
		auto& m = scene->receivers[i];
		baked_data_mesh v(m->num_vertices);
		for (auto j = 0U; j < m->num_vertices; j++)
		{
			v[j] = vertex_ao[i][j];
		}
		result.entries[scene->receivers[i]] = v;
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
