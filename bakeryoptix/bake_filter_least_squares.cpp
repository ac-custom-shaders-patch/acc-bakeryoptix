/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This code resamples occlusion computed at face sample points onto vertices using 
// the method from the following paper:
// Least Squares Vertex Baking, L. Kavan, A. W. Bargteil, P.-P. Sloan, EGSR 2011
// 
// Adapted from code written originally by Peter-Pike Sloan.

#include <bake_api.h>
#include <bake_filter_least_squares.h>
#include <bake_util.h>

#include <cassert>
#include <vector>
#include <optixu/optixu_math_namespace.h>

#include <eigen/Core>
#include <eigen/Geometry>
#include <eigen/SparseCholesky>

using namespace optix;

typedef double scalar_type;
typedef Eigen::SparseMatrix<scalar_type> sparse_matrix;
typedef Eigen::Matrix<scalar_type, 2, 3> matrix23;
typedef Eigen::Matrix<scalar_type, 2, 4> matrix24;
typedef Eigen::Matrix<scalar_type, 4, 4> matrix44;
typedef Eigen::Triplet<scalar_type> triplet;
typedef Eigen::Matrix<scalar_type, 3, 1> vector3;
typedef Eigen::Matrix<scalar_type, 2, 1> vector2;

namespace
{
	scalar_type triangle_area(const vector3& a, const vector3& b, const vector3& c)
	{
		const vector3 ba = b - a;
		const vector3 ca = c - a;
		auto crop = ba.cross(ca);
		return crop.norm() * 0.5f;
	}

	// Embeds 3D triangle v[0], v[1], v[2] into a plane, such that:
	//  p[0] = (0, 0), p[1] = (0, positive number), p[2] = (positive number, any number)
	// If triangle is close to degenerate returns false and p is undefined.
	bool planarize_triangle(const vector3 v[3], vector2 p[3])
	{
		double l01 = (v[0] - v[1]).norm();
		double l02 = (v[0] - v[2]).norm();
		double l12 = (v[1] - v[2]).norm();

		const double eps = 0.0;
		if (l01 <= eps || l02 <= eps || l12 <= eps) return false;

		double p2y = (l02 * l02 + l01 * l01 - l12 * l12) / (2.0 * l01);
		double tmp1 = l02 * l02 - p2y * p2y;
		if (tmp1 <= eps) return false;

		p[0] = vector2(0.0f, 0.0f);
		p[1] = vector2(0.0f, l01);
		p[2] = vector2(sqrt(tmp1), p2y);
		return true;
	}

	// Computes gradient operator (2 x 3 matrix 'grad') for a planar triangle.  If
	// 'normalize' is false then division by determinant is off (and thus the
	// routine cannot fail even for degenerate triangles).
	bool tri_grad_2d(const vector2 p[3], const bool normalize, matrix23& grad)
	{
		auto det = 1.0;
		if (normalize)
		{
			det = -double(p[0](1)) * p[1](0) + double(p[0](0)) * p[1](1) + double(p[0](1)) * p[2](0)
					- double(p[1](1)) * p[2](0) - double(p[0](0)) * p[2](1) + double(p[1](0)) * p[2](1);
			const auto eps = 0.0;
			if (fabs(det) <= eps)
			{
				return false;
			}
		}

		grad(0, 0) = p[1](1) - p[2](1);
		grad(0, 1) = p[2](1) - p[0](1);
		grad(0, 2) = p[0](1) - p[1](1);

		grad(1, 0) = p[2](0) - p[1](0);
		grad(1, 1) = p[0](0) - p[2](0);
		grad(1, 2) = p[1](0) - p[0](0);

		grad /= det;
		return true;
	}

	// Computes difference of gradients operator (2 x 4 matrix 'GD') for a butterfly, i.e., 
	// two edge-adjacent triangles.
	// Performs normalization so that units are [m], so GD^T * GD will have units of area [m^2]:
	bool butterfly_grad_diff(const vector3 v[4], matrix24& gd)
	{
		vector3 v1[3] = {v[0], v[1], v[2]};
		vector3 v2[3] = {v[0], v[1], v[3]};
		vector2 p1[3], p2[3];
		bool success = planarize_triangle(v1, p1);
		if (!success) return false;
		success = planarize_triangle(v2, p2);
		if (!success) return false;
		p2[2](0) *= -1.0; // flip the x coordinate of the last vertex of the second triangle so we get a butterfly

		matrix23 grad1, grad2;
		success = tri_grad_2d(p1, /*normalize*/ true, grad1);
		if (!success) return false;
		success = tri_grad_2d(p2, true, grad2);
		if (!success) return false;

		matrix24 grad_ext1, grad_ext2;
		for (auto i = 0; i < 2; i++)
		{
			grad_ext1(i, 0) = grad1(i, 0);
			grad_ext1(i, 1) = grad1(i, 1);
			grad_ext1(i, 2) = grad1(i, 2);
			grad_ext1(i, 3) = 0.0;
			grad_ext2(i, 0) = grad2(i, 0);
			grad_ext2(i, 1) = grad2(i, 1);
			grad_ext2(i, 2) = 0.0;
			grad_ext2(i, 3) = grad2(i, 2);
		}
		gd = grad_ext1 - grad_ext2;

		const auto area1 = triangle_area(v1[0], v1[1], v1[2]);
		const auto area2 = triangle_area(v2[0], v2[1], v2[2]);
		gd *= area1 + area2;

		return true;
	}

	struct butterfly
	{
		std::pair<int, int> wingverts;
		int count;
		butterfly() : wingverts(std::make_pair(-1, -1)), count(0) {}
	};

	typedef std::map<std::pair<int, int>, butterfly> edge_map;

	const float3* get_vertex(const float* v, unsigned stride_bytes, int index)
	{
		return reinterpret_cast<const float3*>(reinterpret_cast<const unsigned char*>(v) + index * stride_bytes);
	}

	void edge_based_regularizer(const float* verts, size_t num_verts, unsigned vertex_stride_bytes, const int3* faces, const size_t num_faces,
		sparse_matrix& regularization_matrix)
	{
		const unsigned stride_bytes = vertex_stride_bytes > 0 ? vertex_stride_bytes : 3 * sizeof(float);

		// Build edge map.  Each non-boundary edge stores the two opposite "butterfly" vertices that do not lie on the edge.
		edge_map edges;

		for (size_t i = 0; i < num_faces; i++)
		{
			const int indices[] = {faces[i].x, faces[i].y, faces[i].z};
			for (auto k = 0; k < 3; ++k)
			{
				const auto index0 = std::min(indices[k], indices[(k + 1) % 3]);
				const auto index1 = std::max(indices[k], indices[(k + 1) % 3]);
				const auto edge = std::make_pair(index0, index1);

				if (index0 == indices[k])
				{
					edges[edge].wingverts.first = indices[(k + 2) % 3]; // butterfly vert on left side of edge, ccw
					edges[edge].count++;
				}
				else
				{
					edges[edge].wingverts.second = indices[(k + 2) % 3]; // and right side 
					edges[edge].count++;
				}
			}
		}

		size_t skipped = 0;

		std::vector<triplet> triplets;
		size_t edge_index = 0;
		for (edge_map::const_iterator it = edges.begin(); it != edges.end(); ++it, ++edge_index)
		{
			if (it->second.count != 2)
			{
				continue; // not an interior edge, ignore
			}

			int vert_idx[4] = {it->first.first, it->first.second, it->second.wingverts.first, it->second.wingverts.second};
			if (it->second.wingverts.first < 0 || it->second.wingverts.second < 0)
			{
				continue; // duplicate face, ignore
			}

			vector3 butterfly_verts[4];
			for (size_t i = 0; i < 4; i++)
			{
				const float3* v = get_vertex(verts, stride_bytes, vert_idx[i]);
				butterfly_verts[i] = vector3(v->x, v->y, v->z);
			}

			matrix24 gd;
			if (!butterfly_grad_diff(butterfly_verts, gd))
			{
				skipped++;
				continue;
			}

			matrix44 gd_t_gd = gd.transpose() * gd; // units will now be [m^2]

			// scatter GDtGD:
			for (auto i = 0; i < 4; i++)
			{
				for (auto j = 0; j < 4; j++)
				{
					triplets.emplace_back(vert_idx[i], vert_idx[j], gd_t_gd(i, j));
				}
			}
		}

		regularization_matrix.resize(int(num_verts), int(num_verts));
		regularization_matrix.setFromTriplets(triplets.begin(), triplets.end());

		if (skipped > 0)
		{
			// std::cerr << "edgeBasedRegularizer: skipped " << skipped << " edges out of " << edges.size() << std::endl;
		}
	}

	void build_regularization_matrix(
		const bake::Mesh* mesh,
		sparse_matrix& regularization_matrix)
	{
		const int3* tri_vertex_indices = reinterpret_cast<int3*>(mesh->tri_vertex_indices);
		edge_based_regularizer(mesh->vertices, mesh->num_vertices, mesh->vertex_stride_bytes, tri_vertex_indices, mesh->num_triangles, regularization_matrix);
	}

	void filter_mesh_least_squares(
		const bake::Mesh* mesh,
		const bake::AOSamples& ao_samples,
		const float* ao_values,
		const float regularization_weight,
		const sparse_matrix& regularization_matrix,
		float* vertex_ao)
	{
		std::fill(vertex_ao, vertex_ao + mesh->num_vertices, 0.0f);
		
		std::vector<triplet> triplets;
		triplets.reserve(ao_samples.num_samples * 9);
		const int3* tri_vertex_indices = reinterpret_cast<int3*>(mesh->tri_vertex_indices);

		for (size_t i = 0; i < ao_samples.num_samples; i++)
		{
			const auto& info = ao_samples.sample_infos[i];
			const auto& tri = tri_vertex_indices[info.tri_idx];
			const auto val = ao_values[i] * info.dA;

			vertex_ao[tri.x] += info.bary[0] * val;
			vertex_ao[tri.y] += info.bary[1] * val;
			vertex_ao[tri.z] += info.bary[2] * val;

			// Note: the reference paper suggests computing the mass matrix analytically.
			// Building it from samples gave smoother results for low numbers of samples per face.

			triplets.emplace_back(tri.x, tri.x, static_cast<scalar_type>(info.bary[0] * info.bary[0] * info.dA));
			triplets.emplace_back(tri.y, tri.y, static_cast<scalar_type>(info.bary[1] * info.bary[1] * info.dA));
			triplets.emplace_back(tri.z, tri.z, static_cast<scalar_type>(info.bary[2] * info.bary[2] * info.dA));

			{
				const auto elem = static_cast<scalar_type>(info.bary[0] * info.bary[1] * info.dA);
				triplets.emplace_back(tri.x, tri.y, elem);
				triplets.emplace_back(tri.y, tri.x, elem);
			}

			{
				const auto elem = static_cast<scalar_type>(info.bary[1] * info.bary[2] * info.dA);
				triplets.emplace_back(tri.y, tri.z, elem);
				triplets.emplace_back(tri.z, tri.y, elem);
			}

			{
				const auto elem = static_cast<scalar_type>(info.bary[2] * info.bary[0] * info.dA);
				triplets.emplace_back(tri.x, tri.z, elem);
				triplets.emplace_back(tri.z, tri.x, elem);
			}
		}

		// Mass matrix
		sparse_matrix mass_matrix(int(mesh->num_vertices), int(mesh->num_vertices));
		mass_matrix.setFromTriplets(triplets.begin(), triplets.end());

		// Fix missing data due to unreferenced verts
		{
			const Eigen::VectorXd ones = Eigen::VectorXd::Constant(mesh->num_vertices, 1.0);
			Eigen::VectorXd lumped = mass_matrix * ones;
			for (auto i = 0U; i < mesh->num_vertices; i++)
			{
				if (lumped(i) <= 0.0)
				{
					// all valid entries in mass matrix are > 0
					mass_matrix.coeffRef(i, i) = 1.0;
				}
			}
		}
		
		Eigen::SimplicialLDLT<sparse_matrix> solver;

		// Optional edge-based regularization for smoother result, see paper for details
		if (regularization_weight > 0.0f)
		{
			const sparse_matrix a = mass_matrix + regularization_weight * regularization_matrix;
			solver.compute(a);
		}
		else
		{
			solver.compute(mass_matrix);
		}

		assert( solver.info() == Eigen::Success );

		Eigen::VectorXd b(mesh->num_vertices);
		Eigen::VectorXd x(mesh->num_vertices);
		for (size_t k = 0; k < mesh->num_vertices; ++k)
		{
			b(k) = vertex_ao[k];
			x(k) = 0.0;
		}

		x = solver.solve(b);

		assert( solver.info() == Eigen::Success ); // for debug build
		if (solver.info() == Eigen::Success)
		{
			for (size_t k = 0; k < mesh->num_vertices; ++k)
			{
				vertex_ao[k] = static_cast<float>(x(k)); // Note: allow out-of-range values
			}
		}
	}
}

void bake::filter_least_squares(
	const Scene& scene,
	const size_t* num_samples_per_instance,
	const AOSamples& ao_samples,
	const float* ao_values,
	const float regularization_weight,
	float** vertex_ao)
{
	std::vector<size_t> sample_offset_per_instance(scene.receivers.size());
	{
		size_t sample_offset = 0;
		for (size_t i = 0; i < scene.receivers.size(); i++)
		{
			sample_offset_per_instance[i] = sample_offset;
			sample_offset += num_samples_per_instance[i];
		}
	}

#pragma omp parallel for
	for (ptrdiff_t mesh_idx = 0; mesh_idx < ptrdiff_t(scene.receivers.size()); mesh_idx++)
	{
		// Build reg. matrix once, it does not depend on rigid xform per instance
		sparse_matrix regularization_matrix;
		if (regularization_weight > 0.0f)
		{
			build_regularization_matrix(scene.receivers[mesh_idx].get(), regularization_matrix);
		}

		// Filter all the instances that point to this mesh
#pragma omp parallel for
		for (ptrdiff_t i = 0; i < ptrdiff_t(scene.receivers.size()); i++)
		{
			if (i == mesh_idx)
			{
				const auto sample_offset = sample_offset_per_instance[i];

				// Point to samples for this instance
				AOSamples instance_ao_samples{};
				instance_ao_samples.num_samples = num_samples_per_instance[i];
				instance_ao_samples.sample_positions = ao_samples.sample_positions + 3 * sample_offset;
				instance_ao_samples.sample_normals = ao_samples.sample_normals + 3 * sample_offset;
				instance_ao_samples.sample_face_normals = ao_samples.sample_face_normals + 3 * sample_offset;
				instance_ao_samples.sample_infos = ao_samples.sample_infos + sample_offset;

				const auto instance_ao_values = ao_values + sample_offset;
				filter_mesh_least_squares(scene.receivers[mesh_idx].get(), instance_ao_samples, instance_ao_values, regularization_weight, regularization_matrix,
					vertex_ao[i]);
			}
		}
	}
}
