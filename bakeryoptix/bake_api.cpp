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

#include <cassert>
#include <optixu/optixu_matrix_namespace.h>

#include <utils/load_scene_util.h>
#include <bake_api.h>
#include <bake_filter.h>
#include <bake_filter_least_squares.h>

using namespace optix;

bake::NodeTransformation to_transformation(const optix::Matrix4x4& matrix)
{
	bake::NodeTransformation result{};
	std::copy(matrix.getData(), matrix.getData() + 16, result._Elems);
	return result;
}

const optix::Matrix4x4& to_matrix(const bake::NodeTransformation& a)
{
	return *(const Matrix4x4*)&a;
}

bake::NodeTransformation bake::NodeTransformation::operator*(const NodeTransformation& b) const
{
	return to_transformation(to_matrix(*this) * to_matrix(b));
}

bake::NodeTransformation bake::NodeTransformation::transpose() const
{
	return to_transformation(to_matrix(*this).transpose());
}

bake::NodeTransformation bake::NodeTransformation::rotation(const float deg, const float dir[3])
{
	return to_transformation(Matrix4x4::rotate(deg * M_PIf / 180.f, float3{dir[0], dir[1], dir[2]}));
}

bake::NodeTransformation bake::NodeTransformation::identity()
{
	return to_transformation(Matrix4x4::identity());
}

void bake::Node::update_matrix()
{
	matrix = parent ? parent->matrix * matrix_local : matrix_local;
	for (const auto& c : children)
	{
		c->update_matrix();
	}
}

bake::NodeTransformation lerp_tranformation(const bake::NodeTransformation& a, const bake::NodeTransformation& b, float mix)
{
	bake::NodeTransformation result{};
	for (auto i = 0U; i < result.size(); i++)
	{
		result[i] = lerp(a[i], b[i], mix);
	}
	return result;
}

void bake::NodeTransition::apply(float progress) const
{
	progress = clamp(progress, 0.f, 1.f);

	const auto frame_prev = uint(floorf((frames.size() - 1) * progress));
	const auto frame_next = uint(ceilf((frames.size() - 1) * progress));
	if (frame_next == frame_prev)
	{
		node->matrix_local = frames[frame_next];
	}
	else
	{
		const auto mix = clamp((frames.size() - 1) * progress - frame_prev, 0.f, 1.f);
		node->matrix_local = lerp_tranformation(frames[frame_prev], frames[frame_next], mix);
	}
}

void bake::Animation::apply(float progress) const
{
	for (const auto& e : entries)
	{
		e.apply(progress);
	}
}

bake::Scene::Scene(const std::shared_ptr<Node>& root) : Scene(std::vector<std::shared_ptr<Node>>{root}) {}

bake::Scene::Scene(const std::vector<std::shared_ptr<Node>>& nodes)
{
	for (const auto& n : nodes)
	{
		n->update_matrix();
		for (const auto& m : n->get_meshes())
		{
			if (m->receive_shadows) receivers.push_back(m);
			if (m->cast_shadows) blockers.push_back(m);

			auto x = to_matrix(m->matrix);
			float4 f4;
			f4.w = 1.f;

			for (auto i = 0U; i < m->num_vertices; i += 3)
			{
				*(float3*)&f4 = *(float3*)&m->vertices[i * 3];
				const auto w = f4 * x;
				expand_bbox(bbox_min, bbox_max, &w.x);
			}
		}
	}
}

void bake::map_ao_to_vertices(
	const Scene& scene,
	const size_t* num_samples_per_instance,
	const AOSamples& ao_samples,
	const float* ao_values,
	const VertexFilterMode mode,
	const float regularization_weight,
	float** vertex_ao)
{
	if (mode == VERTEX_FILTER_AREA_BASED)
	{
		filter(scene, num_samples_per_instance, ao_samples, ao_values, vertex_ao);
	}
	else if (mode == VERTEX_FILTER_LEAST_SQUARES)
	{
		filter_least_squares(scene, num_samples_per_instance, ao_samples, ao_values, regularization_weight, vertex_ao);
	}
	else
	{
		assert(0 && "invalid vertex filter mode");
	}
}
