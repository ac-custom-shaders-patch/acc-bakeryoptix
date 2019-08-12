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

#include <utils/load_util.h>
#include <utils/vector_operations.h>
#include <bake_api.h>
#include <bake_filter.h>
#include <bake_filter_least_squares.h>
#include <assert.h>
#include <iostream>

using namespace optix;
using namespace bake;

NodeTransformation to_transformation(const Matrix4x4& matrix)
{
	NodeTransformation result{};
	std::copy(matrix.getData(), matrix.getData() + 16, result._Elems);
	return result;
}

const Matrix4x4& to_matrix(const NodeTransformation& a)
{
	return *(const Matrix4x4*)&a;
}

float3 transform_pos(const float3& pos, const Matrix4x4& mat)
{
	float4 f4;
	f4.w = 1.f;
	*(float3*)&f4 = pos;
	const auto w = f4 * mat;
	return *(float3*)&w;
}

float3 transform_dir(const float3& pos, const Matrix4x4& mat)
{
	float4 f4;
	f4.w = 0.f;
	*(float3*)&f4 = pos;
	const auto w = f4 * mat;
	return *(float3*)&w;
}

Vec3 transform_pos(const Vec3& pos, const Matrix4x4& mat)
{
	float4 f4;
	f4.w = 1.f;
	*(Vec3*)&f4 = pos;
	const auto w = f4 * mat;
	return *(Vec3*)&w;
}

Vec3 transform_dir(const Vec3& pos, const Matrix4x4& mat)
{
	float4 f4;
	f4.w = 0.f;
	*(Vec3*)&f4 = pos;
	const auto w = f4 * mat;
	return *(Vec3*)&w;
}

NodeTransformation NodeTransformation::operator*(const NodeTransformation& b) const
{
	return to_transformation(to_matrix(*this) * to_matrix(b));
}

NodeTransformation NodeTransformation::transpose() const
{
	return to_transformation(to_matrix(*this).transpose());
}

NodeTransformation NodeTransformation::rotation(const float deg, const float dir[3])
{
	return to_transformation(Matrix4x4::rotate(deg * M_PIf / 180.f, float3{dir[0], dir[1], dir[2]}));
}

NodeTransformation NodeTransformation::identity()
{
	return to_transformation(Matrix4x4::identity());
}

static bool test_mesh_property(const Mesh* mesh, const std::string& query, bool& valid_property)
{
	const auto sep = query.find_first_of(':');
	if (sep == std::string::npos)
	{
		valid_property = false;
		return false;
	}

	valid_property = true;
	switch (sep)
	{
		case 6:
		{
			const auto p = query.substr(0, sep);
			if (p == "shader")
			{
				return mesh->material && std_ext::match(query.substr(sep + 1), mesh->material->shader);
			}
			break;
		}
		case 7:
		{
			const auto p = query.substr(0, sep);
			if (p == "texture")
			{
				if (!mesh->material) return false;
				for (const auto& t : mesh->material->resources)
				{
					if (std_ext::match(query.substr(sep + 1), t.texture)) return true;
				}
			}
			break;
		}
		case 8:
		{
			const auto p = query.substr(0, sep);
			if (p == "material")
			{
				return mesh->material && std_ext::match(query.substr(sep + 1), mesh->material->name);
			}
			break;
		}
		case 16:
		{
			const auto p = query.substr(0, sep);
			if (p == "materialProperty")
			{
				return mesh->material && mesh->material->get_var_or_null(query.substr(sep + 1)) != nullptr;
			}
			if (p == "materialResource")
			{
				return mesh->material && mesh->material->get_resource_or_null(query.substr(sep + 1)) != nullptr;
			}
			break;
		}
		default: break;
	}
	valid_property = false;
	return false;
}

bool Mesh::matches(const std::string& query) const
{
	bool valid_property;
	return std_ext::match(query, name) || test_mesh_property(this, query, valid_property);
}

void Bone::solve(const std::shared_ptr<Node>& root)
{
	node = root->find_node(name);
}

Vec3 skinned_pos(const Vec3& pos, const Vec4& weight, const Vec4& bone_id, const std::vector<Matrix4x4>& bones, const Node* node)
{
	const auto& bone0 = bones[bone_id.x < 0 ? 0 : uint(bone_id.x)];
	const auto& bone1 = bones[bone_id.y < 0 ? 0 : uint(bone_id.y)];
	const auto& bone2 = bones[bone_id.z < 0 ? 0 : uint(bone_id.z)];
	const auto& bone3 = bones[bone_id.w < 0 ? 0 : uint(bone_id.w)];
	const auto& s = *(float3*)&pos;
	auto p = weight.x * transform_pos(s, bone0);
	p += weight.y * transform_pos(s, bone1);
	p += weight.z * transform_pos(s, bone2);
	p += (1.0f - (weight.x + weight.y + weight.z)) * transform_pos(s, bone3);
	return *(Vec3*)&p;
}

/*void UpdateNodes()
{
	if (_bonesNodes == null) return;

	var fix = Matrix.Invert(ParentMatrix * ModelMatrixInverted);
	var bones = OriginalNode.Bones;
	for (var i = 0; i < bones.Length; i++)
	{
		var node = _bonesNodes[i];
		if (node != null)
		{
			_bones[i] = _bonesTransform[i] * node.RelativeToModel * fix;
		}
	}
}*/

void SkinnedMesh::resolve(const Node* node)
{
	if (vertices_orig.empty() && normals_orig.empty())
	{
		vertices_orig = vertices;
		normals_orig = normals;
	}

	std::vector<Matrix4x4> bones_;
	bones_.resize(bones.size());

	for (auto i = 0U; i < bones.size(); i++)
	{
		auto& b = bones[i];
		b.node = node->find_node(b.name);
		if (!b.node)
		{
			std::cerr << "Node for bone is missing: `" << b.name << "` (mesh: `" << name << "`)" << std::endl;
			exit(3);
		}

		bones_[i] = to_matrix(b.node->matrix * b.tranform).transpose();
	}

	for (auto i = 0U; i < vertices.size(); i++)
	{
		vertices[i] = skinned_pos(vertices_orig[i], weights[i], bone_ids[i], bones_, node);
	}
}

Node::Node(const std::string& name, const NodeTransformation& matrix)
	: matrix_local(matrix), matrix_local_orig(matrix)
{
	this->name = name;
}

std::shared_ptr<Mesh> Node::find_mesh(const std::string& name)
{
	for (const auto& c : children)
	{
		auto n = std::dynamic_pointer_cast<Node>(c);
		if (n)
		{
			auto found = n->find_mesh(name);
			if (found)
			{
				return found;
			}
		}
		else
		{
			auto m = std::dynamic_pointer_cast<Mesh>(c);
			if (m && m->matches(name))
			{
				return m;
			}
		}
	}
	return nullptr;
}

std::vector<std::shared_ptr<Mesh>> Node::find_meshes(const std::vector<std::string>& names)
{
	std::vector<std::shared_ptr<Mesh>> result;
	for (const auto& name : names)
	{
		auto mesh = find_mesh(name);
		if (mesh)
		{
			result.push_back(mesh);
		}
	}
	return result;
}

std::vector<std::shared_ptr<Mesh>> Node::find_any_meshes(const std::vector<std::string>& names)
{
	auto result = find_meshes(names);
	for (const auto& n : find_nodes(names))
	{
		result += n->get_meshes();
	}
	return result;
}

std::shared_ptr<Node> Node::find_node(const std::string& name) const
{
	for (const auto& c : children)
	{
		auto n = std::dynamic_pointer_cast<Node>(c);
		if (n)
		{
			auto found = n->name == name ? n : n->find_node(name);
			if (found)
			{
				return found;
			}

			// If Mesh matches, returns its parent
			for (const auto& nc : n->children)
			{
				auto m = std::dynamic_pointer_cast<Mesh>(nc);
				if (m && m->matches(name))
				{
					return n;
				}
			}
		}
	}
	return nullptr;
}

std::vector<std::shared_ptr<Node>> Node::find_nodes(const std::vector<std::string>& names)
{
	std::vector<std::shared_ptr<Node>> result;
	for (const auto& name : names)
	{
		auto node = find_node(name);
		if (node)
		{
			result.push_back(node);
		}
	}
	return result;
}

bool Node::set_active(const std::vector<std::string>& names, const bool value)
{
	auto any = false;
	for (const auto& n : find_nodes(names))
	{
		n->active_local = value;
		any = true;
	}
	for (const auto& n : find_meshes(names))
	{
		n->active_local = value;
		any = true;
	}
	return any;
}

void Node::add_child(const std::shared_ptr<NodeBase>& node)
{
	if (!node) return;
	children.push_back(node);
	node->parent = this;
}

std::vector<std::shared_ptr<Mesh>> Node::get_meshes()
{
	std::vector<std::shared_ptr<Mesh>> result;
	flatten(result);
	return result;
}

void Node::resolve_skinned()
{
	resolve_skinned(this);
}

void Node::flatten(std::vector<std::shared_ptr<Mesh>>& list)
{
	for (const auto& c : children)
	{
		auto m = std::dynamic_pointer_cast<Mesh>(c);
		if (m)
		{
			list.push_back(m);
		}
		else
		{
			std::dynamic_pointer_cast<Node>(c)->flatten(list);
		}
	}
}

void Node::update_matrix()
{
	matrix = parent ? parent->matrix * matrix_local : matrix_local;
	active = parent ? parent->active && active_local : active_local;
	for (const auto& c : children)
	{
		c->update_matrix();
	}
}

void Node::resolve_skinned(const Node* root)
{
	for (const auto& c : children)
	{
		auto m = std::dynamic_pointer_cast<SkinnedMesh>(c);
		if (m)
		{
			m->resolve(root);
		}
		else
		{
			auto n = std::dynamic_pointer_cast<Node>(c);
			if (n)
			{
				n->resolve_skinned(root);
			}
		}
	}
}

NodeTransformation lerp_tranformation(const NodeTransformation& a, const NodeTransformation& b, float mix)
{
	NodeTransformation result{};
	for (auto i = 0U; i < result.size(); i++)
	{
		result[i] = lerp(a[i], b[i], mix);
	}
	return result;
}

void NodeTransition::apply(float progress) const
{
	progress = clamp(progress, 0.f, 1.f);

	const auto frame_prev = uint(floorf((float(frames.size()) - 1.f) * progress));
	const auto frame_next = uint(ceilf((float(frames.size()) - 1.f) * progress));
	if (frame_next == frame_prev)
	{
		node->matrix_local = frames[frame_next];
	}
	else
	{
		const auto mix = clamp((float(frames.size()) - 1.f) * progress - frame_prev, 0.f, 1.f);
		node->matrix_local = lerp_tranformation(frames[frame_prev], frames[frame_next], mix);
	}
}

void Animation::apply(float progress) const
{
	for (const auto& e : entries)
	{
		e.apply(progress);
	}
}

Scene::Scene(const std::shared_ptr<Node>& root) : Scene(root ? std::vector<std::shared_ptr<Node>>{root} : std::vector<std::shared_ptr<Node>>{}) {}

Scene::Scene(const std::vector<std::shared_ptr<Node>>& nodes)
{
	for (const auto& n : nodes)
	{
		if (!n) continue;
		n->update_matrix();
		n->resolve_skinned();
		for (const auto& m : n->get_meshes())
		{
			if (!m->visible || !m->active) continue;
			if (m->receive_shadows) receivers.push_back(m);
			if (m->cast_shadows) blockers.push_back(m);

			const auto& x = to_matrix(m->matrix).transpose();
			for (auto& v : m->vertices)
			{
				const auto w = transform_pos(v, x);
				expand_bbox(bbox_min, bbox_max, &w.x);
			}
		}
	}
}

void HierarchyNode::align(const std::shared_ptr<Node>& root)
{
	auto target = root->name == name ? root : root->find_node(name);
	if (target)
	{
		target->matrix_local = target->matrix_local_orig = matrix_local;
		for (const auto& c : children)
		{
			c->align(target);
		}
	}
	else
	{
		for (const auto& c : children)
		{
			c->align(root);
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
