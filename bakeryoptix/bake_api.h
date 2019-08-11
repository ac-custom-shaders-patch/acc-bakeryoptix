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

#pragma once
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include <memory>
#include <utils/std_ext.h>

namespace bake
{
	struct SampleInfo
	{
		unsigned tri_idx;
		float bary[3];
		float dA;
	};

	struct Mesh;

	struct NodeTransformation : std::array<float, 16>
	{
		NodeTransformation() = default;
		NodeTransformation operator*(const NodeTransformation& b) const;
		NodeTransformation transpose() const;
		static NodeTransformation rotation(float deg, const float dir[3]);
		static NodeTransformation identity();
	};

	struct NodeBase
	{
		std::string name;
		NodeTransformation matrix = NodeTransformation::identity();
		NodeBase* parent{};
		bool active = true;
		bool active_local = true;
		virtual ~NodeBase() = default;
		virtual void update_matrix() = 0;
	};


	struct MaterialProperty
	{
		std::string name;
		float v1{};
		float v2[2]{};
		float v3[3]{};
		float v4[4]{};
	};

	struct MaterialResource
	{
		std::string name;
		uint32_t slot{};
		std::string texture;
	};

	enum class MaterialBlendMode : uint8_t
	{
		opaque = 0,
		blend = 1,
		coverage = 2
	};

	struct Material
	{
		std::string name;
		std::string shader;
		MaterialBlendMode blend{};
		bool alpha_tested{};
		uint32_t depth_mode{};
		std::vector<MaterialProperty> vars;
		std::vector<MaterialResource> resources;

		const MaterialProperty* get_var_or_null(const std::string& cs)
		{
			for (const auto& v : vars)
			{
				if (v.name == name) return &v;
			}
			return nullptr;
		}

		const MaterialResource* get_resource_or_null(const std::string& cs)
		{
			for (const auto& v : resources)
			{
				if (v.name == name) return &v;
			}
			return nullptr;
		}
	};

	struct Triangle
	{
		uint32_t a, b, c;
	};

	struct Vec3
	{
		float x, y, z;
	};

	struct Vec4
	{
		float x, y, z, w;
	};

	struct Mesh : NodeBase
	{
		std::vector<Vec3> vertices;
		std::vector<Vec3> normals;
		std::vector<Triangle> triangles;
		bool cast_shadows;
		bool receive_shadows;
		bool visible;
		std::shared_ptr<Material> material;

		void update_matrix() override
		{
			matrix = parent->matrix;
			active = parent->active && active_local;
		}

		bool matches(const std::string& query) const;
	};

	struct Node;

	struct Bone
	{
		std::string name;
		NodeTransformation tranform = NodeTransformation::identity();
		std::shared_ptr<Node> node{};

		Bone(std::string name, const NodeTransformation& matrix = NodeTransformation::identity())
			: name(std::move(name)), tranform(matrix)
		{ }

		void solve(const std::shared_ptr<Node>& root);
	};

	struct SkinnedMesh : Mesh
	{
		std::vector<Bone> bones;
		std::vector<Vec4> bone_ids;
		std::vector<Vec4> weights;
	};

	struct Node : NodeBase
	{
		Node(const std::string& name, const NodeTransformation& matrix = NodeTransformation::identity())
			: matrix_local(matrix), matrix_local_orig(matrix)
		{
			this->name = name;
		}

		std::shared_ptr<Mesh> find_mesh(const std::string& name)
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

		std::vector<std::shared_ptr<Mesh>> find_meshes(const std::vector<std::string>& names)
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

		std::shared_ptr<Node> find_node(const std::string& name)
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

		std::vector<std::shared_ptr<Node>> find_nodes(const std::vector<std::string>& names)
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

		bool set_active(const std::vector<std::string>& names, const bool value)
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

		void add_child(const std::shared_ptr<NodeBase>& node)
		{
			if (!node) return;
			children.push_back(node);
			node->parent = this;
		}

		NodeTransformation matrix_local;
		NodeTransformation matrix_local_orig;
		std::vector<std::shared_ptr<NodeBase>> children;

		std::vector<std::shared_ptr<Mesh>> get_meshes()
		{
			std::vector<std::shared_ptr<Mesh>> result;
			flatten(result);
			return result;
		}

		void flatten(std::vector<std::shared_ptr<Mesh>>& list)
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

		void update_matrix() override;
	};

	struct NodeTransition
	{
		std::shared_ptr<Node> node;
		std::vector<NodeTransformation> frames;
		void apply(float progress) const;
	};

	struct Animation
	{
		std::vector<bake::NodeTransition> entries;
		void apply(float progress) const;
	};

	struct Scene
	{
		Scene(const std::shared_ptr<Node>& root);
		Scene(const std::vector<std::shared_ptr<Node>>& nodes);

		std::vector<std::shared_ptr<Mesh>> receivers;
		std::vector<std::shared_ptr<Mesh>> blockers;

		float bbox_min[3]{FLT_MAX, FLT_MAX, FLT_MAX};
		float bbox_max[3]{-FLT_MAX, -FLT_MAX, -FLT_MAX};
	};

	struct HierarchyNode
	{
		std::string name;
		NodeTransformation matrix_local;
		std::vector<std::shared_ptr<HierarchyNode>> children;

		HierarchyNode(const std::string& name, const NodeTransformation& matrix = NodeTransformation::identity())
			: name(name), matrix_local(matrix)
		{ }

		void align(const std::shared_ptr<bake::Node>& root);
	};

	struct AOSamples
	{
		size_t num_samples;
		float* sample_positions;
		float* sample_normals;
		float* sample_face_normals;
		SampleInfo* sample_infos;
	};

	enum VertexFilterMode
	{
		VERTEX_FILTER_AREA_BASED = 0,
		VERTEX_FILTER_LEAST_SQUARES,
		VERTEX_FILTER_INVALID
	};

	void map_ao_to_vertices(
		const Scene& scene,
		const size_t* num_samples_per_instance,
		const AOSamples& ao_samples,
		const float* ao_values,
		VertexFilterMode mode,
		float regularization_weight,
		float** vertex_ao);
}
