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

// ReSharper disable once CppUnusedIncludeDirective
#include <utils/dbg_output.h>

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
		Vec3 signature_point;
		float extra_samples_offset{};
		std::vector<Vec3> vertices;
		std::vector<Vec3> normals;
		std::vector<Triangle> triangles;
		std::shared_ptr<Material> material;
		float lod_in{};
		float lod_out{};
		bool cast_shadows = true;
		bool receive_shadows = true;
		bool visible = true;

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

		Bone()
		{ }

		Bone(std::string name, const NodeTransformation& matrix = NodeTransformation::identity())
			: name(std::move(name)), tranform(matrix)
		{ }

		void solve(const std::shared_ptr<Node>& root);
	};

	struct SkinnedMesh : Mesh
	{
		std::vector<Bone> bones;
		std::vector<Vec4> weights;
		std::vector<Vec4> bone_ids;
		std::vector<Vec3> vertices_orig;
		std::vector<Vec3> normals_orig;
		void resolve(const Node* node);
	};

	struct Filter
	{
		std::vector<std::string> items;
		explicit Filter(std::vector<std::string> items) : items(std::move(items)) { }
	};

	struct Node : NodeBase
	{
		Node(const std::string& name, const NodeTransformation& matrix = NodeTransformation::identity());
		std::shared_ptr<Mesh> find_mesh(const std::string& name);
		std::vector<std::shared_ptr<Mesh>> find_meshes(const Filter& names) const;
		std::vector<std::shared_ptr<Mesh>> find_any_meshes(const Filter& names) const;
		std::vector<std::shared_ptr<Node>> find_nodes(const Filter& names) const;
		bool set_active(const Filter& names, const bool value) const;
		void add_child(const std::shared_ptr<NodeBase>& node);

		std::shared_ptr<Node> find_node(const Filter& filter) const
		{
			const auto nodes = find_nodes(filter);
			return nodes.empty() ? nullptr : nodes[0];
		}

		NodeTransformation matrix_local;
		NodeTransformation matrix_local_orig;
		std::vector<std::shared_ptr<NodeBase>> children;

		std::vector<std::shared_ptr<Mesh>> get_meshes();
		void flatten(std::vector<std::shared_ptr<Mesh>>& list);
		void update_matrix() override;
		void resolve_skinned();

	private:
		void resolve_skinned(const Node* root);
	};

	struct NodeTransition
	{
		std::string name;
		std::shared_ptr<Node> node{};
		std::vector<NodeTransformation> frames;
		bool apply(float progress) const;
	};

	struct Animation
	{
		std::vector<bake::NodeTransition> entries;
		void init(const std::shared_ptr<Node>& root);
		bool apply(const std::shared_ptr<Node>& root, float progress);
		static bool apply_all(const std::shared_ptr<Node>& root, const std::vector<std::shared_ptr<bake::Animation>>& animations, float progress);
	private:
		Node* last_root_{};
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

		HierarchyNode(std::string name, const NodeTransformation& matrix = NodeTransformation::identity())
			: name(std::move(name)), matrix_local(matrix)
		{ }

		void align(const std::shared_ptr<bake::Node>& root) const;
	};

	struct AILanePoint
	{
		Vec3 point;
		float length;
		uint32_t index;
		float side_left;
		float side_right;
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
