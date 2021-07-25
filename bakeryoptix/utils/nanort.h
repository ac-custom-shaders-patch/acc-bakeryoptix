//
// NanoRT, single header only modern ray tracing kernel.
//

/*
The MIT License (MIT)

Copyright (c) 2015 - 2019 Light Transport Entertainment, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

// Some constants
#define kNANORT_MAX_STACK_DEPTH (512)
#define kNANORT_MIN_PRIMITIVES_FOR_PARALLEL_BUILD (1024 * 8)
#define kNANORT_SHALLOW_DEPTH (4)  // will create 2**N subtrees

#define _assert(x) { if(!(x)){ throw std::exception("ASSERT FAILED: "#x); }}

namespace nanort
{
	// ----------------------------------------------------------------------------
	// Small vector class useful for multi-threaded environment.
	//
	// stack_container.h
	//
	// Copyright (c) 2006-2008 The Chromium Authors. All rights reserved.
	// Use of this source code is governed by a BSD-style license that can be
	// found in the LICENSE file.

	// This allocator can be used with STL containers to provide a stack buffer
	// from which to allocate memory and overflows onto the heap. This stack buffer
	// would be allocated on the stack and allows us to avoid heap operations in
	// some situations.
	//
	// STL likes to make copies of allocators, so the allocator itself can't hold
	// the data. Instead, we make the creator responsible for creating a
	// StackAllocator::Source which contains the data. Copying the allocator
	// merely copies the pointer to this shared source, so all allocators created
	// based on our allocator will share the same stack buffer.
	//
	// This stack buffer implementation is very simple. The first allocation that
	// fits in the stack buffer will use the stack buffer. Any subsequent
	// allocations will not use the stack buffer, even if there is unused room.
	// This makes it appropriate for array-like containers, but the caller should
	// be sure to reserve() in the container up to the stack buffer size. Otherwise
	// the container will allocate a small array which will "use up" the stack
	// buffer.
	template <typename T, size_t stack_capacity>
	class StackAllocator : public std::allocator<T>
	{
	public:
		typedef typename std::allocator<T>::pointer pointer;
		typedef typename std::allocator<T>::size_type size_type;

		// Backing store for the allocator. The container owner is responsible for
		// maintaining this for as long as any containers using this allocator are
		// live.
		struct Source
		{
			Source() : used_stack_buffer_(false) {}

			// Casts the buffer in its right type.
			T* stack_buffer() { return reinterpret_cast<T*>(stack_buffer_); }

			const T* stack_buffer() const
			{
				return reinterpret_cast<const T*>(stack_buffer_);
			}

			//
			// IMPORTANT: Take care to ensure that stack_buffer_ is aligned
			// since it is used to mimic an array of T.
			// Be careful while declaring any unaligned types (like bool)
			// before stack_buffer_.
			//

			// The buffer itself. It is not of type T because we don't want the
			// constructors and destructors to be automatically called. Define a POD
			// buffer of the right size instead.
			char stack_buffer_[sizeof(T[stack_capacity])];

			// Set when the stack buffer is used for an allocation. We do not track
			// how much of the buffer is used, only that somebody is using it.
			bool used_stack_buffer_;
		};

		// Used by containers when they want to refer to an allocator of type U.
		template <typename U>
		struct rebind
		{
			typedef StackAllocator<U, stack_capacity> other;
		};

		// For the straight up copy c-tor, we can share storage.
		StackAllocator(const StackAllocator<T, stack_capacity>& rhs)
			: source_(rhs.source_) {}

		// ISO C++ requires the following constructor to be defined,
		// and std::vector in VC++2008SP1 Release fails with an error
		// in the class _Container_base_aux_alloc_real (from <xutility>)
		// if the constructor does not exist.
		// For this constructor, we cannot share storage; there's
		// no guarantee that the Source buffer of Ts is large enough
		// for Us.
		// TODO(Google): If we were fancy pants, perhaps we could share storage
		// iff sizeof(T) == sizeof(U).
		template <typename U, size_t other_capacity>
		StackAllocator(const StackAllocator<U, other_capacity>& other)
			: source_(NULL)
		{
			(void)other;
		}

		explicit StackAllocator(Source* source) : source_(source) {}

		// Actually do the allocation. Use the stack buffer if nobody has used it yet
		// and the size requested fits. Otherwise, fall through to the standard
		// allocator.
		pointer allocate(size_type n, void* hint = 0)
		{
			if (source_ != NULL && !source_->used_stack_buffer_ &&
				n <= stack_capacity)
			{
				source_->used_stack_buffer_ = true;
				return source_->stack_buffer();
			}
			else
			{
				return std::allocator<T>::allocate(n, hint);
			}
		}

		// Free: when trying to free the stack buffer, just mark it as free. For
		// non-stack-buffer pointers, just fall though to the standard allocator.
		void deallocate(pointer p, size_type n)
		{
			if (source_ != NULL && p == source_->stack_buffer()) source_->used_stack_buffer_ = false;
			else std::allocator<T>::deallocate(p, n);
		}

	private:
		Source* source_;
	};

	// A wrapper around STL containers that maintains a stack-sized buffer that the
	// initial capacity of the vector is based on. Growing the container beyond the
	// stack capacity will transparently overflow onto the heap. The container must
	// support reserve().
	//
	// WATCH OUT: the ContainerType MUST use the proper StackAllocator for this
	// type. This object is really intended to be used only internally. You'll want
	// to use the wrappers below for different types.
	template <typename TContainerType, int stack_capacity>
	class StackContainer
	{
	public:
		typedef TContainerType ContainerType;
		typedef typename ContainerType::value_type ContainedType;
		typedef StackAllocator<ContainedType, stack_capacity> Allocator;

		// Allocator must be constructed before the container!
		StackContainer()
			: allocator_(&stack_data_), container_(allocator_)
		{
			// Make the container use the stack allocation by reserving our buffer size
			// before doing anything else.
			container_.reserve(stack_capacity);
		}

		// Getters for the actual container.
		//
		// Danger: any copies of this made using the copy constructor must have
		// shorter lifetimes than the source. The copy will share the same allocator
		// and therefore the same stack buffer as the original. Use std::copy to
		// copy into a "real" container for longer-lived objects.
		ContainerType& container() { return container_; }
		const ContainerType& container() const { return container_; }

		// Support operator-> to get to the container. This allows nicer syntax like:
		//   StackContainer<...> foo;
		//   std::sort(foo->begin(), foo->end());
		ContainerType* operator->() { return &container_; }
		const ContainerType* operator->() const { return &container_; }

	protected:
		typename Allocator::Source stack_data_;
		unsigned char pad_[7];
		Allocator allocator_;
		ContainerType container_;

		// DISALLOW_EVIL_CONSTRUCTORS(StackContainer);
		StackContainer(const StackContainer&);
		void operator=(const StackContainer&);
	};

	// StackVector
	//
	// Example:
	//   StackVector<int, 16> foo;
	//   foo->push_back(22);  // we have overloaded operator->
	//   foo[0] = 10;         // as well as operator[]
	template <typename T, size_t stack_capacity>
	class StackVector
		: public StackContainer<std::vector<T, StackAllocator<T, stack_capacity>>,
			stack_capacity>
	{
	public:
		StackVector()
			: StackContainer<std::vector<T, StackAllocator<T, stack_capacity>>,
				stack_capacity>() {}

		// We need to put this in STL containers sometimes, which requires a copy
		// constructor. We can't call the regular copy constructor because that will
		// take the stack buffer from the original. Here, we create an empty object
		// and make a stack buffer of its own.
		StackVector(const StackVector<T, stack_capacity>& other)
			: StackContainer<std::vector<T, StackAllocator<T, stack_capacity>>,
				stack_capacity>()
		{
			this->container().assign(other->begin(), other->end());
		}

		StackVector<T, stack_capacity>& operator=(
			const StackVector<T, stack_capacity>& other)
		{
			this->container().assign(other->begin(), other->end());
			return *this;
		}

		// Vectors are commonly indexed, which isn't very convenient even with
		// operator-> (using "->at()" does exception stuff we don't want).
		T& operator[](size_t i) { return this->container().operator[](i); }

		const T& operator[](size_t i) const
		{
			return this->container().operator[](i);
		}
	};

	// ----------------------------------------------------------------------------

	template <typename T = float>
	class real3
	{
	public:
		real3() {}

		real3(T x)
		{
			v[0] = x;
			v[1] = x;
			v[2] = x;
		}

		real3(T xx, T yy, T zz)
		{
			v[0] = xx;
			v[1] = yy;
			v[2] = zz;
		}

		explicit real3(const T* p)
		{
			v[0] = p[0];
			v[1] = p[1];
			v[2] = p[2];
		}

		T x() const { return v[0]; }
		T y() const { return v[1]; }
		T z() const { return v[2]; }

		real3 operator*(T f) const { return real3(x() * f, y() * f, z() * f); }

		real3 operator-(const real3& f2) const
		{
			return real3(x() - f2.x(), y() - f2.y(), z() - f2.z());
		}

		real3 operator*(const real3& f2) const
		{
			return real3(x() * f2.x(), y() * f2.y(), z() * f2.z());
		}

		real3 operator+(const real3& f2) const
		{
			return real3(x() + f2.x(), y() + f2.y(), z() + f2.z());
		}

		real3& operator+=(const real3& f2)
		{
			v[0] += f2.x();
			v[1] += f2.y();
			v[2] += f2.z();
			return (*this);
		}

		real3 operator/(const real3& f2) const
		{
			return real3(x() / f2.x(), y() / f2.y(), z() / f2.z());
		}

		real3 operator-() const { return real3(-x(), -y(), -z()); }
		T operator[](int i) const { return v[i]; }
		T& operator[](int i) { return v[i]; }

		T v[3];
		// T pad;  // for alignment (when T = float)
	};

	template <typename T>
	real3<T> operator*(T f, const real3<T>& v)
	{
		return real3<T>(v.x() * f, v.y() * f, v.z() * f);
	}

	template <typename T>
	real3<T> vneg(const real3<T>& rhs)
	{
		return real3<T>(-rhs.x(), -rhs.y(), -rhs.z());
	}

	template <typename T>
	T vlength(const real3<T>& rhs)
	{
		return std::sqrt(rhs.x() * rhs.x() + rhs.y() * rhs.y() + rhs.z() * rhs.z());
	}

	template <typename T>
	real3<T> vnormalize(const real3<T>& rhs)
	{
		real3<T> v = rhs;
		T len = vlength(rhs);
		if (std::fabs(len) > std::numeric_limits<T>::epsilon())
		{
			T inv_len = T(1.0) / len;
			v.v[0] *= inv_len;
			v.v[1] *= inv_len;
			v.v[2] *= inv_len;
		}
		return v;
	}

	template <typename T>
	real3<T> vcross(const real3<T> a, const real3<T> b)
	{
		real3<T> c;
		c[0] = a[1] * b[2] - a[2] * b[1];
		c[1] = a[2] * b[0] - a[0] * b[2];
		c[2] = a[0] * b[1] - a[1] * b[0];
		return c;
	}

	template <typename T>
	T vdot(const real3<T> a, const real3<T> b)
	{
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	}

	template <typename T>
	real3<T> vsafe_inverse(const real3<T> v)
	{
		real3<T> r;
		if (std::fabs(v[0]) < std::numeric_limits<T>::epsilon())
		{
			r[0] = std::numeric_limits<T>::infinity() *
				std::copysign(T(1), v[0]);
		}
		else
		{
			r[0] = T(1.0) / v[0];
		}

		if (std::fabs(v[1]) < std::numeric_limits<T>::epsilon())
		{
			r[1] = std::numeric_limits<T>::infinity() *
				std::copysign(T(1), v[1]);
		}
		else
		{
			r[1] = T(1.0) / v[1];
		}

		if (std::fabs(v[2]) < std::numeric_limits<T>::epsilon())
		{
			r[2] = std::numeric_limits<T>::infinity() *
				std::copysign(T(1), v[2]);
		}
		else
		{
			r[2] = T(1.0) / v[2];
		}
		return r;
	}

	template <typename real>
	const real* get_vertex_addr(const real* p, const size_t idx, const size_t stride_bytes)
	{
		return reinterpret_cast<const real*>(reinterpret_cast<const unsigned char*>(p) + idx * stride_bytes);
	}

	template <typename T = float>
	struct Ray
	{
		T org[3]; // must set
		T dir[3]; // must set
		T min_t;  // minimum ray hit distance.
		T max_t;  // maximum ray hit distance.

		Ray() = default;
		Ray(const T* org, const T* dir, float length)
		{
			memcpy(this->org, org, sizeof(T) * 3);
			memcpy(this->dir, dir, sizeof(T) * 3);
			min_t = 0.f;
			max_t = length;
		}
	};

	template <typename T = float>
	class BVHNode
	{
	public:
		BVHNode() {}

		BVHNode(const BVHNode& rhs)
		{
			bmin[0] = rhs.bmin[0];
			bmin[1] = rhs.bmin[1];
			bmin[2] = rhs.bmin[2];
			flag = rhs.flag;

			bmax[0] = rhs.bmax[0];
			bmax[1] = rhs.bmax[1];
			bmax[2] = rhs.bmax[2];
			axis = rhs.axis;

			data[0] = rhs.data[0];
			data[1] = rhs.data[1];
		}

		BVHNode& operator=(const BVHNode& rhs)
		{
			bmin[0] = rhs.bmin[0];
			bmin[1] = rhs.bmin[1];
			bmin[2] = rhs.bmin[2];
			flag = rhs.flag;

			bmax[0] = rhs.bmax[0];
			bmax[1] = rhs.bmax[1];
			bmax[2] = rhs.bmax[2];
			axis = rhs.axis;

			data[0] = rhs.data[0];
			data[1] = rhs.data[1];

			return (*this);
		}

		~BVHNode() {}

		T bmin[3];
		T bmax[3];

		int flag; // 1 = leaf node, 0 = branch node
		int axis;

		// leaf
		//   data[0] = npoints
		//   data[1] = index
		//
		// branch
		//   data[0] = child[0]
		//   data[1] = child[1]
		uint32_t data[2];
	};

	template <class H>
	class IntersectComparator
	{
	public:
		bool operator()(const H& a, const H& b) const { return a.t < b.t; }
	};

	/// BVH build option.
	template <typename T = float>
	struct BVHBuildOptions
	{
		T cost_t_aabb;
		uint32_t min_leaf_primitives;
		uint32_t max_tree_depth;
		uint32_t bin_size;
		uint32_t shallow_depth;

		// Cache bounding box computation.
		// Requires more memory, but BVHbuild can be faster.
		bool cache_bbox;
		unsigned char pad[3];

		// Set default value: Taabb = 0.2
		BVHBuildOptions()
			: cost_t_aabb(T(0.2)),
			min_leaf_primitives(4),
			max_tree_depth(256),
			bin_size(64),
			shallow_depth(kNANORT_SHALLOW_DEPTH),
			cache_bbox(false) {}
	};

	///
	/// @brief Bounding box.
	///
	template <typename T>
	class BBox
	{
	public:
		real3<T> bmin;
		real3<T> bmax;

		BBox()
		{
			bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<T>::max();
			bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<T>::max();
		}
	};

	///
	/// @brief Hit class for traversing nodes.
	///
	/// Stores hit information of node traversal.
	/// Node traversal is used for two-level ray tracing(efficient ray traversal of a scene hierarchy)
	///
	template <typename T>
	class NodeHit
	{
	public:
		NodeHit()
			: t_min(std::numeric_limits<T>::max()),
			t_max(-std::numeric_limits<T>::max()),
			node_id(uint32_t(-1)) {}

		NodeHit(const NodeHit<T>& rhs)
		{
			t_min = rhs.t_min;
			t_max = rhs.t_max;
			node_id = rhs.node_id;
		}

		NodeHit& operator=(const NodeHit<T>& rhs)
		{
			t_min = rhs.t_min;
			t_max = rhs.t_max;
			node_id = rhs.node_id;

			return (*this);
		}

		~NodeHit() {}

		T t_min;
		T t_max;
		uint32_t node_id;
	};

	///
	/// @brief Comparator object for NodeHit.
	///
	/// Comparator object for finding nearest hit point in node traversal.
	///
	template <typename T>
	class NodeHitComparator
	{
	public:
		bool operator()(const NodeHit<T>& a, const NodeHit<T>& b)
		{
			return a.t_min < b.t_min;
		}
	};

	///
	/// @brief Bounding Volume Hierarchy acceleration.
	///
	/// BVHAccel is central part of ray tracing(ray traversal).
	/// BVHAccel takes an input geometry(primitive) information and build a data structure
	/// for efficient ray tracing(`O(log2 N)` in theory, where N is the number of primitive in the scene).
	///
	/// @tparam T real value type(float or double).
	///
	template <typename T>
	class BVHAccel
	{
	public:
		BVHAccel() : pad0_(0) { (void)pad0_; }

		template <class Prim, class Pred>
		bool Build(uint32_t num_primitives, const Prim& p, const Pred& pred, const BVHBuildOptions<T>& options = BVHBuildOptions<T>());

		template <bool TraverseDown, class I, class H>
		bool Traverse(const Ray<T>& ray, I& intersector, H* isect) const;

		void TraverseDownSimple(const T* vertices, const uint32_t* faces, const float* pos, float& hit_distance, uint32_t& hit_prim_idx) const;

	private:
		/// Builds BVH tree recursively.
		template <class P, class Pred>
		uint32_t BuildTree(std::vector<BVHNode<T>>* out_nodes, uint32_t left_idx, uint32_t right_idx, uint32_t depth, const P& p, const Pred& pred);

		template <bool TraverseDown, class I>
		bool TestLeafNode(const BVHNode<T>& node, I& intersector) const;

		std::vector<BVHNode<T>> nodes_;
		std::vector<uint32_t> indices_;
		std::vector<BBox<T>> bboxes_;
		BVHBuildOptions<T> options_;
		uint32_t pad0_;
	};

	// Predefined SAH predicator for triangle.
	template <typename T = float>
	class TriangleSAHPred
	{
	public:
		TriangleSAHPred(const T* vertices, const uint32_t* faces): axis_(0), pos_(T(0.0)), vertices_(vertices), faces_(faces) {}
		TriangleSAHPred(const TriangleSAHPred<T>& rhs) : axis_(rhs.axis_), pos_(rhs.pos_), vertices_(rhs.vertices_), faces_(rhs.faces_) {}

		TriangleSAHPred<T>& operator=(const TriangleSAHPred<T>& rhs)
		{
			axis_ = rhs.axis_;
			pos_ = rhs.pos_;
			vertices_ = rhs.vertices_;
			faces_ = rhs.faces_;
			return (*this);
		}

		void Set(int axis, T pos) const
		{
			axis_ = axis;
			pos_ = pos;
		}

		bool operator()(uint32_t i) const
		{
			int axis = axis_;
			T pos = pos_;

			uint32_t i0 = faces_[3 * i + 0];
			uint32_t i1 = faces_[3 * i + 1];
			uint32_t i2 = faces_[3 * i + 2];

			real3<T> p0(get_vertex_addr<T>(vertices_, i0, sizeof(T) * 3));
			real3<T> p1(get_vertex_addr<T>(vertices_, i1, sizeof(T) * 3));
			real3<T> p2(get_vertex_addr<T>(vertices_, i2, sizeof(T) * 3));

			T center = p0[axis] + p1[axis] + p2[axis];

			return (center < pos * T(3.0));
		}

	private:
		mutable int axis_;
		mutable T pos_;
		const T* vertices_;
		const uint32_t* faces_;
	};

	// Predefined SAH predicator for triangle.
	template <typename T = float>
	class TriangleNoIndicesSAHPred
	{
	public:
		TriangleNoIndicesSAHPred(const T* vertices): axis_(0), pos_(T(0.0)), vertices_(vertices) {}
		TriangleNoIndicesSAHPred(const TriangleNoIndicesSAHPred<T>& rhs) : axis_(rhs.axis_), pos_(rhs.pos_), vertices_(rhs.vertices_) {}

		TriangleNoIndicesSAHPred<T>& operator=(const TriangleNoIndicesSAHPred<T>& rhs)
		{
			axis_ = rhs.axis_;
			pos_ = rhs.pos_;
			vertices_ = rhs.vertices_;
			return (*this);
		}

		void Set(int axis, T pos) const
		{
			axis_ = axis;
			pos_ = pos;
		}

		bool operator()(uint32_t i) const
		{
			int axis = axis_;
			T pos = pos_;

			real3<T> p0(get_vertex_addr<T>(vertices_, 3 * i, sizeof(T) * 3));
			real3<T> p1(get_vertex_addr<T>(vertices_, 3 * i + 1, sizeof(T) * 3));
			real3<T> p2(get_vertex_addr<T>(vertices_, 3 * i + 2, sizeof(T) * 3));

			T center = p0[axis] + p1[axis] + p2[axis];

			return (center < pos * T(3.0));
		}

	private:
		mutable int axis_;
		mutable T pos_;
		const T* vertices_;
	};

	// Predefined Triangle mesh geometry.
	template <typename T = float>
	class TriangleMesh
	{
	public:
		TriangleMesh(const T* vertices, const uint32_t* faces) : vertices_(vertices), faces_(faces) {}

		/// Compute bounding box for `prim_index`th triangle.
		/// This function is called for each primitive in BVH build.
		void BoundingBox(real3<T>* bmin, real3<T>* bmax, uint32_t prim_index) const
		{
			unsigned vertex = faces_[3 * prim_index + 0];
			(*bmin)[0] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[0];
			(*bmin)[1] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[1];
			(*bmin)[2] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[2];
			(*bmax)[0] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[0];
			(*bmax)[1] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[1];
			(*bmax)[2] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[2];

			// remaining two vertices of the primitive
			for (uint32_t i = 1; i < 3; i++)
			{
				// xyz
				for (int k = 0; k < 3; k++)
				{
					T coord = get_vertex_addr<T>(vertices_, faces_[3 * prim_index + i], sizeof(T) * 3)[k];
					(*bmin)[k] = std::min((*bmin)[k], coord);
					(*bmax)[k] = std::max((*bmax)[k], coord);
				}
			}
		}

		const T* vertices_;
		const uint32_t* faces_;
	};

	// Predefined Triangle mesh geometry.
	template <typename T = float>
	class TriangleNoIndicesMesh
	{
	public:
		TriangleNoIndicesMesh(const T* vertices) : vertices_(vertices) {}

		/// Compute bounding box for `prim_index`th triangle.
		/// This function is called for each primitive in BVH build.
		void BoundingBox(real3<T>* bmin, real3<T>* bmax, uint32_t prim_index) const
		{
			unsigned vertex = 3 * prim_index + 0;
			(*bmin)[0] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[0];
			(*bmin)[1] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[1];
			(*bmin)[2] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[2];
			(*bmax)[0] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[0];
			(*bmax)[1] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[1];
			(*bmax)[2] = get_vertex_addr(vertices_, vertex, sizeof(T) * 3)[2];

			// remaining two vertices of the primitive
			for (uint32_t i = 1; i < 3; i++)
			{
				// xyz
				for (int k = 0; k < 3; k++)
				{
					T coord = get_vertex_addr<T>(vertices_, 3 * prim_index + i, sizeof(T) * 3)[k];
					(*bmin)[k] = std::min((*bmin)[k], coord);
					(*bmax)[k] = std::max((*bmax)[k], coord);
				}
			}
		}

		const T* vertices_;
	};

	///
	/// Stores intersection point information for triangle geometry.
	///
	template <typename T = float>
	class TriangleIntersection
	{
	public:
		// Required member variables.
		T t;
		uint32_t prim_id;
	};

	///
	/// Intersector is a template class which implements intersection method and stores
	/// intesection point information(`H`)
	///
	/// @tparam T Precision(float or double)
	/// @tparam H Intersection point information struct
	///
	template <typename T = float, class H = TriangleIntersection<T>>
	class TriangleIntersector
	{
	public:
		TriangleIntersector(const T* vertices, const uint32_t* faces) : vertices_(vertices), faces_(faces) {}

		// For Watertight Ray/Triangle Intersection.
		typedef struct
		{
			T Sx;
			T Sy;
			T Sz;
			int kx;
			int ky;
			int kz;
		} RayCoeff;

		/// Do ray intersection stuff for `prim_index` th primitive and return hit
		/// distance `t`, barycentric coordinate `u` and `v`.
		/// Returns true if there's intersection.
		bool Intersect(T* t_inout, const uint32_t prim_index) const
		{
			const uint32_t f0 = faces_[3 * prim_index + 0];
			const uint32_t f1 = faces_[3 * prim_index + 1];
			const uint32_t f2 = faces_[3 * prim_index + 2];

			const auto& p0 = *(real3<T>*)(get_vertex_addr(vertices_, f0, sizeof(T) * 3));
			const auto& p1 = *(real3<T>*)(get_vertex_addr(vertices_, f1, sizeof(T) * 3));
			const auto& p2 = *(real3<T>*)(get_vertex_addr(vertices_, f2, sizeof(T) * 3));

			const real3<T> A = p0 - ray_org_;
			const real3<T> B = p1 - ray_org_;
			const real3<T> C = p2 - ray_org_;

			const auto kx = ray_coeff_.kx;
			const auto ky = ray_coeff_.ky;
			const auto kz = ray_coeff_.kz;
			const auto Sx = ray_coeff_.Sx;
			const auto Sy = ray_coeff_.Sy;
			const auto Sz = ray_coeff_.Sz;
			const T Ax = A[kx] - Sx * A[kz];
			const T Ay = A[ky] - Sy * A[kz];
			const T Bx = B[kx] - Sx * B[kz];
			const T By = B[ky] - Sy * B[kz];
			const T Cx = C[kx] - Sx * C[kz];
			const T Cy = C[ky] - Sy * C[kz];
			T U = Cx * By - Cy * Bx;
			T V = Ax * Cy - Ay * Cx;
			T W = Bx * Ay - By * Ax;

			// Fall back to test against edges using double precision.
			if (U == T(0.0) || V == T(0.0) || W == T(0.0))
			{
				U = T(double(Cx) * double(By) - double(Cy) * double(Bx));
				V = T(double(Ax) * double(Cy) - double(Ay) * double(Cx));
				W = T(double(Bx) * double(Ay) - double(By) * double(Ax));
			}

			T det = U + V + W;
			if (U < T(0.0) || V < T(0.0) || W < T(0.0) || det == T(0.0))
			{
				return false;
			}

			const T Az = Sz * A[kz];
			const T Bz = Sz * B[kz];
			const T Cz = Sz * C[kz];
			const T D = U * Az + V * Bz + W * Cz;
			const T tt = D / det;
			if (tt > (*t_inout) || tt < t_min_)
			{
				return false;
			}

			*t_inout = tt;
			return true;
		}

		/// Returns the nearest hit distance.
		T GetT() const { return t_; }

		/// Update is called when initializing intersection and nearest hit is found.
		void Update(T t, uint32_t prim_idx)
		{
			t_ = t;
			prim_id_ = prim_idx;
		}

		/// Prepare BVH traversal (e.g. compute inverse ray direction)
		/// This function is called only once in BVH traversal.
		void PrepareTraversal(const Ray<T>& ray)
		{
			ray_org_[0] = ray.org[0];
			ray_org_[1] = ray.org[1];
			ray_org_[2] = ray.org[2];

			// Calculate dimension where the ray direction is maximal.
			ray_coeff_.kz = 0;
			T absDir = std::fabs(ray.dir[0]);
			if (absDir < std::fabs(ray.dir[1]))
			{
				ray_coeff_.kz = 1;
				absDir = std::fabs(ray.dir[1]);
			}
			if (absDir < std::fabs(ray.dir[2]))
			{
				ray_coeff_.kz = 2;
				absDir = std::fabs(ray.dir[2]);
			}

			ray_coeff_.kx = ray_coeff_.kz + 1;
			if (ray_coeff_.kx == 3) ray_coeff_.kx = 0;
			ray_coeff_.ky = ray_coeff_.kx + 1;
			if (ray_coeff_.ky == 3) ray_coeff_.ky = 0;

			// Swap kx and ky dimension to preserve winding direction of triangles.
			if (ray.dir[ray_coeff_.kz] < T(0.0)) std::swap(ray_coeff_.kx, ray_coeff_.ky);

			// Calculate shear constants.
			ray_coeff_.Sx = ray.dir[ray_coeff_.kx] / ray.dir[ray_coeff_.kz];
			ray_coeff_.Sy = ray.dir[ray_coeff_.ky] / ray.dir[ray_coeff_.kz];
			ray_coeff_.Sz = T(1.0) / ray.dir[ray_coeff_.kz];

			t_min_ = ray.min_t;
		}

		/// Post BVH traversal stuff.
		/// Fill `isect` if there is a hit.
		void PostTraversal(bool hit, H* isect) const
		{
			if (hit && isect)
			{
				isect->t = t_;
				isect->prim_id = prim_id_;
			}
		}

	private:
		const T* vertices_;
		const uint32_t* faces_;

		real3<T> ray_org_;
		RayCoeff ray_coeff_;
		T t_min_;

		T t_;
		uint32_t prim_id_{};
	};
	
	template <typename T = float, class H = TriangleIntersection<T>>
	class TriangleNoIndicesIntersector
	{
	public:
		TriangleNoIndicesIntersector(const T* vertices) : vertices_(vertices) {}

		// For Watertight Ray/Triangle Intersection.
		typedef struct
		{
			T Sx;
			T Sy;
			T Sz;
			int kx;
			int ky;
			int kz;
		} RayCoeff;

		/// Do ray intersection stuff for `prim_index` th primitive and return hit
		/// distance `t`, barycentric coordinate `u` and `v`.
		/// Returns true if there's intersection.
		bool Intersect(T* t_inout, const uint32_t prim_index) const
		{
			const auto& p0 = *(real3<T>*)(get_vertex_addr(vertices_, 3 * prim_index, sizeof(T) * 3));
			const auto& p1 = *(real3<T>*)(get_vertex_addr(vertices_, 3 * prim_index + 1, sizeof(T) * 3));
			const auto& p2 = *(real3<T>*)(get_vertex_addr(vertices_, 3 * prim_index + 2, sizeof(T) * 3));

			const real3<T> A = p0 - ray_org_;
			const real3<T> B = p1 - ray_org_;
			const real3<T> C = p2 - ray_org_;

			const auto kx = ray_coeff_.kx;
			const auto ky = ray_coeff_.ky;
			const auto kz = ray_coeff_.kz;
			const auto Sx = ray_coeff_.Sx;
			const auto Sy = ray_coeff_.Sy;
			const auto Sz = ray_coeff_.Sz;
			const T Ax = A[kx] - Sx * A[kz];
			const T Ay = A[ky] - Sy * A[kz];
			const T Bx = B[kx] - Sx * B[kz];
			const T By = B[ky] - Sy * B[kz];
			const T Cx = C[kx] - Sx * C[kz];
			const T Cy = C[ky] - Sy * C[kz];
			T U = Cx * By - Cy * Bx;
			T V = Ax * Cy - Ay * Cx;
			T W = Bx * Ay - By * Ax;

			// Fall back to test against edges using double precision.
			if (U == T(0.0) || V == T(0.0) || W == T(0.0))
			{
				U = T(double(Cx) * double(By) - double(Cy) * double(Bx));
				V = T(double(Ax) * double(Cy) - double(Ay) * double(Cx));
				W = T(double(Bx) * double(Ay) - double(By) * double(Ax));
			}

			T det = U + V + W;
			if (U < T(0.0) || V < T(0.0) || W < T(0.0) || det == T(0.0))
			{
				return false;
			}

			const T Az = Sz * A[kz];
			const T Bz = Sz * B[kz];
			const T Cz = Sz * C[kz];
			const T D = U * Az + V * Bz + W * Cz;
			const T tt = D / det;
			if (tt > (*t_inout) || tt < t_min_)
			{
				return false;
			}

			*t_inout = tt;
			return true;
		}

		/// Returns the nearest hit distance.
		T GetT() const { return t_; }

		/// Update is called when initializing intersection and nearest hit is found.
		void Update(T t, uint32_t prim_idx)
		{
			t_ = t;
			prim_id_ = prim_idx;
		}

		/// Prepare BVH traversal (e.g. compute inverse ray direction)
		/// This function is called only once in BVH traversal.
		void PrepareTraversal(const Ray<T>& ray)
		{
			ray_org_[0] = ray.org[0];
			ray_org_[1] = ray.org[1];
			ray_org_[2] = ray.org[2];

			// Calculate dimension where the ray direction is maximal.
			ray_coeff_.kz = 0;
			T absDir = std::fabs(ray.dir[0]);
			if (absDir < std::fabs(ray.dir[1]))
			{
				ray_coeff_.kz = 1;
				absDir = std::fabs(ray.dir[1]);
			}
			if (absDir < std::fabs(ray.dir[2]))
			{
				ray_coeff_.kz = 2;
				absDir = std::fabs(ray.dir[2]);
			}

			ray_coeff_.kx = ray_coeff_.kz + 1;
			if (ray_coeff_.kx == 3) ray_coeff_.kx = 0;
			ray_coeff_.ky = ray_coeff_.kx + 1;
			if (ray_coeff_.ky == 3) ray_coeff_.ky = 0;

			// Swap kx and ky dimension to preserve winding direction of triangles.
			if (ray.dir[ray_coeff_.kz] < T(0.0)) std::swap(ray_coeff_.kx, ray_coeff_.ky);

			// Calculate shear constants.
			ray_coeff_.Sx = ray.dir[ray_coeff_.kx] / ray.dir[ray_coeff_.kz];
			ray_coeff_.Sy = ray.dir[ray_coeff_.ky] / ray.dir[ray_coeff_.kz];
			ray_coeff_.Sz = T(1.0) / ray.dir[ray_coeff_.kz];

			t_min_ = ray.min_t;
		}

		/// Post BVH traversal stuff.
		/// Fill `isect` if there is a hit.
		void PostTraversal(bool hit, H* isect) const
		{
			if (hit && isect)
			{
				isect->t = t_;
				isect->prim_id = prim_id_;
			}
		}

	private:
		const T* vertices_;

		real3<T> ray_org_;
		RayCoeff ray_coeff_;
		T t_min_;

		T t_;
		uint32_t prim_id_{};
	};

	///
	/// Intersector is a template class which implements intersection method and stores
	/// intesection point information(`H`)
	///
	/// @tparam T Precision(float or double)
	/// @tparam H Intersection point information struct
	///
	template <typename T = float, class H = TriangleIntersection<T>>
	class TriangleDownIntersector
	{
	public:
		TriangleDownIntersector(const T* vertices, const uint32_t* faces) : vertices_(vertices), faces_(faces) {}

		/// Do ray intersection stuff for `prim_index` th primitive and return hit
		/// distance `t`, barycentric coordinate `u` and `v`.
		/// Returns true if there's intersection.
		bool Intersect(T* t_inout, const uint32_t prim_index) const
		{
			const uint32_t f0 = faces_[3 * prim_index + 0];
			const uint32_t f1 = faces_[3 * prim_index + 1];
			const uint32_t f2 = faces_[3 * prim_index + 2];

			const auto& p0 = *(real3<T>*)(get_vertex_addr(vertices_, f0, sizeof(T) * 3));
			const auto& p1 = *(real3<T>*)(get_vertex_addr(vertices_, f1, sizeof(T) * 3));
			const auto& p2 = *(real3<T>*)(get_vertex_addr(vertices_, f2, sizeof(T) * 3));

			const real3<T> A = p0 - ray_org_;
			const real3<T> B = p1 - ray_org_;
			const real3<T> C = p2 - ray_org_;

			T U = C[0] * B[2] - C[2] * B[0];
			T V = A[0] * C[2] - A[2] * C[0];
			T W = B[0] * A[2] - B[2] * A[0];

			// Fall back to test against edges using double precision.
			if (U == T(0.0) || V == T(0.0) || W == T(0.0))
			{
				U = T(double(C[0]) * double(B[2]) - double(C[2]) * double(B[0]));
				V = T(double(A[0]) * double(C[2]) - double(A[2]) * double(C[0]));
				W = T(double(B[0]) * double(A[2]) - double(B[2]) * double(A[0]));
			}

			T det = U + V + W;
			if (U < T(0.0) || V < T(0.0) || W < T(0.0) || det == T(0.0))
			{
				return false;
			}

			const T D = U * A[1] + V * B[1] + W * C[1];
			const T tt = -D / det;
			if (tt > (*t_inout) || tt < t_min_)
			{
				return false;
			}

			*t_inout = tt;
			return true;
		}

		/// Returns the nearest hit distance.
		T GetT() const { return t_; }

		/// Update is called when initializing intersection and nearest hit is found.
		void Update(T t, uint32_t prim_idx)
		{
			t_ = t;
			prim_id_ = prim_idx;
		}

		/// Prepare BVH traversal (e.g. compute inverse ray direction)
		/// This function is called only once in BVH traversal.
		void PrepareTraversal(const Ray<T>& ray)
		{
			ray_org_[0] = ray.org[0];
			ray_org_[1] = ray.org[1];
			ray_org_[2] = ray.org[2];
			t_min_ = ray.min_t;
		}

		/// Post BVH traversal stuff.
		/// Fill `isect` if there is a hit.
		void PostTraversal(bool hit, H* isect) const
		{
			if (hit && isect)
			{
				isect->t = t_;
				isect->prim_id = prim_id_;
			}
		}

	private:
		const T* vertices_;
		const uint32_t* faces_;

		real3<T> ray_org_;
		T t_min_;

		T t_;
		uint32_t prim_id_;
	};

	//
	// Robust BVH Ray Traversal : http://jcgt.org/published/0002/02/02/paper.pdf
	//

	// NaN-safe min and max function.
	template <class T>
	const T& safemin(const T& a, const T& b) { return (a < b) ? a : b; }

	template <class T>
	const T& safemax(const T& a, const T& b) { return (a > b) ? a : b; }

	//
	// SAH functions
	//
	struct BinBuffer
	{
		explicit BinBuffer(uint32_t size)
		{
			bin_size = size;
			bin.resize(2 * 3 * size);
			clear();
		}

		void clear() { memset(&bin[0], 0, sizeof(size_t) * 2 * 3 * bin_size); }

		std::vector<size_t> bin; // (min, max) * xyz * binsize
		uint32_t bin_size;
		uint32_t pad0;
	};

	template <typename T>
	T CalculateSurfaceArea(const real3<T>& min, const real3<T>& max)
	{
		real3<T> box = max - min;
		return T(2.0) * (box[0] * box[1] + box[1] * box[2] + box[2] * box[0]);
	}

	template <typename T>
	void GetBoundingBoxOfTriangle(real3<T>* bmin, real3<T>* bmax, const T* vertices, const uint32_t* faces, uint32_t index)
	{
		uint32_t f0 = faces[3 * index + 0];
		uint32_t f1 = faces[3 * index + 1];
		uint32_t f2 = faces[3 * index + 2];

		real3<T> p[3];
		p[0] = real3<T>(&vertices[3 * f0]);
		p[1] = real3<T>(&vertices[3 * f1]);
		p[2] = real3<T>(&vertices[3 * f2]);

		(*bmin) = p[0];
		(*bmax) = p[0];

		for (int i = 1; i < 3; i++)
		{
			(*bmin)[0] = std::min((*bmin)[0], p[i][0]);
			(*bmin)[1] = std::min((*bmin)[1], p[i][1]);
			(*bmin)[2] = std::min((*bmin)[2], p[i][2]);

			(*bmax)[0] = std::max((*bmax)[0], p[i][0]);
			(*bmax)[1] = std::max((*bmax)[1], p[i][1]);
			(*bmax)[2] = std::max((*bmax)[2], p[i][2]);
		}
	}

	template <typename T, class P>
	void ContributeBinBuffer(BinBuffer* bins, const real3<T>& scene_min, const real3<T>& scene_max, uint32_t* indices, uint32_t left_idx, uint32_t right_idx, const P& p)
	{
		T bin_size = T(bins->bin_size);

		// Calculate extent
		real3<T> scene_inv_size;
		real3<T> scene_size = scene_max - scene_min;

		for (int i = 0; i < 3; ++i)
		{
			_assert(scene_size[i] >= T(0.0));
			if (scene_size[i] > T(0.0))
			{
				scene_inv_size[i] = bin_size / scene_size[i];
			}
			else
			{
				scene_inv_size[i] = T(0.0);
			}
		}

		// Clear bin data
		std::fill(bins->bin.begin(), bins->bin.end(), 0);

		size_t idx_bmin[3];
		size_t idx_bmax[3];
		for (size_t i = left_idx; i < right_idx; i++)
		{
			//
			// Quantize the position into [0, BIN_SIZE)
			//
			// q[i] = (int)(p[i] - scene_bmin) / scene_size
			//
			real3<T> bmin;
			real3<T> bmax;

			p.BoundingBox(&bmin, &bmax, indices[i]);
			// GetBoundingBoxOfTriangle(&bmin, &bmax, vertices, faces, indices[i]);

			real3<T> quantized_bmin = (bmin - scene_min) * scene_inv_size;
			real3<T> quantized_bmax = (bmax - scene_min) * scene_inv_size;

			// idx is now in [0, BIN_SIZE)
			for (int j = 0; j < 3; ++j)
			{
				int q0 = static_cast<int>(quantized_bmin[j]);
				if (q0 < 0) q0 = 0;
				int q1 = static_cast<int>(quantized_bmax[j]);
				if (q1 < 0) q1 = 0;

				idx_bmin[j] = uint32_t(q0);
				idx_bmax[j] = uint32_t(q1);

				if (idx_bmin[j] >= bin_size) idx_bmin[j] = uint32_t(bin_size) - 1;

				if (idx_bmax[j] >= bin_size) idx_bmax[j] = uint32_t(bin_size) - 1;

				// Increment bin counter
				bins->bin[0 * (bins->bin_size * 3) +
					static_cast<size_t>(j) * bins->bin_size + idx_bmin[j]] += 1;
				bins->bin[1 * (bins->bin_size * 3) +
					static_cast<size_t>(j) * bins->bin_size + idx_bmax[j]] += 1;
			}
		}
	}

	template <typename T>
	T SAH(size_t ns1, T leftArea, size_t ns2, T rightArea, T invS, T Taabb, T Ttri)
	{
		return T(2.0) * Taabb +
			(leftArea * invS) * T(ns1) * Ttri +
			(rightArea * invS) * T(ns2) * Ttri;
	}

	template <typename T>
	bool FindCutFromBinBuffer(T* cut_pos, int* minCostAxis, const BinBuffer* bins, const real3<T>& bmin, const real3<T>& bmax, size_t num_primitives, T costTaabb)
	{
		// should be in [0.0, 1.0]
		const T kEPS = std::numeric_limits<T>::epsilon(); // * epsScale;

		real3<T> bminRight, bmaxRight;
		T minCost[3];
		T costTtri = T(1.0) - costTaabb;
		*minCostAxis = 0;
		real3<T> bsize = bmax - bmin;
		real3<T> bstep = bsize * (T(1.0) / bins->bin_size);
		T saTotal = CalculateSurfaceArea(bmin, bmax);
		T invSaTotal = T(0.0);
		if (saTotal > kEPS)
		{
			invSaTotal = T(1.0) / saTotal;
		}

		for (int j = 0; j < 3; ++j)
		{
			//
			// Compute SAH cost for the right side of each cell of the bbox.
			// Exclude both extreme sides of the bbox.
			//
			//  i:      0    1    2    3
			//     +----+----+----+----+----+
			//     |    |    |    |    |    |
			//     +----+----+----+----+----+
			//

			T minCostPos = bmin[j] + T(1.0) * bstep[j];
			minCost[j] = std::numeric_limits<T>::max();

			size_t left = 0;
			size_t right = num_primitives;
			real3<T> bminLeft = bminRight = bmin;
			real3<T> bmaxLeft = bmaxRight = bmax;

			for (int i = 0; i < static_cast<int>(bins->bin_size) - 1; ++i)
			{
				left += bins->bin[0 * (3 * bins->bin_size) +
					static_cast<size_t>(j) * bins->bin_size +
					static_cast<size_t>(i)];
				right -= bins->bin[1 * (3 * bins->bin_size) +
					static_cast<size_t>(j) * bins->bin_size +
					static_cast<size_t>(i)];

				_assert(left <= num_primitives);
				_assert(right <= num_primitives);

				//
				// Split pos bmin + (i + 1) * (bsize / BIN_SIZE)
				// +1 for i since we want a position on right side of the cell.
				//

				T pos = bmin[j] + (i + T(1.0)) * bstep[j];
				bmaxLeft[j] = pos;
				bminRight[j] = pos;

				T saLeft = CalculateSurfaceArea(bminLeft, bmaxLeft);
				T saRight = CalculateSurfaceArea(bminRight, bmaxRight);

				T cost =
					SAH(left, saLeft, right, saRight, invSaTotal, costTaabb, costTtri);

				if (cost < minCost[j])
				{
					//
					// Update the min cost
					//
					minCost[j] = cost;
					minCostPos = pos;
					// minCostAxis = j;
				}
			}

			cut_pos[j] = minCostPos;
		}

		// Find min cost axis
		T cost = minCost[0];
		*minCostAxis = 0;

		if (cost > minCost[1])
		{
			*minCostAxis = 1;
			cost = minCost[1];
		}
		if (cost > minCost[2])
		{
			*minCostAxis = 2;
			cost = minCost[2];
		}

		return true;
	}

	template <typename T, class P>
	void ComputeBoundingBox(real3<T>* bmin, real3<T>* bmax, const uint32_t* indices, uint32_t left_index, uint32_t right_index, const P& p)
	{
		uint32_t idx = indices[left_index];
		p.BoundingBox(bmin, bmax, idx);

		{
			// for each primitive
			for (uint32_t i = left_index + 1; i < right_index; i++)
			{
				idx = indices[i];
				real3<T> bbox_min, bbox_max;
				p.BoundingBox(&bbox_min, &bbox_max, idx);

				// xyz
				for (int k = 0; k < 3; k++)
				{
					(*bmin)[k] = std::min((*bmin)[k], bbox_min[k]);
					(*bmax)[k] = std::max((*bmax)[k], bbox_max[k]);
				}
			}
		}
	}

	template <typename T>
	void GetBoundingBox(real3<T>* bmin, real3<T>* bmax, const std::vector<BBox<T>>& bboxes, uint32_t* indices, uint32_t left_index, uint32_t right_index)
	{
		uint32_t i = left_index;
		uint32_t idx = indices[i];

		(*bmin)[0] = bboxes[idx].bmin[0];
		(*bmin)[1] = bboxes[idx].bmin[1];
		(*bmin)[2] = bboxes[idx].bmin[2];
		(*bmax)[0] = bboxes[idx].bmax[0];
		(*bmax)[1] = bboxes[idx].bmax[1];
		(*bmax)[2] = bboxes[idx].bmax[2];

		// for each face
		for (i = left_index + 1; i < right_index; i++)
		{
			idx = indices[i];

			// xyz
			for (int k = 0; k < 3; k++)
			{
				(*bmin)[k] = std::min((*bmin)[k], bboxes[idx].bmin[k]);
				(*bmax)[k] = std::max((*bmax)[k], bboxes[idx].bmax[k]);
			}
		}
	}

	//
	// --
	//

	template <typename T>
	template <class P, class Pred>
	uint32_t BVHAccel<T>::BuildTree(std::vector<BVHNode<T>>* out_nodes, uint32_t left_idx, uint32_t right_idx, uint32_t depth, const P& p,
		const Pred& pred)
	{
		_assert(left_idx <= right_idx);

		uint32_t offset = uint32_t(out_nodes->size());
		real3<T> bmin, bmax;
		if (!bboxes_.empty())
		{
			GetBoundingBox(&bmin, &bmax, bboxes_, &indices_.at(0), left_idx, right_idx);
		}
		else
		{
			ComputeBoundingBox(&bmin, &bmax, &indices_.at(0), left_idx, right_idx, p);
		}

		uint32_t n = right_idx - left_idx;
		if ((n <= options_.min_leaf_primitives) ||
			(depth >= options_.max_tree_depth))
		{
			// Create leaf node.
			BVHNode<T> leaf;

			leaf.bmin[0] = bmin[0];
			leaf.bmin[1] = bmin[1];
			leaf.bmin[2] = bmin[2];

			leaf.bmax[0] = bmax[0];
			leaf.bmax[1] = bmax[1];
			leaf.bmax[2] = bmax[2];

			_assert(left_idx < std::numeric_limits<uint32_t>::max());

			leaf.flag = 1; // leaf
			leaf.data[0] = n;
			leaf.data[1] = left_idx;

			out_nodes->push_back(leaf); // atomic update
			return offset;
		}

		//
		// Create branch node.
		//

		//
		// Compute SAH and find best split axis and position
		//
		int min_cut_axis = 0;
		T cut_pos[3] = {0.0, 0.0, 0.0};

		BinBuffer bins(options_.bin_size);
		ContributeBinBuffer(&bins, bmin, bmax, &indices_.at(0), left_idx, right_idx, p);
		FindCutFromBinBuffer(cut_pos, &min_cut_axis, &bins, bmin, bmax, n, options_.cost_t_aabb);

		// Try all 3 axis until good cut position avaiable.
		uint32_t mid_idx = left_idx;
		int cut_axis = min_cut_axis;

		for (int axis_try = 0; axis_try < 3; axis_try++)
		{
			auto begin = &indices_[left_idx];
			auto end = &indices_[right_idx - 1] + 1; // mimics end() iterator.

			// try min_cut_axis first.
			cut_axis = (min_cut_axis + axis_try) % 3;
			pred.Set(cut_axis, cut_pos[cut_axis]);

			// Split at (cut_axis, cut_pos), indices_ will be modified.
			auto mid = std::partition(begin, end, pred);

			mid_idx = left_idx + uint32_t((mid - begin));

			if (mid_idx == left_idx || mid_idx == right_idx)
			{
				// Can't split well. Switch to object median(which may create unoptimized tree, but stable).
				mid_idx = left_idx + (n >> 1);
			}
			else
			{
				// Found good cut. exit loop.
				break;
			}
		}

		BVHNode<T> node;
		node.axis = cut_axis;
		node.flag = 0; // 0 = branch

		out_nodes->push_back(node);

		uint32_t left_child_index = BuildTree(out_nodes, left_idx, mid_idx, depth + 1, p, pred);
		uint32_t right_child_index = BuildTree(out_nodes, mid_idx, right_idx, depth + 1, p, pred);

		{
			(*out_nodes)[offset].data[0] = left_child_index;
			(*out_nodes)[offset].data[1] = right_child_index;

			(*out_nodes)[offset].bmin[0] = bmin[0];
			(*out_nodes)[offset].bmin[1] = bmin[1];
			(*out_nodes)[offset].bmin[2] = bmin[2];

			(*out_nodes)[offset].bmax[0] = bmax[0];
			(*out_nodes)[offset].bmax[1] = bmax[1];
			(*out_nodes)[offset].bmax[2] = bmax[2];
		}

		return offset;
	}

	template <typename T>
	template <class Prim, class Pred>
	bool BVHAccel<T>::Build(uint32_t num_primitives, const Prim& p,
		const Pred& pred, const BVHBuildOptions<T>& options)
	{
		options_ = options;

		nodes_.clear();
		bboxes_.clear();

		if (num_primitives == 0)
		{
			return false;
		}

		uint32_t n = num_primitives;

		//
		// 1. Create triangle indices(this will be permutated in BuildTree)
		//
		indices_.resize(n);
		for (int i = 0; i < static_cast<int>(n); i++)
		{
			indices_[static_cast<size_t>(i)] = uint32_t(i);
		}

		//
		// 2. Compute bounding box (optional).
		//
		real3<T> bmin, bmax;

		if (options.cache_bbox)
		{
			bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<T>::max();
			bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<T>::max();

			bboxes_.resize(n);

			for (size_t i = 0; i < n; i++)
			{
				// for each primitive
				uint32_t idx = indices_[i];

				BBox<T> bbox;
				p.BoundingBox(&(bbox.bmin), &(bbox.bmax), uint32_t(i));
				bboxes_[idx] = bbox;

				// xyz
				for (int k = 0; k < 3; k++)
				{
					bmin[k] = std::min(bmin[k], bbox.bmin[k]);
					bmax[k] = std::max(bmax[k], bbox.bmax[k]);
				}
			}
		}
		else
		{
			ComputeBoundingBox(&bmin, &bmax, &indices_.at(0), 0, n, p);
		}

		//
		// 3. Build tree
		//
		BuildTree(&nodes_, 0, n, /* root depth */ 0, p, pred); // [0, n)
		return true;
	}

	template <typename T>
	bool IntersectRayAABB(T* tminOut, T* tmaxOut, T min_t, T max_t, const T bmin[3], const T bmax[3], real3<T> ray_org, real3<T> ray_inv_dir, int ray_dir_sign[3]);

	template <>
	inline bool IntersectRayAABB<float>(float* tminOut, float* tmaxOut, float min_t, float max_t, const float bmin[3], const float bmax[3], real3<float> ray_org,
		real3<float> ray_inv_dir, int ray_dir_sign[3])
	{
		const float min_x = ray_dir_sign[0] ? bmax[0] : bmin[0];
		const float min_y = ray_dir_sign[1] ? bmax[1] : bmin[1];
		const float min_z = ray_dir_sign[2] ? bmax[2] : bmin[2];
		const float max_x = ray_dir_sign[0] ? bmin[0] : bmax[0];
		const float max_y = ray_dir_sign[1] ? bmin[1] : bmax[1];
		const float max_z = ray_dir_sign[2] ? bmin[2] : bmax[2];

		// X
		const float tmin_x = (min_x - ray_org[0]) * ray_inv_dir[0];
		// MaxMult robust BVH traversal(up to 4 ulp).
		// 1.0000000000000004 for double precision.
		const float tmax_x = (max_x - ray_org[0]) * ray_inv_dir[0] * 1.00000024f;

		// Y
		const float tmin_y = (min_y - ray_org[1]) * ray_inv_dir[1];
		const float tmax_y = (max_y - ray_org[1]) * ray_inv_dir[1] * 1.00000024f;

		// Z
		const float tmin_z = (min_z - ray_org[2]) * ray_inv_dir[2];
		const float tmax_z = (max_z - ray_org[2]) * ray_inv_dir[2] * 1.00000024f;

		float tmin = safemax(tmin_z, safemax(tmin_y, safemax(tmin_x, min_t)));
		float tmax = safemin(tmax_z, safemin(tmax_y, safemin(tmax_x, max_t)));

		if (tmin <= tmax)
		{
			*tminOut = tmin;
			*tmaxOut = tmax;

			return true;
		}
		return false; // no hit
	}

	template <typename T>
	bool IntersectRayDownAABB(T* tminOut, T* tmaxOut, T min_t, T max_t, const T bmin[3], const T bmax[3], real3<T> ray_org);

	template <>
	inline bool IntersectRayDownAABB<float>(float* tminOut, float* tmaxOut, float min_t, float max_t, const float bmin[3], const float bmax[3], real3<float> ray_org)
	{
		const auto tmin_x = (bmin[0] - ray_org[0]) * std::numeric_limits<float>::infinity();
		const auto tmax_x = (bmax[0] - ray_org[0]) * std::numeric_limits<float>::infinity() * 1.00000024f;
		const auto tmin_y = -(bmax[1] - ray_org[1]);
		const auto tmax_y = -(bmin[1] - ray_org[1]) * 1.00000024f;
		const auto tmin_z = (bmin[2] - ray_org[2]) * std::numeric_limits<float>::infinity();
		const auto tmax_z = (bmax[2] - ray_org[2]) * std::numeric_limits<float>::infinity() * 1.00000024f;
		const auto tmin = safemax(tmin_z, safemax(tmin_y, safemax(tmin_x, min_t)));
		const auto tmax = safemin(tmax_z, safemin(tmax_y, safemin(tmax_x, max_t)));
		if (tmin <= tmax)
		{
			*tminOut = tmin;
			*tmaxOut = tmax;
			return true;
		}
		return false; // no hit
	}

	template <typename T>
	bool IntersectRayDownAABBSimple(const T bmin[3], const T bmax[3], const float* ray_org);

	template <>
	inline bool IntersectRayDownAABBSimple<float>(const float bmin[3], const float bmax[3], const float* ray_org)
	{
		return bmin[0] < ray_org[0] && ray_org[0] < bmax[0] && bmin[2] < ray_org[2] && ray_org[2] < bmax[2];
	}

	template <typename T>
	template <bool TraverseDown, class I>
	bool BVHAccel<T>::TestLeafNode(const BVHNode<T>& node, I& intersector) const
	{
		bool hit = false;
		uint32_t num_primitives = node.data[0];
		uint32_t offset = node.data[1];
		T t = intersector.GetT(); // current hit distance
		for (uint32_t i = 0; i < num_primitives; i++)
		{
			uint32_t prim_idx = indices_[i + offset];
			if (intersector.Intersect(&t, prim_idx))
			{
				intersector.Update(t, prim_idx);
				hit = true;
			}
		}
		return hit;
	}

	template <typename T>
	template <bool TraverseDown, class I, class H>
	bool BVHAccel<T>::Traverse(const Ray<T>& ray, I& intersector, H* isect) const
	{
		const int kMaxStackDepth = 512;
		(void)kMaxStackDepth;

		T hit_t = ray.max_t;

		auto node_stack_index = 0;
		uint32_t node_stack[512];
		node_stack[0] = 0;

		// Init isect info as no hit
		intersector.Update(hit_t, uint32_t(-1));
		intersector.PrepareTraversal(ray);

		int dir_sign[3];
		dir_sign[0] = ray.dir[0] < T(0.0) ? 1 : 0;
		dir_sign[1] = ray.dir[1] < T(0.0) ? 1 : 0;
		dir_sign[2] = ray.dir[2] < T(0.0) ? 1 : 0;

		real3<T> ray_dir(ray.dir[0], ray.dir[1], ray.dir[2]);
		real3<T> ray_inv_dir = vsafe_inverse(ray_dir);
		real3<T> ray_org(ray.org[0], ray.org[1], ray.org[2]);
		T min_t = std::numeric_limits<T>::max();
		T max_t = -std::numeric_limits<T>::max();

		while (node_stack_index >= 0)
		{
			uint32_t index = node_stack[node_stack_index];
			const BVHNode<T>& node = nodes_[index];

			node_stack_index--;

			if (TraverseDown
				? IntersectRayDownAABB(&min_t, &max_t, ray.min_t, hit_t, node.bmin, node.bmax, ray_org)
				: IntersectRayAABB(&min_t, &max_t, ray.min_t, hit_t, node.bmin, node.bmax, ray_org, ray_inv_dir, dir_sign))
			{
				// Branch node
				if (node.flag == 0)
				{
					int order_near = dir_sign[node.axis];
					int order_far = 1 - order_near;

					// Traverse near first.
					node_stack[++node_stack_index] = node.data[order_far];
					node_stack[++node_stack_index] = node.data[order_near];
				}
				else if (TestLeafNode<TraverseDown>(node, intersector))
				{
					// Leaf node
					hit_t = intersector.GetT();
				}
			}
		}

		bool hit = intersector.GetT() < ray.max_t;
		intersector.PostTraversal(hit, isect);
		return hit;
	}


	template <typename T>
	void BVHAccel<T>::TraverseDownSimple(const T* vertices, const uint32_t* faces, const float* pos, float& hit_distance, uint32_t& hit_prim_idx) const
	{
		auto node_stack_index = 0;
		uint32_t node_stack[512];
		node_stack[0] = 0;

		while (node_stack_index >= 0)
		{
			const BVHNode<T>& node = nodes_[node_stack[node_stack_index--]];
			if (node.bmin[0] < pos[0] && pos[0] < node.bmax[0]
				&& node.bmin[2] < pos[2] && pos[2] < node.bmax[2]
				&& node.bmin[1] < pos[1] && pos[1] < node.bmax[1] + hit_distance)
			{
				if (node.flag == 0)
				{
					node_stack[++node_stack_index] = node.data[1];
					node_stack[++node_stack_index] = node.data[0];
				}
				else
				{
					for (uint32_t i = 0, s = node.data[0], o = node.data[1]; i < s; i++)
					{
						const auto x = indices_[i + o];
						const auto a = *(real3<T>*)pos - *(real3<T>*)&vertices[faces[3 * x] * 3];
						const auto b = *(real3<T>*)pos - *(real3<T>*)&vertices[faces[3 * x + 1] * 3];
						const auto c = *(real3<T>*)pos - *(real3<T>*)&vertices[faces[3 * x + 2] * 3];
						const T u = c[0] * b[2] - c[2] * b[0];
						const T v = a[0] * c[2] - a[2] * c[0];
						const T w = b[0] * a[2] - b[2] * a[0];
						const T t = (u * a[1] + v * b[1] + w * c[1]) / (u + v + w);
						if (u >= T(0.0) && v >= T(0.0) && w >= T(0.0) && t >= T(0.0) && t < hit_distance)
						{
							hit_distance = t;
							hit_prim_idx = x;
						}
					}
				}
			}
		}
	}
}
