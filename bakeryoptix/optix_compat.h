// Drop-in replacement for CUDA vector_types.h, optixu_math_namespace.h and optixu_matrix_namespace.h
// Provides all types and functions previously supplied by NVIDIA OptiX/CUDA headers.
#pragma once

#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef M_1_PIf
#define M_1_PIf (1.0f / M_PIf)
#endif

// CUDA qualifier stubs (no-op on CPU)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __inline__
#define __inline__ inline
#endif

// ============================================================
// Basic vector types (matching CUDA vector_types.h layout)
// ============================================================

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct int2   { int x, y; };
struct int3   { int x, y, z; };
struct uint2  { unsigned int x, y; };
struct uint3  { unsigned int x, y, z; };
struct uchar4 { unsigned char x, y, z, w; };

// ============================================================
// Make functions
// ============================================================

inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float3 make_float3(const float4& v) { return {v.x, v.y, v.z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline float4 make_float4(const float3& v, float w) { return {v.x, v.y, v.z, w}; }
inline int3   make_int3(int x, int y, int z) { return {x, y, z}; }
inline uint2  make_uint2(unsigned x, unsigned y) { return {x, y}; }
inline uint3  make_uint3(unsigned x, unsigned y, unsigned z) { return {x, y, z}; }
inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { return {x, y, z, w}; }

// ============================================================
// float2 operators
// ============================================================

inline float2 operator+(const float2& a, const float2& b) { return {a.x + b.x, a.y + b.y}; }
inline float2 operator-(const float2& a, const float2& b) { return {a.x - b.x, a.y - b.y}; }
inline float2 operator*(const float2& a, float s) { return {a.x * s, a.y * s}; }
inline float2 operator*(float s, const float2& a) { return {a.x * s, a.y * s}; }
inline float2 operator/(const float2& a, float s) { return {a.x / s, a.y / s}; }
inline float2& operator+=(float2& a, const float2& b) { a.x += b.x; a.y += b.y; return a; }
inline float2& operator-=(float2& a, const float2& b) { a.x -= b.x; a.y -= b.y; return a; }
inline float2& operator*=(float2& a, float s) { a.x *= s; a.y *= s; return a; }
inline float2& operator/=(float2& a, float s) { a.x /= s; a.y /= s; return a; }

// ============================================================
// float3 operators
// ============================================================

inline float3 operator+(const float3& a, const float3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline float3 operator-(const float3& a, const float3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float3 operator*(const float3& a, const float3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline float3 operator*(const float3& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float3 operator*(float s, const float3& a) { return {a.x * s, a.y * s, a.z * s}; }
inline float3 operator/(const float3& a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline float3 operator-(const float3& a) { return {-a.x, -a.y, -a.z}; }
inline float3& operator+=(float3& a, const float3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline float3& operator-=(float3& a, const float3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
inline float3& operator*=(float3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }

// ============================================================
// float4 operators
// ============================================================

inline float4 operator+(const float4& a, const float4& b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline float4 operator-(const float4& a, const float4& b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
inline float4 operator*(const float4& a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
inline float4 operator*(float s, const float4& a) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }

// ============================================================
// int3 operators
// ============================================================

inline int3 operator+(const int3& a, const int3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

// ============================================================
// Math functions
// ============================================================

inline float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(const float2& a, const float2& b) { return a.x * b.x + a.y * b.y; }
inline float length(const float3& v) { return sqrtf(dot(v, v)); }
inline float length(const float2& v) { return sqrtf(dot(v, v)); }
inline float3 normalize(const float3& v) { float inv = 1.0f / length(v); return v * inv; }
inline float3 cross(const float3& a, const float3& b) {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float clamp(float v, float lo, float hi) { return (std::min)((std::max)(v, lo), hi); }
inline float3 clamp(const float3& v, float lo, float hi) { return {clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi)}; }
inline float lerp(float a, float b, float t) { return a + t * (b - a); }
inline float3 lerp(const float3& a, const float3& b, float t) { return a + t * (b - a); }

inline float3 fminf(const float3& a, const float3& b) { return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)}; }
inline float3 fmaxf(const float3& a, const float3& b) { return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)}; }

// ============================================================
// Onb (orthonormal basis from a normal)
// ============================================================

struct Onb {
	float3 m_tangent;
	float3 m_binormal;
	float3 m_normal;

	Onb() = default;
	explicit Onb(const float3& normal) : m_normal(normal) {
		if (fabsf(m_normal.x) > fabsf(m_normal.z)) {
			m_binormal = make_float3(-m_normal.y, m_normal.x, 0.0f);
		} else {
			m_binormal = make_float3(0.0f, -m_normal.z, m_normal.y);
		}
		m_binormal = normalize(m_binormal);
		m_tangent = cross(m_binormal, m_normal);
	}

	void inverse_transform(float3& p) const {
		p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
	}
};

// ============================================================
// cosine_sample_hemisphere
// ============================================================

inline void cosine_sample_hemisphere(float u1, float u2, float3& p) {
	const float r = sqrtf(u1);
	const float phi = 2.0f * M_PIf * u2;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);
	p.z = sqrtf(fmaxf(0.0f, 1.0f - u1));
}

// ============================================================
// Matrix4x4 (matching OptiX's optixu::Matrix4x4 API)
// Row-major 4x4 float matrix.
// ============================================================

class Matrix4x4 {
public:
	Matrix4x4() { memset(m_data, 0, sizeof(m_data)); }

	explicit Matrix4x4(const float data[16]) {
		memcpy(m_data, data, sizeof(m_data));
	}

	static Matrix4x4 identity() {
		float d[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
		return Matrix4x4(d);
	}

	const float* getData() const { return m_data; }
	float* getData() { return m_data; }

	float operator[](int i) const { return m_data[i]; }
	float& operator[](int i) { return m_data[i]; }

	Matrix4x4 transpose() const {
		Matrix4x4 r;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				r.m_data[i * 4 + j] = m_data[j * 4 + i];
		return r;
	}

	// Matrix * Matrix
	Matrix4x4 operator*(const Matrix4x4& b) const {
		Matrix4x4 r;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j) {
				float s = 0;
				for (int k = 0; k < 4; ++k)
					s += m_data[i * 4 + k] * b.m_data[k * 4 + j];
				r.m_data[i * 4 + j] = s;
			}
		return r;
	}

	// Matrix * float4 column vector
	float4 operator*(const float4& v) const {
		return {
			m_data[0]*v.x + m_data[1]*v.y + m_data[2]*v.z  + m_data[3]*v.w,
			m_data[4]*v.x + m_data[5]*v.y + m_data[6]*v.z  + m_data[7]*v.w,
			m_data[8]*v.x + m_data[9]*v.y + m_data[10]*v.z + m_data[11]*v.w,
			m_data[12]*v.x + m_data[13]*v.y + m_data[14]*v.z + m_data[15]*v.w
		};
	}

	// 4x4 matrix inverse (general, using cofactor expansion)
	Matrix4x4 inverse() const {
		const float* m = m_data;
		float inv[16];
		inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
		inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
		inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
		inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
		inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
		inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
		inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
		inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
		inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
		inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
		inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
		inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
		inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
		inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
		inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
		inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];
		float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
		if (fabsf(det) < 1e-30f) return Matrix4x4::identity();
		float inv_det = 1.0f / det;
		Matrix4x4 r;
		for (int i = 0; i < 16; ++i) r.m_data[i] = inv[i] * inv_det;
		return r;
	}

	// Rotation around an axis (angle in radians)
	static Matrix4x4 rotate(float angle, const float3& axis) {
		float3 a = normalize(axis);
		float c = cosf(angle), s = sinf(angle), t = 1.0f - c;
		float d[16] = {
			t*a.x*a.x + c,      t*a.x*a.y - s*a.z,  t*a.x*a.z + s*a.y, 0,
			t*a.x*a.y + s*a.z,  t*a.y*a.y + c,       t*a.y*a.z - s*a.x, 0,
			t*a.x*a.z - s*a.y,  t*a.y*a.z + s*a.x,   t*a.z*a.z + c,     0,
			0, 0, 0, 1
		};
		return Matrix4x4(d);
	}

	// Translation matrix
	static Matrix4x4 translate(const float3& t) {
		float d[16] = {
			1, 0, 0, t.x,
			0, 1, 0, t.y,
			0, 0, 1, t.z,
			0, 0, 0, 1
		};
		return Matrix4x4(d);
	}

	// Scale matrix
	static Matrix4x4 scale(const float3& s) {
		float d[16] = {
			s.x, 0,   0,   0,
			0,   s.y, 0,   0,
			0,   0,   s.z, 0,
			0,   0,   0,   1
		};
		return Matrix4x4(d);
	}

private:
	float m_data[16];
};

// Row-vector * Matrix (v^T * M): result[j] = sum_i(v[i] * M[i*4+j])
inline float4 operator*(const float4& v, const Matrix4x4& m) {
	const float* d = m.getData();
	return {
		v.x * d[0] + v.y * d[4] + v.z * d[8]  + v.w * d[12],
		v.x * d[1] + v.y * d[5] + v.z * d[9]  + v.w * d[13],
		v.x * d[2] + v.y * d[6] + v.z * d[10] + v.w * d[14],
		v.x * d[3] + v.y * d[7] + v.z * d[11] + v.w * d[15]
	};
}

// ============================================================
// Namespace aliases for compatibility
// Keep "optix::" namespace usable in existing code.
// ============================================================

namespace optix {
	using ::float2;
	using ::float3;
	using ::float4;
	using ::int2;
	using ::int3;
	using ::uint2;
	using ::uint3;
	using ::uchar4;

	using ::make_float2;
	using ::make_float3;
	using ::make_float4;
	using ::make_int3;
	using ::make_uint2;
	using ::make_uint3;
	using ::make_uchar4;

	using ::dot;
	using ::length;
	using ::normalize;
	using ::cross;
	using ::clamp;
	using ::lerp;

	using ::Onb;
	using ::cosine_sample_hemisphere;

	using ::Matrix4x4;

	typedef unsigned int uint;
}
