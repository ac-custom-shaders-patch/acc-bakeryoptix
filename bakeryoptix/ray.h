#pragma once

#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1
#define DENOMINATOR_EPSILON 1.0e-6f
#define USE_DEBUG_EXCEPTIONS 0

#ifndef RT_FUNCTION
#define RT_FUNCTION __forceinline__ __device__
#endif

#include <optix.h>
#include <cuda/random.h>
#include <optixu/optixu_math_namespace.h>

#define FLAG_FRONTFACE      0x00000010
#define FLAG_TERMINATE      0x80000000

//using optix::int;
using optix::int2;
using optix::uint;
using optix::uint2;
//using optix::float;
using optix::float3;
using optix::float4;

struct PerRayData
{
  optix::float3 pos;            // Current surface hit point, in world space
  optix::float3 wo;             // Outgoing direction, to observer, in world space.
  optix::float3 wi;             // Incoming direction, to light, in world space.

  optix::float3 radiance;       // Radiance along the current path segment.
  int           flags;          // Bitfield with flags. See FLAG_* defines for its contents.

  optix::float3 f_over_pdf;     // BSDF sample throughput, pre-multiplied f_over_pdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf; 
  float         pdf;            // The last BSDF sample's pdf, tracked for multiple importance sampling.

  unsigned int  seed;           // Random number generator input.
};

// Tiny Encryption Algorithm (TEA) to calculate a the seed per launch index and iteration.
template<unsigned int N>
RT_FUNCTION unsigned int tea_custom(const unsigned int val0, const unsigned int val1)
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for (unsigned int n = 0; n < N; ++n)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
    v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
  }
  return v0;
}

// Return a random sample in the range [0, 1) with a simple Linear Congruential Generator.
RT_FUNCTION float rng(unsigned int& previous)
{
  previous = previous * 1664525u + 1013904223u;
  
  return float(previous & 0X00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
  // return float(previous >> 8) / float(0x01000000u);      // Use the upper 24 bits
}

// Convenience function to generate a 2D unit square sample.
RT_FUNCTION float2 rng2(unsigned int& previous)
{
  float2 s;

  previous = previous * 1664525u + 1013904223u;
  s.x = float(previous & 0X00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
  //s.x = float(previous >> 8) / float(0x01000000u);      // Use the upper 24 bits

  previous = previous * 1664525u + 1013904223u;
  s.y = float(previous & 0X00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
  //s.y = float(previous >> 8) / float(0x01000000u);      // Use the upper 24 bits

  return s;
}

RT_FUNCTION bool isNull(const optix::float3& v)
{
  return (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f);
}