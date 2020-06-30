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
	optix::float3 radiance;
};