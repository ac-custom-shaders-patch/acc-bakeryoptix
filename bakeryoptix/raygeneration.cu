/* 
 * Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "ray.h"

rtDeclareVariable(rtObject, sysTopObject, , );

rtBuffer<float3, 1> inNormalsBuffer;
rtBuffer<float3, 1> inFaceNormalsBuffer;
rtBuffer<float3, 1> inPositionsBuffer;
rtBuffer<float, 2> sysOutputBuffer; // RGBA32F

rtDeclareVariable(int, px, , );
rtDeclareVariable(int, py, , );
rtDeclareVariable(int, sqrtPasses, , );
rtDeclareVariable(uint, bounceCounts, , );
rtDeclareVariable(uint, numSamples, , );
rtDeclareVariable(uint, baseSeed, , );
rtDeclareVariable(float, sceneOffsetHorizontal, , );
rtDeclareVariable(float, sceneOffsetVertical, , );
rtDeclareVariable(float, rayDirAlign, , );

rtDeclareVariable(uint2, theLaunchDim, rtLaunchDim, );
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

RT_FUNCTION void integrator(PerRayData& prd, float3& radiance)
{
	radiance = make_float3(0.0f); // Start with black.

	float3 throughput = make_float3(1.0f); // The throughput for the next radiance, starts with 1.0f.
	uint depth = 0U;                       // Path segment index. Primary ray is 0.
	while (depth < bounceCounts)
	{
		prd.wo = -prd.wi; // wi is the next path segment ray.direction. wo is the direction to the observer.
		prd.flags = 0;    // Clear all non-persistent flags. None in this version.

		// Note that the primary rays wouldn't need to offset the ray t_min by sysSceneEpsilon.
		optix::Ray ray = optix::make_Ray(prd.pos, prd.wi, 0, 0.001f, RT_DEFAULT_MAX);
		rtTrace(sysTopObject, ray, prd);

		radiance += throughput * prd.radiance;

		// Path termination by miss shader or sample() routines.
		// If terminate is true, f_over_pdf and pdf might be undefined.
		if ((prd.flags & FLAG_TERMINATE) || prd.pdf <= 0.0f || isNull(prd.f_over_pdf))
		{
			break;
		}

		// PERF f_over_pdf already contains the proper throughput adjustment for diffuse materials: f * (fabsf(optix::dot(prd.wi, state.normal)) / prd.pdf);
		throughput *= prd.f_over_pdf;

		// Russian Roulette path termination after a specified number of bounces in sysPathLengths.x would go here. See next examples.

		++depth; // Next path segment.
	}
}

// Entry point
RT_PROGRAM void raygeneration()
{
	// The launch index is the pixel coordinate.
	// Note that launchIndex = (0, 0) is the bottom left corner of the image,
	// which matches the origin in the OpenGL texture used to display the result.
	/*const float2 pixel = make_float2(theLaunchIndex);
	// Sample the ray in the center of the pixel.
	const float2 fragment = pixel + make_float2(0.5f);
	// The launch dimension (set with rtContextLaunch) is the full client window in this demo's setup.
	const float2 screen = make_float2(theLaunchDim);
	// Normalized device coordinates in range [-1, 1].
	const float2 ndc = (fragment / screen) * 2.0f - 1.0f;*/

	// const float3 origin = sysCameraPosition;
	// const float3 direction = optix::normalize(ndc.x * sysCameraU + ndc.y * sysCameraV + sysCameraW);

	uint idx = theLaunchIndex.x + theLaunchIndex.y * theLaunchDim.x;
	if (idx >= numSamples) return;

	const unsigned int tea_seed = (baseSeed << 16) | (px * sqrtPasses + py);
	unsigned seed = tea<2>(tea_seed, idx);

	float3 sample_norm = inNormalsBuffer[idx];
	float3 sample_face_norm = inFaceNormalsBuffer[idx];
	if (optix::dot(sample_norm, sample_face_norm) < 0)
	{
		sample_norm = sample_face_norm;
	}
	
	const float3 sample_pos = inPositionsBuffer[idx];
	optix::Onb onb(sample_norm);

	float3 ray_dir;
	float u0 = (static_cast<float>(px) + rnd(seed)) / static_cast<float>(sqrtPasses);
	float u1 = (static_cast<float>(py) + rnd(seed)) / static_cast<float>(sqrtPasses);
	uint j = 0U;
	do
	{
		optix::cosine_sample_hemisphere(u0, u1, ray_dir);

		onb.inverse_transform(ray_dir);
		++j;
		u0 = rnd(seed);
		u1 = rnd(seed);
	}
	while (j < 5 && optix::dot(ray_dir, sample_face_norm) <= 0.0f);

	if (optix::dot(ray_dir, sample_face_norm) <= 0.0f)
	{
		ray_dir = -ray_dir;
	}

	if (rayDirAlign > 0)
	{
		ray_dir = optix::normalize(ray_dir + sample_face_norm * rayDirAlign);
	}

	PerRayData prd;
	prd.seed = tea_custom<8>(tea_seed, idx);
	prd.pos = sample_pos + float3{sceneOffsetHorizontal, sceneOffsetVertical, sceneOffsetHorizontal} * (sample_norm + ray_dir) / 2.f;
	prd.wi = ray_dir;

	float3 radiance;
	integrator(prd, radiance); // In this case a unidirectional path tracer.
	sysOutputBuffer[make_uint2(idx, 0)] += radiance.y;
}
