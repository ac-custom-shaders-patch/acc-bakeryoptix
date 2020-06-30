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
rtBuffer<float, 2> sysOutputBuffer;     // RGBA32F

rtDeclareVariable(int, px, , );
rtDeclareVariable(int, py, , );
rtDeclareVariable(int, sqrtPasses, , );
rtDeclareVariable(uint, numSamples, , );
rtDeclareVariable(uint, baseSeed, , );
rtDeclareVariable(float, sceneOffset, , );

rtDeclareVariable(uint2, theLaunchDim, rtLaunchDim, );
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

// Entry point
RT_PROGRAM void raygeneration()
{
	PerRayData prd;

	prd.radiance = make_float3(0.0f);

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

	const float3 sample_norm = inNormalsBuffer[idx];
	const float3 sample_face_norm = inFaceNormalsBuffer[idx];
	const float3 sample_pos = inPositionsBuffer[idx];
	const float3 ray_origin = sample_pos + sceneOffset * sample_norm;
	optix::Onb onb(sample_norm);

	float3 ray_dir;
	float u0 = (static_cast<float>(px) + rnd(seed)) / static_cast<float>(sqrtPasses);
	float u1 = (static_cast<float>(py) + rnd(seed)) / static_cast<float>(sqrtPasses);
	int j = 0;
	do
	{
		optix::cosine_sample_hemisphere(u0, u1, ray_dir);

		onb.inverse_transform(ray_dir);
		++j;
		u0 = rnd(seed);
		u1 = rnd(seed);
	}
	while (j < 5 && optix::dot(ray_dir, sample_face_norm) <= 0.0f);


	// Shoot a ray from origin into direction (must always be normalized!) for ray type 0 and test the interval between 0.0f and RT_DEFAULT_MAX for intersections.
	optix::Ray ray = optix::make_Ray(ray_origin, ray_dir, 0, 0.0f, RT_DEFAULT_MAX);

	// Start the ray traversal at the scene's root node.
	// The ray becomes the variable with rtCurrentRay semantic in the other program domains.
	// The PerRayData becomes the variable with the semantic rtPayload in the other program domains,
	// which allows to exchange arbitrary data between the program domains.
	rtTrace(sysTopObject, ray, prd);
	sysOutputBuffer[make_uint2(idx, 0)] += prd.radiance.y;
}
