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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

// Context global variables provided by the renderer system.
rtDeclareVariable(rtObject, sysTopObject, , );

// Semantic variables.
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(float, theIntersectionDistance, rtIntersectionDistance, );
rtDeclareVariable(PerRayData, thePrd, rtPayload, );

// Attributes.
rtDeclareVariable(optix::float3, varGeoNormal, attribute GEO_NORMAL, );
//rtDeclareVariable(optix::float3, varTangent,   attribute TANGENT, );
rtDeclareVariable(optix::float3, varNormal, attribute NORMAL, );
rtDeclareVariable(optix::float3, varTexCoord, attribute TEXCOORD, );

// Material parameter definition.
rtDeclareVariable(float, parMaterialAlbedo, , ); // Per Material index into the sysMaterialParameters array.
rtDeclareVariable(float, parMaterialEmissive, , );  // Per Material index into the sysMaterialParameters array.
// 
// This closest hit program only uses the geometric normal and the shading normal attributes.
// OptiX will remove all code from the intersection programs for unused attributes automatically.

// Note that the matching between attribute outputs from the intersection program and 
// the inputs in the closesthit and anyhit programs is done with the type (here float3) and
// the user defined attribute semantic (e.g. here NORMAL). 
// The actual variable name doesn't need to match but it's recommended for clarity.

RT_FUNCTION void alignVector(float3 const& axis, float3& w)
{
	// Align w with axis.
	const float s = copysign(1.0f, axis.z);
	w.z *= s;
	const float3 h = make_float3(axis.x, axis.y, axis.z + s);
	const float k = optix::dot(w, h) / (1.0f + fabsf(axis.z));
	w = k * h - w;
}

RT_FUNCTION void unitSquareToCosineHemisphere(const float2 sample, float3 const& axis, float3& w, float& pdf)
{
	// Choose a point on the local hemisphere coordinates about +z.
	const float theta = 2.0f * M_PIf * sample.x;
	const float r = sqrtf(sample.y);
	w.x = r * cosf(theta);
	w.y = r * sinf(theta);
	w.z = 1.0f - w.x * w.x - w.y * w.y;
	w.z = (0.0f < w.z) ? sqrtf(w.z) : 0.0f;

	pdf = w.z * M_1_PIf;

	// Align with axis.
	alignVector(axis, w);
}

RT_PROGRAM void rayhit_emissive()
{
  thePrd.radiance = make_float3(parMaterialEmissive); // Constant white emission. No next event estimation (direct lighting).
  thePrd.flags   |= FLAG_TERMINATE;    // End of path.
}
