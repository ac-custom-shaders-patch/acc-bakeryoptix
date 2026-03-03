#include <bake_ao_embree.h>
#include <optix_compat.h>
#include <cuda/random.h>
#include <embree3/include/rtcore.h>
#include <utils/cout_progress.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>

#pragma comment(lib, "embree3/embree3.lib")
#pragma comment(lib, "embree3/embree_sse42.lib")
#pragma comment(lib, "embree3/embree_avx.lib")
#pragma comment(lib, "embree3/embree_avx2.lib")
#pragma comment(lib, "embree3/lexers.lib")
#pragma comment(lib, "embree3/math.lib")
#pragma comment(lib, "embree3/simd.lib")
#pragma comment(lib, "embree3/sys.lib")
#pragma comment(lib, "embree3/tasking.lib")

using namespace bake;

// ============================================================
// Per-geometry material info (stored alongside Embree scene)
// ============================================================

enum class MaterialType : uint8_t
{
	opaque,
	alpha_test,
	foliage,        // alpha test + hash pass-through
	proc_foliage,   // procedural hash pass-through only
	emissive
};

struct GeometryInfo
{
	const Mesh* mesh;
	MaterialType mat_type;
	float alpha_ref;
	float pass_chance;
	float albedo;
	float emissive;
};

// ============================================================
// UV interpolation using barycentric coordinates
// ============================================================

static Vec2 interpolate_uv(const Mesh* mesh, unsigned prim_id, float u, float v)
{
	const auto& tri = mesh->triangles[prim_id];
	const float w = 1.0f - u - v;
	const auto& uv0 = mesh->vertices[tri.a].tex;
	const auto& uv1 = mesh->vertices[tri.b].tex;
	const auto& uv2 = mesh->vertices[tri.c].tex;
	return {w * uv0.x + u * uv1.x + v * uv2.x,
	        w * uv0.y + u * uv1.y + v * uv2.y};
}

// ============================================================
// Procedural hash (matches the OptiX PTX implementation)
// ============================================================

static float uv_hash(float tex_x, float tex_y)
{
	double h = double(tex_x) * 693.453 + double(tex_y) * 246.827;
	return float(h - floor(h));
}

// ============================================================
// Texture alpha lookup (returns 0–255 alpha value)
// ============================================================

static float texture_alpha(const Mesh* mesh, float tex_x, float tex_y)
{
	if (!mesh->material || !mesh->material->texture || !mesh->material->texture->data)
		return 255.f;
	const auto& tex = *mesh->material->texture;
	// Wrap UVs to [0,1)
	float fu = tex_x - floorf(tex_x);
	float fv = tex_y - floorf(tex_y);
	int ix = int(fu * tex.width) % tex.width;
	int iy = int(fv * tex.height) % tex.height;
	if (ix < 0) ix += tex.width;
	if (iy < 0) iy += tex.height;
	// dds_loader::data is std::unique_ptr<std::string> containing raw RGBA pixels
	const auto pixel = reinterpret_cast<const uint8_t*>(tex.data->data()) + (iy * tex.width + ix) * 4;
	return float(pixel[3]); // alpha channel
}

// ============================================================
// Embree intersection filter — reject alpha-transparent hits
// during BVH traversal (avoids expensive retrace loop)
// ============================================================

static void alpha_filter(const RTCFilterFunctionNArguments* args)
{
	// For rtcIntersect1/rtcOccluded1, valid[0] == -1 means active
	if (args->valid[0] != -1) return;

	const auto* info = static_cast<const GeometryInfo*>(args->geometryUserPtr);

	// Access hit data (N==1 for single-ray calls)
	const unsigned prim_id = RTCHitN_primID(args->hit, args->N, 0);
	const float u = RTCHitN_u(args->hit, args->N, 0);
	const float v = RTCHitN_v(args->hit, args->N, 0);

	bool accept;
	switch (info->mat_type)
	{
	case MaterialType::alpha_test:
	{
		const auto uv = interpolate_uv(info->mesh, prim_id, u, v);
		accept = texture_alpha(info->mesh, uv.x, uv.y) > info->alpha_ref;
		break;
	}
	case MaterialType::foliage:
	{
		const auto uv = interpolate_uv(info->mesh, prim_id, u, v);
		const float alpha = texture_alpha(info->mesh, uv.x, uv.y);
		if (alpha > info->alpha_ref) { accept = true; break; }
		accept = uv_hash(uv.x, uv.y) >= info->pass_chance;
		break;
	}
	case MaterialType::proc_foliage:
	{
		const auto uv = interpolate_uv(info->mesh, prim_id, u, v);
		accept = uv_hash(uv.x, uv.y) >= 0.3f;
		break;
	}
	default:
		return; // opaque, emissive — always accept
	}

	if (!accept)
		args->valid[0] = 0; // reject this hit — traversal continues automatically
}

// ============================================================
// Transform a position by a 4x4 row-major matrix (M * v)
// ============================================================

static float3 xform_point(const float* m, const Vec3& p)
{
	return {
		p.x * m[0] + p.y * m[1] + p.z * m[2]  + m[3],
		p.x * m[4] + p.y * m[5] + p.z * m[6]  + m[7],
		p.x * m[8] + p.y * m[9] + p.z * m[10] + m[11]
	};
}

static float3 xform_normal(const float* m, const Vec3& n)
{
	// For normals, use the upper-left 3x3 (assumes no non-uniform scale)
	float3 r = {
		n.x * m[0] + n.y * m[1] + n.z * m[2],
		n.x * m[4] + n.y * m[5] + n.z * m[6],
		n.x * m[8] + n.y * m[9] + n.z * m[10]
	};
	float len = length(r);
	return len > 0.f ? r * (1.f / len) : r;
}

// ============================================================
// Interpolate hit normal from mesh vertex normals
// ============================================================

static float3 interpolate_normal(const Mesh* mesh, const float* matrix, unsigned prim_id, float u, float v)
{
	if (mesh->normals.empty())
		return make_float3(0, 1, 0);
	const auto& tri = mesh->triangles[prim_id];
	const float w = 1.0f - u - v;
	Vec3 n = {
		w * mesh->normals[tri.a].x + u * mesh->normals[tri.b].x + v * mesh->normals[tri.c].x,
		w * mesh->normals[tri.a].y + u * mesh->normals[tri.b].y + v * mesh->normals[tri.c].y,
		w * mesh->normals[tri.a].z + u * mesh->normals[tri.b].z + v * mesh->normals[tri.c].z
	};
	return xform_normal(matrix, n);
}

// ============================================================
// Main AO computation with Embree
// ============================================================

void bake::ao_embree(const std::vector<Mesh*>& blockers,
	const AOSamples& ao_samples, const int rays_per_sample, const float albedo, const uint32_t bounce_counts,
	const float scene_offset_scale_horizontal, const float scene_offset_scale_vertical, const float trees_light_pass_chance,
	const bool debug_mode, float* ao_values)
{
	// --- Create Embree device and scene ---
	RTCDevice device = rtcNewDevice("threads=0"); // use all HW threads for BVH build
	if (!device)
	{
		std::cerr << "Embree: Failed to create device (error " << rtcGetDeviceError(nullptr) << ")\n";
		return;
	}
	RTCScene scene = rtcNewScene(device);

	// PERF: Don't use RTC_SCENE_FLAG_ROBUST — default is faster.
	//       Use MEDIUM build quality (default) — HIGH is slower to build for marginal trace gain.
	rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM);

	// --- Build geometry for each blocker mesh ---
	// Pre-allocate so pointers remain stable (used by filter function via user data).
	std::vector<GeometryInfo> geom_infos;
	geom_infos.resize(blockers.size());
	unsigned info_count = 0;
	bool has_emissive = false;

	for (const auto* mesh : blockers)
	{
		if (mesh->triangles.empty() || mesh->vertices.empty()) continue;

		// Populate material info FIRST (needed before rtcCommitGeometry for user data)
		auto& info = geom_infos[info_count];
		info.mesh = mesh;
		info.albedo = albedo;
		info.emissive = 0.f;
		info.alpha_ref = 0.5f * 255.f;
		info.pass_chance = trees_light_pass_chance;

		if (mesh->material && mesh->material->texture && mesh->material->texture->data)
		{
			const bool is_foliage = mesh->material->shader.find("ksTree") == 0;
			info.mat_type = is_foliage ? MaterialType::foliage : MaterialType::alpha_test;
			const auto* var = mesh->material->get_var_or_null("ksAlphaRef");
			info.alpha_ref = (var ? var->v1 : 0.5f) * 255.f;
		}
		else if (mesh->material)
		{
			info.mat_type = MaterialType::opaque;
			info.albedo = albedo;
		}
		else
		{
			if (mesh->name == "blocker")
			{
				info.mat_type = MaterialType::opaque;
				info.albedo = 0.2f;
			}
			else if (mesh->name == "trees")
			{
				info.mat_type = MaterialType::proc_foliage;
				info.albedo = 0.2f;
			}
			else if (mesh->name == "emissive")
			{
				info.mat_type = MaterialType::emissive;
				info.albedo = 1.f;
				info.emissive = mesh->lod_out;
			}
			else
			{
				info.mat_type = MaterialType::opaque;
				info.albedo = albedo;
			}
		}

		if (info.mat_type == MaterialType::emissive)
			has_emissive = true;

		// --- Create Embree geometry ---
		RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

		// Vertices — pre-transformed to world space, 16-byte stride for SIMD-friendly alignment
		auto* vb = static_cast<float*>(rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
			RTC_FORMAT_FLOAT3, 4 * sizeof(float), mesh->vertices.size()));
		const float* mat = mesh->matrix.data();
		for (size_t i = 0; i < mesh->vertices.size(); ++i)
		{
			float3 wp = xform_point(mat, mesh->vertices[i].pos);
			vb[i * 4 + 0] = wp.x;
			vb[i * 4 + 1] = wp.y;
			vb[i * 4 + 2] = wp.z;
			// vb[i * 4 + 3] is padding (unused)
		}

		// Indices
		auto* ib = static_cast<unsigned*>(rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
			RTC_FORMAT_UINT3, 3 * sizeof(unsigned), mesh->triangles.size()));
		for (size_t i = 0; i < mesh->triangles.size(); ++i)
		{
			ib[i * 3 + 0] = mesh->triangles[i].a;
			ib[i * 3 + 1] = mesh->triangles[i].b;
			ib[i * 3 + 2] = mesh->triangles[i].c;
		}

		// PERF: Set user data and filter function BEFORE commit.
		// The filter function handles alpha testing inside BVH traversal,
		// eliminating the need for the manual retrace loop.
		rtcSetGeometryUserData(geom, &info);

		const bool needs_filter = info.mat_type == MaterialType::alpha_test
		                       || info.mat_type == MaterialType::foliage
		                       || info.mat_type == MaterialType::proc_foliage;
		if (needs_filter)
		{
			rtcSetGeometryIntersectFilterFunction(geom, alpha_filter);
			rtcSetGeometryOccludedFilterFunction(geom, alpha_filter);
		}

		rtcCommitGeometry(geom);
		unsigned geom_id = rtcAttachGeometry(scene, geom);
		rtcReleaseGeometry(geom);

		// Verify sequential ID assumption
		assert(geom_id == info_count);
		(void)geom_id;

		++info_count;
	}
	geom_infos.resize(info_count);

	rtcCommitScene(scene);

	// --- Trace rays ---
	const int sqrt_passes = static_cast<int>(lroundf(sqrtf(static_cast<float>(rays_per_sample))));
	const size_t num_samples = ao_samples.num_samples;
	const float ray_tmin = 1e-4f;
	const float ray_tmax = 1e20f;

	// Can use cheaper occlusion test when we don't need hit details
	const bool use_occlusion_shortcut = (bounce_counts == 0) && !has_emissive;

	memset(ao_values, 0, sizeof(float) * num_samples);

	const size_t total_passes = size_t(sqrt_passes) * size_t(sqrt_passes);
	cout_progress progress{total_passes};

	// Pre-cast sample data pointers (invariant across all passes)
	const auto* sample_norms = reinterpret_cast<const float3*>(ao_samples.sample_normals);
	const auto* sample_face_norms = reinterpret_cast<const float3*>(ao_samples.sample_face_normals);
	const auto* sample_pos = reinterpret_cast<const float3*>(ao_samples.sample_positions);

	for (int px = 0; px < sqrt_passes; ++px)
	{
		for (int py = 0; py < sqrt_passes; ++py)
		{
			const unsigned pass_seed = unsigned(px * sqrt_passes + py);

			#pragma omp parallel for schedule(dynamic, 512)
			for (ptrdiff_t si = 0; si < static_cast<ptrdiff_t>(num_samples); ++si)
			{
				const float3 norm = sample_norms[si];
				const float3 face_norm = sample_face_norms[si];
				const float3 pos = sample_pos[si];

				// Seed RNG (matches OptiX tea<2> seeding)
				unsigned seed = tea<2>(unsigned(si), pass_seed);

				// Generate cosine-weighted hemisphere direction
				Onb onb(norm);
				float3 ray_dir;
				float u0 = (float(px) + rnd(seed)) / float(sqrt_passes);
				float u1 = (float(py) + rnd(seed)) / float(sqrt_passes);
				int retries = 0;
				do
				{
					cosine_sample_hemisphere(u0, u1, ray_dir);
					onb.inverse_transform(ray_dir);
					++retries;
					u0 = rnd(seed);
					u1 = rnd(seed);
				} while (retries < 5 && dot(ray_dir, face_norm) <= 0.0f);

				// Offset origin
				float3 origin = {
					pos.x + scene_offset_scale_horizontal * ray_dir.x * 0.5f,
					pos.y + scene_offset_scale_vertical * ray_dir.y * 0.5f,
					pos.z + scene_offset_scale_horizontal * ray_dir.z * 0.5f
				};

				// --- Fast path: simple occlusion test (no bounces, no emissive) ---
				if (use_occlusion_shortcut)
				{
					RTCRay ray;
					ray.org_x = origin.x;
					ray.org_y = origin.y;
					ray.org_z = origin.z;
					ray.dir_x = ray_dir.x;
					ray.dir_y = ray_dir.y;
					ray.dir_z = ray_dir.z;
					ray.tnear = ray_tmin;
					ray.tfar  = ray_tmax;
					ray.mask  = 0xFFFFFFFF;
					ray.flags = 0;

					RTCIntersectContext ctx;
					rtcInitIntersectContext(&ctx);
					rtcOccluded1(scene, &ctx, &ray);

					// If tfar == -inf, ray was occluded (hit something solid)
					if (ray.tfar >= 0.f)
						ao_values[si] += 1.0f; // miss — sky visible
					continue;
				}

				// --- Full path tracing with bounces ---
				float throughput = 1.0f;
				float ao_accum = 0.f;

				RTCIntersectContext ctx;
				rtcInitIntersectContext(&ctx);

				for (uint32_t bounce = 0; bounce <= bounce_counts; ++bounce)
				{
					RTCRayHit rayhit;
					rayhit.ray.org_x = origin.x;
					rayhit.ray.org_y = origin.y;
					rayhit.ray.org_z = origin.z;
					rayhit.ray.dir_x = ray_dir.x;
					rayhit.ray.dir_y = ray_dir.y;
					rayhit.ray.dir_z = ray_dir.z;
					rayhit.ray.tnear = ray_tmin;
					rayhit.ray.tfar  = ray_tmax;
					rayhit.ray.mask  = 0xFFFFFFFF;
					rayhit.ray.flags = 0;
					rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
					rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

					// PERF: Single call — filter functions handle alpha testing
					// inside BVH traversal (no manual retrace loop needed).
					rtcIntersect1(scene, &ctx, &rayhit);

					if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
					{
						// Miss — sky is visible
						ao_accum += throughput * 1.0f;
						break;
					}

					// We have a solid hit (filter already rejected transparent surfaces)
					const auto& hit_info = geom_infos[rayhit.hit.geomID];

					if (hit_info.mat_type == MaterialType::emissive)
					{
						ao_accum += throughput * hit_info.emissive;
						break;
					}

					if (bounce >= bounce_counts || hit_info.albedo <= 0.f)
						break; // occluded, no more bounces

					// --- Bounce ---
					throughput *= hit_info.albedo;

					float3 hit_pos = {
						rayhit.ray.org_x + rayhit.ray.tfar * rayhit.ray.dir_x,
						rayhit.ray.org_y + rayhit.ray.tfar * rayhit.ray.dir_y,
						rayhit.ray.org_z + rayhit.ray.tfar * rayhit.ray.dir_z
					};

					float3 hit_gnorm = normalize(make_float3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));
					if (dot(hit_gnorm, ray_dir) > 0.f)
						hit_gnorm = -hit_gnorm;

					float3 hit_norm = interpolate_normal(hit_info.mesh, hit_info.mesh->matrix.data(),
						rayhit.hit.primID, rayhit.hit.u, rayhit.hit.v);
					if (dot(hit_norm, ray_dir) > 0.f)
						hit_norm = -hit_norm;

					Onb bounce_onb(hit_norm);
					u0 = rnd(seed);
					u1 = rnd(seed);
					cosine_sample_hemisphere(u0, u1, ray_dir);
					bounce_onb.inverse_transform(ray_dir);

					if (dot(ray_dir, hit_gnorm) <= 0.f)
						break; // absorbed

					origin = {
						hit_pos.x + scene_offset_scale_horizontal * ray_dir.x * 0.5f,
						hit_pos.y + scene_offset_scale_vertical * ray_dir.y * 0.5f,
						hit_pos.z + scene_offset_scale_horizontal * ray_dir.z * 0.5f
					};
				}

				ao_values[si] += ao_accum;
			}

			progress.report();
		}
	}

	// Normalize
	const float inv_total = 1.0f / float(rays_per_sample);
	for (size_t i = 0; i < num_samples; ++i)
	{
		ao_values[i] *= inv_total;
	}

	// Cleanup
	rtcReleaseScene(scene);
	rtcReleaseDevice(device);
}
