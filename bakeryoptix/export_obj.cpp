#include "export_obj.h"

#include <fstream>
#include <iostream>
#include <bake_api.h>
#include <optix_compat.h>

using bake::Vec3;

static float3 xform_point(const float* m, const Vec3& p)
{
	return {
		p.x * m[0] + p.y * m[1] + p.z * m[2]  + m[3],
		p.x * m[4] + p.y * m[5] + p.z * m[6]  + m[7],
		p.x * m[8] + p.y * m[9] + p.z * m[10] + m[11]
	};
}

void export_blockers_obj(const std::vector<bake::Mesh*>& meshes, const std::string& filename)
{
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Failed to open " << filename << " for writing\n";
		return;
	}

	file << "# Exported blocker geometry\n";

	uint32_t vertex_offset = 0;
	for (size_t mi = 0; mi < meshes.size(); ++mi)
	{
		const auto* mesh = meshes[mi];
		file << "o " << (mesh->name.empty() ? "mesh_" + std::to_string(mi) : mesh->name + "_" + std::to_string(mi)) << "\n";

		const float* mat = mesh->matrix.data();
		for (size_t i = 0; i < mesh->vertices.size(); ++i)
		{
			float3 wp = xform_point(mat, mesh->vertices[i].pos);
			file << "v " << wp.x << " " << wp.y << " " << wp.z << "\n";
		}

		for (size_t i = 0; i < mesh->triangles.size(); ++i)
		{
			const auto& tri = mesh->triangles[i];
			file << "f " << (tri.a + 1 + vertex_offset)
				 << " " << (tri.b + 1 + vertex_offset)
				 << " " << (tri.c + 1 + vertex_offset) << "\n";
		}

		vertex_offset += static_cast<uint32_t>(mesh->vertices.size());
	}

	std::cout << "\tExported " << meshes.size() << " blocker mesh(es) to " << filename << "\n";
}
