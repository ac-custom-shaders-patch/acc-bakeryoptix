#pragma once
#include <vector>
#include <string>

namespace bake { struct Mesh; }

void export_blockers_obj(const std::vector<bake::Mesh*>& meshes, const std::string& filename);
