#pragma once

#include "fix_intf.h"
#include "medial_struct.h"
#include "rpd3d_wrapper.h"

// check all medial edges and add internal feature spheres
int pre_insert_intf_spheres(const int num_itr_global,
                            const SurfaceMesh& sf_mesh, const TetMesh& tet_mesh,
                            const MedialMesh& mmesh,
                            std::vector<MedialSphere>& all_medial_spheres,
                            bool is_debug);

int pre_insert_intf_spheres_for_SE(
    const int num_itr_global, const SurfaceMesh& sf_mesh,
    const TetMesh& tet_mesh, const MedialMesh& mmesh,
    std::vector<MedialSphere>& all_medial_spheres, bool is_debug);