#pragma once

#include <set>

#include "medial_mesh.h"
#include "rpd3d_wrapper.h"

enum TRACE_CODE {
  TRACE_UNKNOWN = -1,
  TRACE_TER_ENDPOINT = 1,
  TRACE_TER_LOOP = 2,
};

// call after RPD3D_Wrapper::cluster_sphere_type_using_samples_fids
void trace_medial_structure(const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
                            RPD3D_Wrapper& rpd3d, MedialMesh& mmesh,
                            bool is_debug);

// for VC or Voronoi based MAT
void trace_medial_structure_nonmanifold(MedialMesh& mmesh, bool is_debug);