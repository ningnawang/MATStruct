#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "medial_mesh.h"

class SphereTopK {
 public:
  SphereTopK() {};
  std::vector<std::vector<int>> sphere_top_sf_fid_indices;
  std::vector<std::vector<float>> sphere_top_sf_fid_dists;
  std::vector<std::vector<afloat3>> sphere_top_tri_tangents;
};

double eval_triangle_quality(const MedialMesh& mmesh);

double eval_seam_junction_quality_after_tracing(
    const MedialMesh& mmesh, std::vector<Vector3>& bad_sphere_centers,
    std::vector<Vector3>& all_intf_sphere_centers);

double eval_mat_intf_mspheres(
    const SurfaceMesh& sf_mesh, const MedialMesh& mmesh,
    std::vector<std::vector<int>>& sphere_top_tri_indices,
    std::vector<std::vector<float>>& sphere_top_tri_dists,
    std::vector<std::vector<afloat3>>& sphere_top_tri_tangents);