#include "rpd3d_wrapper.h"
#include "thinning.h"

#pragma once

class ThinningWrapper : public Thinning {
 public:
  // we trace the sheet of medial mesh before thinning,
  // and give mface not on the same sheet a very low importance,
  // so that they will be pruned first
  static void prune_mtype(const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
                          const std::vector<MedialSphere>& all_medial_spheres,
                          RPD3D_Wrapper& rpd3d, MedialMesh& mat,
                          double imp_thres = -1, bool is_debug = false);

  static void load_all_mat_face_importance_globally_mtype(const TetMesh& tet_mesh, 
      const SurfaceMesh& sf_mesh, RPD3D_Wrapper& rpd3d, MedialMesh& mat,
      bool is_debug = false);

 private:
  static constexpr double mface_non_sheet_importance = -1.f;
  static void sort_mat_face_importance_globally_mtype(
      MedialMesh& mat, std::set<Thinning::face_importance>& imp_queue,
      bool is_debug);
};