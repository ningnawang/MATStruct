#include "thinning_wrapper.h"

// priority queue is ordered by first element of the pair
void ThinningWrapper::sort_mat_face_importance_globally_mtype(
    MedialMesh& mat, std::set<Thinning::face_importance>& imp_queue,
    bool is_debug) {
  imp_queue.clear();
  for (const auto& one_tet : mat.tets) {
    int tet_id = one_tet.tid;
    for (const auto& fid : one_tet.faces_) {
      auto& face = mat.faces[fid];
      if (face.tets_.empty()) {
        // face.importance = DBL_MAX;
        printf("face of tet cannot have empty tets_!!!\n");
        assert(false);
        continue;
      }

      // make it very low importance
      if (!face.is_on_same_sheet)
        face.importance = mface_non_sheet_importance;
      else
        compute_face_importance(mat, face, false /*is_sort_randomly*/,
                                is_debug);

      // store to priority queue
      imp_queue.insert(std::pair<double, int>(face.importance, fid));
      if (is_debug)
        printf("[Prune] fid %d has importance: %f, pushed to priority queue \n",
               fid, face.importance);

    }  // for mat face
  }  // for mat tets
  if (is_debug)
    printf("[Prune] total %ld mat faces in priority queue \n",
           imp_queue.size());
}

void ThinningWrapper::load_all_mat_face_importance_globally_mtype(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh, RPD3D_Wrapper& rpd3d,
    MedialMesh& mat, bool is_debug) {
  bool is_sort_randomly = false;

  //   printf("start loading mat faces importances...\n");

  // for checking mface is_on_same_sheet
  // rpd3d.cluster_medges_same_sheet(tet_mesh, sf_mesh, mat, is_debug
  // /*is_debug*/);
  rpd3d.cluster_mfaces_same_sheet(tet_mesh, sf_mesh, mat, is_debug);

  //   printf("cluster_medges_same_sheet done\n");

  for (auto& face : mat.faces) {
    // make it very low importance
    if (!face.is_on_same_sheet)
      face.importance = mface_non_sheet_importance;
    else
      compute_face_importance(mat, face, is_sort_randomly, is_debug);

    if (is_debug)
      printf("[Prune] fid %d has importance: %f\n", face.fid, face.importance);
  }
  printf("Saved mat faces importances\n");
}

// we trace the sheet of medial mesh before thinning,
// and give mface not on the same sheet a very low importance,
// so that they will be pruned first
void ThinningWrapper::prune_mtype(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
    const std::vector<MedialSphere>& all_medial_spheres, RPD3D_Wrapper& rpd3d,
    MedialMesh& mat, double imp_thres, bool is_debug) {
  bool is_dup_cnt = false;
  bool is_import_given = false;
  bool is_sort_randomly = false;

  // for checking mface is_on_same_sheet
  // rpd3d.cluster_medges_same_sheet(tet_mesh, sf_mesh, mat, is_debug
  // /*is_debug*/);
  rpd3d.cluster_mfaces_same_sheet(tet_mesh, sf_mesh, mat, is_debug);

  printf(
      "start thinning with importance threshold: %f is_dup_cnt: %d, is_debug: "
      "%d... \n",
      imp_thres, is_dup_cnt, is_debug);

  // using set as priority queue
  // (importance, fid) starts with the smallest importance
  // tie-broken with smallest fid.
  std::set<Thinning::face_importance> imp_queue;  // in ascending order
  //   sort_mat_face_importance_globally(mat, imp_queue, is_import_given,
  //                                     is_sort_randomly, is_debug);
  sort_mat_face_importance_globally_mtype(mat, imp_queue, is_debug);

  if (is_dup_cnt) {
    mat.update_mmesh_dup_ef_map(is_debug);
    prune_tets_while_iteration(mat, imp_queue, is_dup_cnt, is_debug);
    // printf("done tet-face prune...\n");
    handle_dupcnt_face_edge_pair(mat, is_debug);
    // printf("done dupcnt_face_edge ...\n");
    prune_faces_while_iteration(all_medial_spheres, mat, imp_queue, imp_thres,
                                is_debug);
    prune_edges_while_iteration(all_medial_spheres, mat, is_debug);
    assert_mmesh(mat);
  } else {
    prune_tets_while_iteration(mat, imp_queue, is_dup_cnt, is_debug);
    prune_faces_while_iteration(all_medial_spheres, mat, imp_queue, imp_thres,
                                is_debug);
  }
}