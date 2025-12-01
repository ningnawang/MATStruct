#include "fix_intf_wrapper.h"

bool add_new_intf_sphere(const int num_itr_global, const SurfaceMesh& sf_mesh,
                         const TetMesh& tet_mesh, const MedialEdge& medge_cand,
                         std::vector<MedialSphere>& all_medial_spheres,
                         bool is_debug) {
  MedialSphere& msphere1 = all_medial_spheres.at(medge_cand.vertices_[0]);
  MedialSphere& msphere2 = all_medial_spheres.at(medge_cand.vertices_[1]);

  // if (msphere1->id == 125 || msphere2->id == 125)
  //   is_debug = true;
  // else
  // is_debug = false;

  if (is_debug)
    printf("[FIX_INTF] checking internal sphere for medge (%d,%d)\n",
           msphere1.id, msphere2.id);

  ////////////////////////////
  // Pre-step: checking common/diff tangent info
  const std::vector<TangentPlane>& common_tan_pls = medge_cand.common_tan_pls;
  const std::vector<TangentPlane>& A_non_common_tan_pls =
      medge_cand.v1_non_common_tan_pls;
  const std::vector<TangentPlane>& B_non_common_tan_pls =
      medge_cand.v2_non_common_tan_pls;

  if (common_tan_pls.empty() && A_non_common_tan_pls.empty() &&
      B_non_common_tan_pls.empty())
    return true;

  // no need to add a new sphere case 1
  if (B_non_common_tan_pls.empty() || A_non_common_tan_pls.empty()) {
    if (is_debug)
      printf(
          "[CREATE_more] medial edge (%d,%d) no need to add new sphere, "
          "empty\n",
          msphere1.id, msphere2.id);
    return true;
  }

  // no need to add a new sphere case 2
  int total_tan = common_tan_pls.size() + A_non_common_tan_pls.size() +
                  B_non_common_tan_pls.size();
  if (total_tan < 3) {
    if (is_debug)
      printf(
          "[CREATE_more] medial edge (%d,%d) no need to add new sphere, , "
          "common_tan_pls size: %zu, A_non_common_tan_pls: %zu, "
          "B_non_common_tan_pls: %zu \n",
          msphere1.id, msphere2.id, common_tan_pls.size(),
          A_non_common_tan_pls.size(), B_non_common_tan_pls.size());
    return true;
  }

  // create new sphere
  bool is_good = false;
  Vector3 mid_center = (msphere1.center + msphere2.center) / 2;
  MedialSphere new_sphere(all_medial_spheres.size(), mid_center, INIT_RADIUS,
                          SphereType::T_3_MORE, num_itr_global);
  if (is_debug)
    printf("[FIX_INTF] trying to add new_sphere %d\n", new_sphere.id);

  ////////////////////////////
  // Step1: 1. both not on internal feature
  //        2. both on external feature, but not the same SL (checked
  //        ahead) aggregate tangent info then try to add T_3_MORE/T_N_c sphere
  if (!msphere1.is_on_intf() && !msphere2.is_on_intf() ||
      msphere1.is_on_extf() && msphere2.is_on_extf()) {
    if (is_debug) printf("[FIX_INTF] step 1: aggregate all tangent info \n");

    // add tangent planes, including adjacent planes of tan_cc_line
    if (is_debug) {
      printf("common_tan_pls: ----------------------------\n");
      for (const auto& common : common_tan_pls) {
        new_sphere.tan_planes.push_back(common);
        common.print_info();
      }
    }
    for (const auto& a_diff : A_non_common_tan_pls)
      new_sphere.tan_planes.push_back(a_diff);
    for (const auto& b_diff : B_non_common_tan_pls)
      new_sphere.tan_planes.push_back(b_diff);
    // add tangent concave lines, since iterate_sphere() may not detect
    for (const auto& a_tan_cc_line : msphere1.tan_cc_lines)
      new_sphere.new_cc_line_no_dup(a_tan_cc_line);
    for (const auto& b_tan_cc_line : msphere2.tan_cc_lines)
      new_sphere.new_cc_line_no_dup(b_tan_cc_line);
    // purge using tan_cc_lines
    new_sphere.purge_and_delete_tan_planes();

    // try to add T_3_MORE/T_N_c sphere after aggregation
    is_good =
        iterate_sphere(sf_mesh, sf_mesh.aabb_wrapper, sf_mesh.fe_sf_fs_pairs,
                       tet_mesh.feature_edges, new_sphere, false /*is_debug*/);

    if (is_debug)
      printf("[FIX_INTF] step 1 is_good: %d, with tan_planes: %zu\n", is_good,
             new_sphere.tan_planes.size());
  }

  if (!is_good) {
    if (is_debug)
      printf(
          "[FIX_INTF] failed to add internal feature sphere for medge "
          "(%d,%d)\n",
          msphere1.id, msphere2.id);
    return false;
  }

  // iterate_sphere may merge tangent planes as well
  if (new_sphere.tan_planes.size() + new_sphere.tan_cc_lines.size() < 3) {
    if (is_debug)
      printf(
          "[FIX_INTF] medial edge (%d,%d) no need to add sphere after "
          "iteration, tan_plane size: %ld, tan_cc_line size: %ld\n",
          msphere1.id, msphere2.id, new_sphere.tan_planes.size(),
          new_sphere.tan_cc_lines.size());
    return true;
  }

  // internal feature condition 3
  // not add sphere if newly added sphere is already covered by msphere1
  std::vector<TangentPlane> tmp_common_tan_pls, tmp_A_non_common_tan_pls,
      tmp_B_non_common_tan_pls;
  A_B_spheres_common_diff_tangent_info_from_surface(
      sf_mesh, msphere1, new_sphere, tmp_common_tan_pls,
      tmp_A_non_common_tan_pls, tmp_B_non_common_tan_pls);
  if (tmp_B_non_common_tan_pls.empty()) {
    if (is_debug)
      printf(
          "[FIX_INTF] medial edge (%d,%d) no need to add new sphere %d, "
          " same tangent info as old spheres \n",
          msphere1.id, msphere2.id, new_sphere.id);
    return true;
  }

  if (!add_new_sphere_validate(all_medial_spheres, new_sphere)) return false;
  return true;
}

// check all medial edges and add internal feature spheres
int pre_insert_intf_spheres(const int num_itr_global,
                            const SurfaceMesh& sf_mesh, const TetMesh& tet_mesh,
                            const MedialMesh& mmesh,
                            std::vector<MedialSphere>& all_medial_spheres,
                            bool is_debug) {
  if (is_debug) printf("calling pre_insert_intf_spheres...\n");
  int num_added = all_medial_spheres.size();
  for (const auto& medge_cand : mmesh.edges) {
    // skip checking medges if on the same sheet
    if (medge_cand.is_on_same_sheet) {
      if (is_debug)
        printf("[FIX_INTF] medial edge %d (%d,%d) on the same sheet, pass!\n",
               medge_cand.eid, medge_cand.vertices_[0],
               medge_cand.vertices_[1]);
      continue;
    }
    MedialSphere& msphere1 = all_medial_spheres.at(medge_cand.vertices_[0]);
    MedialSphere& msphere2 = all_medial_spheres.at(medge_cand.vertices_[1]);
    bool is_good_or_add = false;
    // not handling corners so far
    if (msphere1.is_on_corner() || msphere2.is_on_corner()) continue;

    // if both on intf, then skip
    if (msphere1.is_on_intf() && msphere2.is_on_intf()) {
      continue;
    }
    // if any not on sheet, then skip
    if (!msphere1.is_on_sheet() || msphere2.is_on_sheet()) continue;

    // if both spheres on same sharp line, not corners, skip
    if (msphere1.is_on_se() && msphere2.is_on_se() &&
        is_two_mspheres_on_same_se(msphere1, msphere2)) {
      continue;
    }

    // insert new sphere
    is_good_or_add =
        add_new_intf_sphere(num_itr_global, sf_mesh, tet_mesh, medge_cand,
                            all_medial_spheres, is_debug);

    if (is_debug)
      printf("[FIX_INTF] is_good_or_add: %d for edge (%d,%d) \n",
             is_good_or_add, msphere1.id, msphere2.id);
  }
  num_added = all_medial_spheres.size() - num_added;

  if (is_debug)
    printf("[FIX_INTF] INTF_PRE check added %d spheres\n", num_added);
  return num_added;
}

// check all medial edges and add internal feature spheres
int pre_insert_intf_spheres_for_SE(
    const int num_itr_global, const SurfaceMesh& sf_mesh,
    const TetMesh& tet_mesh, const MedialMesh& mmesh,
    std::vector<MedialSphere>& all_medial_spheres, bool is_debug) {
  if (is_debug) printf("FIX_INTF_SE] calling pre_insert_intf_spheres...\n");
  int num_added = all_medial_spheres.size();
  for (const auto& medge_cand : mmesh.edges) {
    MedialSphere& msphere1 = all_medial_spheres.at(medge_cand.vertices_[0]);
    MedialSphere& msphere2 = all_medial_spheres.at(medge_cand.vertices_[1]);
    // only care about two spheres on different SE
    if (!msphere1.is_on_extf() || !msphere2.is_on_extf()) continue;
    if (is_two_mspheres_on_same_sl_including_corners(msphere1, msphere2))
      continue;

    // create new sphere
    bool is_good = false;
    Vector3 mid_center = (msphere1.center + msphere2.center) / 2;
    MedialSphere new_sphere(all_medial_spheres.size(), mid_center, INIT_RADIUS,
                            SphereType::T_3_MORE, num_itr_global);
    if (is_debug)
      printf("[FIX_INTF_SE] trying to add new_sphere %d\n", new_sphere.id);
    new_sphere.tan_planes.insert(new_sphere.tan_planes.end(),
                                 msphere1.tan_planes.begin(),
                                 msphere1.tan_planes.end());
    new_sphere.tan_planes.insert(new_sphere.tan_planes.end(),
                                 msphere2.tan_planes.begin(),
                                 msphere2.tan_planes.end());
    // try to add T_3_MORE/T_N_c sphere after aggregation
    is_good =
        iterate_sphere(sf_mesh, sf_mesh.aabb_wrapper, sf_mesh.fe_sf_fs_pairs,
                       tet_mesh.feature_edges, new_sphere, false /*is_debug*/);
    if (!is_good) {
      if (is_debug)
        printf(
            "[FIX_INTF_SE] failed to add internal feature sphere for medge "
            "(%d,%d)\n",
            msphere1.id, msphere2.id);
      continue;
    }

    if (is_debug)
      printf("[FIX_INTF_SE] is_good: %d, with tan_planes: %zu\n", is_good,
             new_sphere.tan_planes.size());
    add_new_sphere_validate(all_medial_spheres, new_sphere, false, is_debug);
  }  // for medges
  num_added = all_medial_spheres.size() - num_added;
  // if (is_debug)
  printf("[FIX_INTF_SE] INTF_PRE check added %d spheres\n", num_added);
  return num_added;
}