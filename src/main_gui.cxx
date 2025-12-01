
#include "main_gui.h"

#include <polyscope/curve_network.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/volume_mesh.h>

#include "eval/mesh_eval.h"
#include "fix_extf.h"
#include "fix_geo.h"
#include "fix_geo_error.h"
#include "fix_intf.h"
#include "fix_intf_wrapper.h"
#include "fix_topo.h"
#include "io_cuda.h"
#include "io_wrapper.h"
#include "medial_struct.h"
#include "opt/opt_rpd.h"
#include "particle.h"
#include "rpd_api.h"
#include "thinning_wrapper.h"
#include "voronoi.h"

MainGuiWindow* MainGuiWindow::instance_ = nullptr;
MainGuiWindow::MainGuiWindow() {
  if (instance_ != nullptr) {
    printf("ERROR: GuiWindow instance is not nullptr!!");
    exit(1);
  }
  instance_ = this;
}

MainGuiWindow::~MainGuiWindow() {
  // we delete here since we allocate memory
  // (called new) for each of these variables
  delete params;
  delete tet_mesh;
  delete sf_mesh;
  delete all_medial_spheres;
  delete mmesh;
  delete spheres_to_fix;
  delete rt;
  delete rpd3d;
  delete opt_rpd;

  this->instance_ = nullptr;
}

void MainGuiWindow::set_params(Parameter& _params) { params = &_params; }
void MainGuiWindow::set_tet_mesh(TetMesh& _tet_mesh) { tet_mesh = &_tet_mesh; }
void MainGuiWindow::set_sf_mesh(const SurfaceMesh& _sf_mesh) {
  sf_mesh = &_sf_mesh;
}
void MainGuiWindow::set_all_medial_spheres(
    std::vector<MedialSphere>& _all_medial_spheres) {
  all_medial_spheres = &_all_medial_spheres;
}
void MainGuiWindow::set_rt(RegularTriangulationNN& _rt) { rt = &_rt; }
void MainGuiWindow::set_medial_mesh(MedialMesh& _mmesh) { mmesh = &_mmesh; }
void MainGuiWindow::set_rpd3d(RPD3D_Wrapper* _rpd3d) {
  rpd3d = _rpd3d;
  // rpd3d->init(this->tet_mesh, this->sf_mesh, this->params);
  // rpd3d->set_mspheres(this->all_medial_spheres);
}
void MainGuiWindow::set_opt_rpd(OPT_RPD* _opt_rpd) { opt_rpd = _opt_rpd; }
void MainGuiWindow::set_sphere_topk(SphereTopK& _sphere_topk) {
  sphere_topk = &_sphere_topk;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Iterations functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////

int MainGuiWindow::run_topo_fix_itr(
    const SurfaceMesh& sf_mesh, const TetMesh& tet_mesh,
    std::vector<MedialSphere>& all_medial_spheres,
    std::vector<ConvexCellHost>& cells_to_show, bool is_debug) {
  printf("--------------------Topo Check %d--------------------\n",
         num_topo_itr);

  std::set<int> spheres_to_fix;
  check_cc_and_euler(cells_to_show, all_medial_spheres, spheres_to_fix,
                     is_debug /*is_debug*/);

  if (spheres_to_fix.empty()) {
    polyscope::info("Topology check has passed!");
    return 0;
  }

  // int old_size = all_medial_spheres.size();
  int num_sphere_change = fix_topo_by_adding_new_sphere(
      instance_->num_itr_global, num_topo_itr, cells_to_show, sf_mesh, tet_mesh,
      spheres_to_fix, all_medial_spheres, is_debug /*is_debug*/);
  // int num_added = all_medial_spheres.size() - old_size;

  polyscope::info("Topo check change " + std::to_string(num_sphere_change) +
                  " spheres");
  num_topo_itr++;
  return num_sphere_change;
}

void MainGuiWindow::run_mmesh_generator(
    const SurfaceMesh& sf_mesh, MedialMesh& mmesh,
    std::vector<MedialSphere>& all_medial_spheres,
    bool is_compute_common_diff_sfids, bool is_trace_mstruc, bool is_debug) {
  printf("--------------------Cal MedialMesh--------------------\n");
  instance_->rpd3d->update_spheres_power_cells(false /*is_compute_se_sfids*/);
  mmesh.clear();
  mmesh.genearte_medial_spheres(all_medial_spheres);
  mmesh.generate_medial_edges(sf_mesh, is_compute_common_diff_sfids);
  mmesh.generate_medial_faces();
  mmesh.compute_faces_st_meta(sf_mesh.aabb_wrapper);  // for fix_geo
  mmesh.generate_medial_tets();
  // if (is_trace_mstruc) mmesh.trace_medial_structure(true /*is_debug*/);

  // mmesh.check_and_store_unthin_tets_in_mat();
  // compute euler
  // int euler = mmesh.vertices->size() - mmesh.edges.size() +
  // mmesh.faces.size();
  int euler = compute_Euler(mmesh);
  printf("[Euler] MedialMesh has Euler %d: v %ld, e %ld, f %ld, t %ld \n",
         euler, mmesh.vertices->size(), mmesh.edges.size(), mmesh.faces.size(),
         mmesh.tets.size());

  // compute medial mesh quality
  double tri_quality = eval_triangle_quality(mmesh);
  polyscope::info("Done calculating medial mesh");
}

int MainGuiWindow::run_fix_geo_itr(
    const SurfaceMesh& sf_mesh, const TetMesh& tet_mesh,
    const Parameter& params, MedialMesh& mmesh,
    std::vector<MedialSphere>& all_medial_spheres, bool is_debug) {
  printf("--------------------Geo Check %d--------------------\n", num_geo_itr);
  int old_size = all_medial_spheres.size();
  check_and_fix_mm_geo(instance_->num_itr_global, sf_mesh, tet_mesh, params,
                       mmesh, all_medial_spheres, this->fix_geo_samples,
                       this->fix_geo_samples_dist2mat,
                       this->fix_geo_samples_clostprim, is_debug /*is_debug*/);
  int num_added = all_medial_spheres.size() - old_size;

  polyscope::info("Geo check added " + std::to_string(num_added) + " spheres");
  num_geo_itr++;
  return num_added;
}

int MainGuiWindow::run_extf_fix_itr(
    const Parameter& param, TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
    const std::vector<ConvexCellHost>& cells_to_show,
    std::vector<MedialSphere>& all_medial_spheres, MedialMesh& mmesh,
    bool is_debug) {
  printf("--------------------EXTF Check %d--------------------\n",
         num_extf_itr);

  // add conrers first
  if (!mmesh.is_corner_spheres_created && !tet_mesh.corners_se_tet.empty()) {
    int num_corners = init_corner_spheres(instance_->num_itr_global, tet_mesh,
                                          all_medial_spheres);
    mmesh.is_corner_spheres_created = true;
    polyscope::info("EXTF check add " + std::to_string(num_corners) +
                    " corners");
    num_extf_itr++;
    return num_corners;
  }

  int old_size = all_medial_spheres.size();
  int num_changed = check_and_fix_external_feature(
      instance_->num_itr_global, param, tet_mesh.tet_vertices, cells_to_show,
      sf_mesh, tet_mesh.tet_vs_lfs2tvs_map, tet_mesh.fl2corner_sphere,
      all_medial_spheres, is_debug /*is_debug*/);
  int num_added = all_medial_spheres.size() - old_size;
  polyscope::info("EXTF check changed " + std::to_string(num_changed) +
                  " spheres");
  num_extf_itr++;
  return num_changed;
}

int MainGuiWindow::run_intf_fix_itr(
    const SurfaceMesh& sf_mesh, const TetMesh& tet_mesh,
    const MedialMesh& mmesh, std::vector<MedialSphere>& all_medial_spheres,
    bool is_debug) {
  printf("--------------------INTF Check %d--------------------\n",
         num_intf_itr);

  int old_size = all_medial_spheres.size();
  random_check_edges_and_fix_internal_feature(
      instance_->num_itr_global, sf_mesh, tet_mesh, mmesh, all_medial_spheres,
      this->checked_mmesh_edges, is_debug);
  int num_added = all_medial_spheres.size() - old_size;

  // update T2 by TN
  int num_updated = 0;
  polyscope::info("INTF check added " + std::to_string(num_added) +
                  ", updated " + std::to_string(num_updated) + " spheres");
  num_intf_itr++;
  return num_added + num_updated;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void get_convex_cell_tet(ConvexCellHost& cc_trans,
                         std::vector<cfloat3>& voro_points,
                         std::vector<std::array<int, 4>>& voro_tets,
                         std::vector<int>& voro_tets_sites,
                         std::vector<int>& voro_tets_cell_ids,
                         std::vector<float>& voro_tets_euler) {
  if (!cc_trans.is_pc_explicit) cc_trans.reload_pc_explicit();
  // assert(cc_trans.is_pc_explicit);
  if (cc_trans.is_vertex_null) {
    printf("[ERROR] do not show cell %d since some vertices are null \n",
           cc_trans.id);
    return;
  }

  // printf("saving cell of site %d \n", cc_trans.voro_id);
  int row = voro_points.size();  // index from 0
  int bary_id = -1;
  cfloat3 bary = {0.0f, 0.0f, 0.0f};  // barycenter of cell
  for (const auto& voro_vertex : cc_trans.pc_points) {
    bary = cplus3(bary, voro_vertex);
    voro_points.push_back(
        cmake_float3(voro_vertex.x, voro_vertex.y, voro_vertex.z));
  }
  bary = cdivide3(bary, cc_trans.nb_v);
  // voro_points.push_back(make_float3(bary.x, bary.y, bary.z));
  voro_points.push_back(bary);
  bary_id = voro_points.size() - 1;

  // some clipping planes may not exist in tri but
  // we still sotre it, here is to filter those planes
  if (!cc_trans.is_active_updated) cc_trans.reload_active();
  assert(cc_trans.is_active_updated);
  const std::vector<int>& active_clipping_planes =
      cc_trans.active_clipping_planes;

  std::vector<std::vector<int>> voro_local_faces;
  int lf = 0;  // local fid

  for (int plane = 0; plane < cc_trans.nb_p; plane++) {
    if (active_clipping_planes[plane] > 0) {
      std::vector<int> tab_v;   // index of dual vertex
      std::vector<int> tab_lp;  // local index of dual vertex in dual triangle
      // for each dual triangle
      for (int t = 0; t < cc_trans.nb_v; t++) {
        // store info of dual vertex
        if ((int)cc_trans.ver_trans(t).x == plane) {
          tab_v.push_back(t);
          tab_lp.push_back(0);
        } else if ((int)cc_trans.ver_trans(t).y == plane) {
          tab_v.push_back(t);
          tab_lp.push_back(1);
        } else if ((int)cc_trans.ver_trans(t).z == plane) {
          tab_v.push_back(t);
          tab_lp.push_back(2);
        }
      }

      if (tab_lp.size() <= 2) {
        std::cout << (int)plane << std::endl;
      }

      int i = 0;
      int j = 0;
      voro_local_faces.push_back(std::vector<int>(0));

      while (voro_local_faces[lf].size() < tab_lp.size()) {
        int ind_i = (tab_lp[i] + 1) % 3;
        bool temp = false;
        j = 0;
        while (temp == false) {
          int ind_j = (tab_lp[j] + 2) % 3;
          if ((int)cc_trans.ith_plane(tab_v[i], ind_i) ==
              (int)cc_trans.ith_plane(tab_v[j], ind_j)) {
            voro_local_faces[lf].push_back(tab_v[i]);
            temp = true;
            i = j;
          }
          j++;
        }
      }

      // triangulate face, then make a tet
      // (bary, p1, p2, p3) makes a tet
      int nb_pts = voro_local_faces[lf].size();
      for (uint p = 1; p < voro_local_faces[lf].size() - 1; p++) {
        // (0, p, p+1) is a triangulation of face
        voro_tets.push_back({{bary_id, row + voro_local_faces[lf][0],
                              row + voro_local_faces[lf][p],
                              row + voro_local_faces[lf][(p + 1) % nb_pts]}});
        voro_tets_sites.push_back(cc_trans.voro_id);
        voro_tets_euler.push_back(cc_trans.euler);
        voro_tets_cell_ids.push_back(cc_trans.id);
      }

      // voro_faces += "f";
      // FOR(i, result[ind].size()) {
      //   voro_faces += " ";
      //   voro_faces += std::to_string(row + voro_faces[ind][i]);
      // }
      // voro_faces += "\n";
      lf++;
    }
  }
}

// 3d triangle mesh
void get_convex_cell_faces(
    ConvexCellHost& cc_trans, std::vector<cfloat3>& voro_points,
    std::vector<float>& voro_points_num_adjs, std::vector<aint3>& voro_faces,
    std::vector<int>& voro_faces_sites, std::vector<int>& voro_faces_cell_ids,
    std::vector<int>& voro_faces_tet_ids,
    std::vector<float>& voro_faces_cell_eulers,
    std::vector<int>& voro_faces_ids, std::vector<int>& voro_faces_sf_ids,
    std::vector<int>& voro_faces_neigh_id, std::vector<int>& voro_faces_num_adj,
    int max_sf_fid, bool is_boundary_only) {
  // assert(cc_trans.is_pc_explicit);
  if (!cc_trans.is_pc_explicit) cc_trans.reload_pc_explicit();
  if (cc_trans.is_vertex_null) {
    printf("[ERROR] do not show cell %d since some vertices are null \n",
           cc_trans.id);
    return;
  }

  int row = voro_points.size();  // index from 0
  // save all vertices
  for (int i = 0; i < cc_trans.pc_points.size(); i++) {
    voro_points.push_back(cc_trans.pc_points[i]);
    voro_points_num_adjs.push_back((int)cc_trans.ver_trans(i).w);
  }

  // some clipping planes may not exist in tri but
  // we still sotre it, here is to filter those planes
  if (!cc_trans.is_active_updated) cc_trans.reload_active();
  assert(cc_trans.is_active_updated);
  const std::vector<int>& active_clipping_planes =
      cc_trans.active_clipping_planes;

  for (int plane = 0; plane < cc_trans.nb_p; plane++) {
    if (active_clipping_planes[plane] <= 0) continue;
    cint2 hp = cc_trans.clip_id2_const(plane);

    // check if only shown boundary or not
    // 1. halfplane; 2. on sf_mesh fid
    bool is_shown = true;
    if (is_boundary_only) {
      is_shown = false;
      if ((hp.y != -1) || (hp.y == -1 && hp.x < max_sf_fid)) {
        is_shown = true;
      }
    }
    if (!is_shown) continue;

    std::vector<int> tab_v;   // index of dual vertex
    std::vector<int> tab_lp;  // local index of dual vertex in dual triangle
    // for each dual triangle
    for (int t = 0; t < cc_trans.nb_v; t++) {
      // store info of dual vertex
      if ((int)cc_trans.ver_trans(t).x == plane) {
        tab_v.push_back(t);
        tab_lp.push_back(0);
      } else if ((int)cc_trans.ver_trans(t).y == plane) {
        tab_v.push_back(t);
        tab_lp.push_back(1);
      } else if ((int)cc_trans.ver_trans(t).z == plane) {
        tab_v.push_back(t);
        tab_lp.push_back(2);
      }
    }

    if (tab_lp.size() <= 2) {
      std::cout << (int)plane << std::endl;
    }

    int i = 0;
    int j = 0;
    int lf = 0;  // local fid
    std::vector<std::vector<int>> voro_local_faces;
    voro_local_faces.push_back(std::vector<int>(0));

    while (voro_local_faces[lf].size() < tab_lp.size()) {
      int ind_i = (tab_lp[i] + 1) % 3;
      bool temp = false;
      j = 0;
      while (temp == false) {
        int ind_j = (tab_lp[j] + 2) % 3;
        if ((int)cc_trans.ith_plane(tab_v[i], ind_i) ==
            (int)cc_trans.ith_plane(tab_v[j], ind_j)) {
          voro_local_faces[lf].push_back(tab_v[i]);
          temp = true;
          i = j;
        }
        j++;
      }
    }

    // triangulate face, then make a tet
    // (bary, p1, p2, p3) makes a tet
    int nb_pts = voro_local_faces[lf].size();
    for (uint p = 1; p < voro_local_faces[lf].size() - 1; p++) {
      // (0, p, p+1) is a triangulation of face
      voro_faces.push_back(
          {{row + voro_local_faces[lf][0], row + voro_local_faces[lf][p],
            row + voro_local_faces[lf][(p + 1) % nb_pts]}});
      voro_faces_sites.push_back(cc_trans.voro_id);
      voro_faces_cell_eulers.push_back(cc_trans.euler);
      voro_faces_cell_ids.push_back(cc_trans.id);
      voro_faces_tet_ids.push_back(cc_trans.tet_id);
      voro_faces_ids.push_back(plane);
      voro_faces_sf_ids.push_back(hp.y == -1 ? hp.x : -1);
      voro_faces_neigh_id.push_back(
          hp.y == -1 ? -1 : (hp.x == cc_trans.voro_id ? hp.y : hp.x));
      voro_faces_num_adj.push_back(cc_trans.clip_trans(plane).h);
    }

    lf++;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// MainGuiWindow functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void MainGuiWindow::compute_rpd_partial_init() {
  instance_->num_sphere_added = all_medial_spheres->size();
  instance_->rpd3d->calculate_partial(instance_->num_itr_global,
                                      instance_->num_sphere_added,
                                      true /*is_given_all_tets*/);
}

// Note: will reset mmesh
void MainGuiWindow::compute_rpd() {
  std::vector<float> spheres;
  instance_->rpd3d->load_spheres(spheres, this->opt_rpd->get_is_rpd());
  instance_->rpd3d->calculate_given_spheres(spheres);
  instance_->rpd3d->update_pcell_samples_cuda(*instance_->sf_mesh,
                                              this->params->is_sample_rpd);
  instance_->rpd3d->cluster_sphere_type_using_samples_fids(
      *instance_->tet_mesh, *instance_->sf_mesh, false /*is_debug*/);
  // clean mmesh if exists
  if (instance_->mmesh) instance_->mmesh->clear();
}

// Note: will reset mmesh
void MainGuiWindow::update_rpd_after_partial() {
  std::vector<float> spheres;
  instance_->rpd3d->load_spheres(spheres, this->opt_rpd->get_is_rpd());
  instance_->rpd3d->update_pcell_samples_cuda(*instance_->sf_mesh,
                                              this->params->is_sample_rpd);
  instance_->rpd3d->cluster_sphere_type_using_samples_fids(
      *instance_->tet_mesh, *instance_->sf_mesh, false /*is_debug*/);
  // clean mmesh if exists
  if (instance_->mmesh) instance_->mmesh->clear();
}

void MainGuiWindow::show(bool is_compute_rpd) {
  // a few camera options
  polyscope::view::upDir = polyscope::UpDir::ZUp;

  // Initialize Polyscope
  polyscope::init();

  // show input tet_mesh
  show_tet_mesh(*(tet_mesh), true /*is_shown_meta*/);

  if (is_compute_rpd) {
    // Note: do not use compute_rpd_partial_init() unless
    //       its first time to run RPD, otherwise it may load
    //       partial spheres than all spheres
    instance_->compute_rpd();
  }

  if (instance_->rpd3d != nullptr && instance_->opt_rpd != nullptr) {
    printf("calling show_result_convex_cells ...\n");
    instance_->show_result_convex_cells(instance_->rpd3d->powercells,
                                        false /*is_slice_plane*/);
    // compute medial mesh
    instance_->run_mmesh_generator(*(instance_->sf_mesh), *(instance_->mmesh),
                                   *(instance_->all_medial_spheres));
  }

  if (instance_->mmesh != nullptr && instance_->mmesh->vertices != nullptr) {
    printf("calling show medial mesh ...\n");
    instance_->show_medial_mesh(*(instance_->mmesh));
  }

  // Show the GUI
  polyscope::show();
}

void MainGuiWindow::auto_eval_add_medge_len(const int itr_print) {
  const clock_t start_t = clock();
  // compute mmesh first if not exist
  if (instance_->mmesh == nullptr || instance_->mmesh->vertices == nullptr)
    instance_->run_mmesh_generator(*(instance_->sf_mesh), *(instance_->mmesh),
                                   *(instance_->all_medial_spheres));
  /////////////////////
  instance_->rpd3d->eval_mmesh_medge_length(*(instance_->mmesh),
                                            false /*is_debug*/);
  instance_->rpd3d->add_new_sphere_after_eval_medge_length(*(instance_->mmesh),
                                                           false /*is_debug*/);
  std::cout << "[EvalAddMedgeLen] itr" << itr_print << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// Note: start by computing mmesh,
//       end by NOT computing RPD, NOT computing mmesh
void MainGuiWindow::auto_eval_add_medge(const int itr_print) {
  const clock_t start_t = clock();
  // compute mmesh first if not exist
  if (instance_->mmesh == nullptr || instance_->mmesh->vertices == nullptr)
    instance_->run_mmesh_generator(*(instance_->sf_mesh), *(instance_->mmesh),
                                   *(instance_->all_medial_spheres));
  /////////////////////
  instance_->rpd3d->eval_mmesh_cand_medge_not_same_sheet(*(instance_->mmesh),
                                                         false /*is_debug*/);
  instance_->rpd3d->add_new_sphere_wrapper_after_eval(*(instance_->mmesh),
                                                      false /*is_debug*/);
  std::cout << "[EvalAddMedge] itr" << itr_print << "took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

void MainGuiWindow::auto_opt_rpd_only_particle() {
  const clock_t start_t = clock();
  instance_->num_all_itr = 0;
  instance_->num_particle_itr = 20;
  int K = 3;
  if (params->is_run_organic) {
    instance_->num_particle_itr = 10;
    K = 5;
  }
  while (instance_->num_all_itr < K) {
    int itr_print = instance_->num_all_itr;
    {
      if (params->is_run_cad)
        instance_->opt_rpd->calculate_particle_lbfgs_globally(
            instance_->num_itr_global, OPT_RPD::RUN_TYPE::PARTICLE_MMESH,
            instance_->num_particle_itr);
      else if (params->is_run_organic)
        instance_->opt_rpd->calculate_particle_lbfgs_globally(
            instance_->num_itr_global, OPT_RPD::RUN_TYPE::PARTICLE_KNN,
            instance_->num_particle_itr);
      else
        assert(false);
    }
    instance_->compute_rpd();  // this is needed for eval
    instance_->auto_eval_add_medge_len(itr_print);
    instance_->compute_rpd();  // this is needed for eval
    instance_->auto_eval_add_medge(itr_print);
    {
      if (params->is_run_cad)
        instance_->proj_all_sphere_clean_extf();
      else if (params->is_run_organic)
        instance_->proj_sphere_SQEM_clean_extf();
      else
        assert(false);
    }
    instance_->num_all_itr++;
    if (params->is_run_cad) instance_->num_particle_itr -= 5;
  }

  // fix topo and extf one more time
  auto_fix_topo_and_clean();
  auto_fix_extf();

  // re-compute mmesh
  instance_->run_mmesh_generator(*(instance_->sf_mesh), *(instance_->mmesh),
                                 *(instance_->all_medial_spheres));

  // cluster
  instance_->rpd3d->cluster_all(*(instance_->tet_mesh), *(instance_->sf_mesh),
                                *(instance_->mmesh), false /*is_debug*/);

  // thinning
  if (params->is_run_organic) instance_->given_thinning_thres = 0.1;
  ThinningWrapper::prune_mtype(
      *(instance_->tet_mesh), *(instance_->sf_mesh),
      *(instance_->all_medial_spheres), *(instance_->rpd3d),
      *(instance_->mmesh), instance_->given_thinning_thres, false /*is_debug*/);
  compute_Euler(*(instance_->mmesh));

  // trace
  if (params->is_run_cad)
    trace_medial_structure(*(instance_->tet_mesh), *(instance_->sf_mesh),
                           *(instance_->rpd3d), *(instance_->mmesh),
                           false /*is_debug*/);

  std::cout << "Total auto_opt_rpd_only_particle took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;

  // save mat
  if (params->is_run_cad)
    export_ma_and_struct_clean(instance_->name_no_ext, *(instance_->mmesh));
  if (params->is_run_organic)
    export_ma_obj_only(instance_->name_no_ext, *(instance_->mmesh));
}

// Note: start by computing mmesh,
//       end by computing RPD, NOT computing mmesh
void MainGuiWindow::auto_fix_extf() {
  const clock_t start_t = clock();
  // compute mmesh first if not exist
  if (instance_->mmesh == nullptr || instance_->mmesh->vertices == nullptr)
    instance_->run_mmesh_generator(*(instance_->sf_mesh), *(instance_->mmesh),
                                   *(instance_->all_medial_spheres));
  // loop fix extf
  while (
      (instance_->num_sphere_added = instance_->run_extf_fix_itr(
           *(instance_->params), *(instance_->tet_mesh), *(instance_->sf_mesh),
           instance_->rpd3d->powercells, *(instance_->all_medial_spheres),
           *(instance_->mmesh), false /*is_debug*/)) > 0) {
    instance_->rpd3d->calculate_partial(instance_->num_itr_global,
                                        instance_->num_sphere_added,
                                        false /*is_given_all_tets*/);
    // save_spheres_file(*(instance_->all_medial_spheres),
    //                   instance_->name_no_ext, true /*is_save_type*/,
    //                   true /*is_load_deleted*/, instance_->num_itr_global,
    //                   instance_->sub_folder_name);
  }
  instance_->update_rpd_after_partial();
  std::cout << "ExtfFix " << instance_->num_extf_itr << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

void MainGuiWindow::auto_fix_intf() {
  const clock_t start_t = clock();
  while (
      (instance_->num_sphere_added = instance_->run_intf_fix_itr(
           *(instance_->sf_mesh), *(instance_->tet_mesh), *(instance_->mmesh),
           *(instance_->all_medial_spheres), false /*is_debug*/)) > 0) {
    instance_->rpd3d->calculate_partial(instance_->num_itr_global,
                                        instance_->num_sphere_added,
                                        false /*is_given_all_tets*/);
    instance_->run_mmesh_generator(*(instance_->sf_mesh), *(instance_->mmesh),
                                   *(instance_->all_medial_spheres),
                                   false /*is_compute_common_diff_sfids*/);
  }  // while
  instance_->update_rpd_after_partial();
  std::cout << "IntfFix " << instance_->num_intf_itr << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

void MainGuiWindow::auto_fix_intf_pre() {
  printf("--------------------INTF_PRE_SE Check--------------------\n");
  const clock_t start_t = clock();
  // compute mmesh first if not exist
  if (instance_->mmesh == nullptr || instance_->mmesh->vertices == nullptr)
    instance_->run_mmesh_generator(*(instance_->sf_mesh), *(instance_->mmesh),
                                   *(instance_->all_medial_spheres));
  // insert
  instance_->num_sphere_added = pre_insert_intf_spheres_for_SE(
      instance_->num_itr_global, *(instance_->sf_mesh), *(instance_->tet_mesh),
      *(instance_->mmesh), *(instance_->all_medial_spheres),
      false /*is_debug*/);
  instance_->rpd3d->calculate_partial(instance_->num_itr_global,
                                      instance_->num_sphere_added,
                                      false /*is_given_all_tets*/);
  instance_->update_rpd_after_partial();
  // save_spheres_file(*(instance_->all_medial_spheres), instance_->name_no_ext,
  //                   true /*is_save_type*/, true /*is_load_deleted*/,
  //                   instance_->num_itr_global, instance_->sub_folder_name);
  std::cout << "IntfFix " << instance_->num_intf_itr << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// Note: start by NOT computing RPD/mmesh,
//       end by computing RPD, NOT computing mmesh
void MainGuiWindow::auto_fix_topo_and_clean() {
  const clock_t start_t = clock();
  // reset MedialSphere::itr_topo_fix
  // this is important since we call fix_topo multiple times
  for (auto& ms : *(instance_->all_medial_spheres)) {
    ms.itr_topo_fix = 0;
  }
  // Note: no need to compute mmesh ahead
  //
  // run topo fix
  while (instance_->num_topo_itr < TOPO_ITR_MAX &&
         (instance_->num_sphere_added = instance_->run_topo_fix_itr(
              *(instance_->sf_mesh), *(instance_->tet_mesh),
              *(instance_->all_medial_spheres), instance_->rpd3d->powercells,
              false /*is_debug*/)) > 0) {
    instance_->rpd3d->calculate_partial(instance_->num_itr_global,
                                        instance_->num_sphere_added,
                                        false /*is_given_all_tets*/);
  }
  // topo fix may mark some spheres as deleted
  clean_deleted_T1_medial_spheres(*instance_->all_medial_spheres);
  // NOTE: cannot call MainGuiWindow::update_rpd_after_partial() here
  // since we need to re-compute the entire RPD
  instance_->compute_rpd();
  // save_spheres_file(*(instance_->all_medial_spheres), instance_->name_no_ext,
  //                   true /*is_save_type*/, true /*is_load_deleted*/,
  //                   instance_->num_itr_global, instance_->sub_folder_name);
  std::cout << "TopoFix " << instance_->num_topo_itr << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// make sure to call compute_rpd() ahead
void MainGuiWindow::proj_all_sphere_clean_extf() {
  ///////////////////////////////////////////
  // project spheres
  // projected UNK, SE and other spheres
  instance_->opt_rpd->calculate_proj_SQEM_KNN(instance_->num_itr_global);
  // re-project TN to make it more accurate
  instance_->compute_rpd();
  OPT_RPD::PROJ_TYPE proj_type;
  proj_type.is_proj_SE = true;
  proj_type.is_proj_TN = true;
  proj_type.is_proj_UNK = true;
  instance_->opt_rpd->calculate_proj(instance_->num_itr_global, proj_type);

  // need to compute RPD for detecting T1 spheres
  instance_->compute_rpd();

  // clean
  remove_close_medial_spheres(*(instance_->all_medial_spheres),
                              instance_->params->clean_sphere_thres_rel *
                                  instance_->params->bbox_diag_l);
  remove_outside_medial_spheres(*(instance_->sf_mesh),
                                *(instance_->all_medial_spheres));
  clean_deleted_T1_medial_spheres(*instance_->all_medial_spheres);

  // fix extf
  instance_->compute_rpd();
  auto_fix_extf();
  //
  // NOTE: auto_fix_extf() has computed RPD, but not medial mesh
}

void MainGuiWindow::proj_sphere_SQEM_clean_extf() {
  ///////////////////////////////////////////
  // project spheres
  instance_->opt_rpd->calculate_proj_SQEM_KNN(instance_->num_itr_global);

  // need to compute RPD for detecting T1 spheres
  instance_->compute_rpd();

  // clean
  remove_close_medial_spheres(*(instance_->all_medial_spheres),
                              instance_->params->clean_sphere_thres_rel *
                                  instance_->params->bbox_diag_l);
  remove_outside_medial_spheres(*(instance_->sf_mesh),
                                *(instance_->all_medial_spheres));
  clean_deleted_T1_medial_spheres(*instance_->all_medial_spheres);

  // fix extf
  instance_->compute_rpd();
  auto_fix_extf();
  //
  // NOTE: auto_fix_extf() has computed RPD, but not medial mesh
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Callback functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void MainGuiWindow::show_result_convex_cells(
    std::vector<ConvexCellHost>& cells_to_show, bool is_slice_plane) {
  std::vector<cfloat3> voro_points;
  std::vector<float> voro_points_num_adjs;
  std::vector<std::array<int, 4>> voro_tets;  // tet
  std::vector<adouble3> voro_colors;          // random color per site
  std::vector<float> voro_tets_euler;
  std::vector<int> voro_tets_sites, voro_tets_cell_ids;
  std::map<int, adouble3> color_map_per_site;

  int num_old_tet = 0;
  for (auto& cell_trans : cells_to_show) {
    assert(cell_trans.id != -1);
    // filter spheres by slicing plane
    Vector3 last_bary = compute_cell_barycenter(cell_trans);
    if (is_slice_plane && is_slice_by_plane(last_bary, *(this->params)))
      continue;

    // if (cell_trans.voro_id != 5 && cell_trans.voro_id != 19 &&
    //     cell_trans.voro_id != 29)
    //   continue;
    num_old_tet = voro_tets.size();
    get_convex_cell_tet(cell_trans, voro_points, voro_tets, voro_tets_sites,
                        voro_tets_cell_ids, voro_tets_euler);
    // assign random color per site
    if (color_map_per_site.find(cell_trans.voro_id) == color_map_per_site.end())
      color_map_per_site[cell_trans.voro_id] = {{polyscope::randomUnit(),
                                                 polyscope::randomUnit(),
                                                 polyscope::randomUnit()}};
    for (int i = 0; i < voro_tets.size() - num_old_tet; i++) {
      voro_colors.push_back(color_map_per_site.at(cell_trans.voro_id));
    }
  }

  // Register the volume mesh with Polyscope
  auto my_mesh =
      polyscope::registerTetMesh("my results", voro_points, voro_tets);
  // Add a scalar quantity
  my_mesh->addCellScalarQuantity("site", voro_tets_sites);
  my_mesh->addCellColorQuantity("site_color", voro_colors)->setEnabled(true);
  my_mesh->addCellScalarQuantity("euler", voro_tets_euler);
  my_mesh->addCellScalarQuantity("cell_id", voro_tets_cell_ids);
}

void MainGuiWindow::show_tet_mesh(const TetMesh& tet_mesh, bool is_shown_meta) {
  std::vector<std::array<float, 3>> tet_vertices_new;
  std::vector<std::array<int, 4>> tet_indices_new;
  std::vector<int> tet_ids;
  for (int i = 0; i < tet_mesh.tet_vertices.size() / 3; i++) {
    tet_vertices_new.push_back(
        {{tet_mesh.tet_vertices[i * 3], tet_mesh.tet_vertices[i * 3 + 1],
          tet_mesh.tet_vertices[i * 3 + 2]}});
  }
  for (int i = 0; i < tet_mesh.tet_indices.size() / 4; i++) {
    tet_indices_new.push_back(
        {{tet_mesh.tet_indices[i * 4], tet_mesh.tet_indices[i * 4 + 1],
          tet_mesh.tet_indices[i * 4 + 2], tet_mesh.tet_indices[i * 4 + 3]}});
    tet_ids.push_back(i);
  }
  auto my_tet =
      polyscope::registerTetMesh("my tet", tet_vertices_new, tet_indices_new);
  my_tet->addCellScalarQuantity("tet_id", tet_ids);
  my_tet->setEnabled(false);

  if (!is_shown_meta) return;
  // show concave corners
  std::vector<int> is_vertex_cc_corner(tet_vertices_new.size(), false);
  for (const auto& cc_corner : tet_mesh.cc_corners) {
    is_vertex_cc_corner[cc_corner.tvid] = true;
  }
  my_tet->addVertexScalarQuantity("is_cc_corner", is_vertex_cc_corner);
  // show convex/concave feature lines
  std::vector<Vector3> ce_edge_vs, se_edge_vs;
  std::vector<aint2> ce_edges, se_edges;
  std::vector<int> ce_fl_ids, ce_fe_ids, se_fl_ids, se_fe_ids;
  for (const auto& one_ce_line : tet_mesh.ce_lines) {
    for (const auto& one_ce_id : one_ce_line.fe_ids) {  // FeatureEdge::id
      const auto one_ce = tet_mesh.feature_edges.at(one_ce_id);
      ce_edge_vs.push_back(one_ce.t2vs_pos[0]);  // v0
      ce_edge_vs.push_back(one_ce.t2vs_pos[1]);  // v1
      ce_edges.push_back({{ce_edge_vs.size() - 2, ce_edge_vs.size() - 1}});
      ce_fl_ids.push_back(one_ce_line.id);  // FeatureLine::id
      ce_fe_ids.push_back(one_ce.id);       // FeatureEdge::id
    }
  }
  auto my_ce_edges =
      polyscope::registerCurveNetwork("edges_ce", ce_edge_vs, ce_edges);
  my_ce_edges->addEdgeScalarQuantity("fl_id", ce_fl_ids)->setEnabled(true);
  my_ce_edges->addEdgeScalarQuantity("fe_id", ce_fe_ids)->setEnabled(false);
  my_ce_edges->setEnabled(false);

  for (const auto& one_se_line : tet_mesh.se_lines) {
    for (const auto& one_se_id : one_se_line.fe_ids) {  // FeatureEdge::id
      const auto one_se = tet_mesh.feature_edges.at(one_se_id);
      se_edge_vs.push_back(one_se.t2vs_pos[0]);  // v0
      se_edge_vs.push_back(one_se.t2vs_pos[1]);  // v1
      se_edges.push_back({{se_edge_vs.size() - 2, se_edge_vs.size() - 1}});
      se_fl_ids.push_back(one_se_line.id);  // FeatureLine::id
      se_fe_ids.push_back(one_se.id);       // FeatureEdge::id
    }
  }
  auto my_se_edges =
      polyscope::registerCurveNetwork("edges_se", se_edge_vs, se_edges);
  my_se_edges->addEdgeScalarQuantity("fl_id", se_fl_ids)->setEnabled(true);
  my_se_edges->addEdgeScalarQuantity("fe_id", se_fe_ids)->setEnabled(true);
  my_se_edges->setEnabled(false);
}

void MainGuiWindow::show_medial_mesh(const MedialMesh& mmesh,
                                     std::string mmname) {
  if (mmesh.vertices == nullptr) return;
  const auto& mspheres = *(mmesh.vertices);
  const auto& mfaces = mmesh.faces;

  std::vector<Vector3> mat_pos(mspheres.size());
  std::vector<double> mat_radius(mspheres.size());
  std::vector<aint3> mat_faces(mfaces.size(), {{0, 0, 0}});
  // show adj mfaces same sheet
  std::vector<Vector3> mat_faces_centers;

  // store mat vertices
  for (uint i = 0; i < mspheres.size(); i++) {
    mat_pos[i] = mspheres[i].center;
    mat_radius[i] = mspheres[i].radius;
  }

  // store mat faces
  // need to enbale culling for drawing ma faces twice
  for (uint f = 0; f < mfaces.size(); f++) {
    if (mfaces[f].is_deleted) continue;
    // draw facets
    uint lv = 0;
    for (const auto& v : mfaces[f].vertices_) {
      mat_faces[f][lv++] = v;
    }
  }

  // Register medial mesh
  auto medial_mesh = polyscope::registerSurfaceMesh(mmname, mat_pos, mat_faces);
  medial_mesh->setBackFacePolicy(polyscope::BackFacePolicy::Identical);
  medial_mesh->addVertexScalarQuantity("radius", mat_radius,
                                       polyscope::DataType::MAGNITUDE);
}
