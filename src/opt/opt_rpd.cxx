#include "opt_rpd.h"

#include "Integral.h"
#include "common_cxx.h"
#include "input_types.h"
#include "io.h"
#include "shrinking.h"
#include "updating.h"

OPT_RPD::~OPT_RPD() {
  delete _tet_mesh;
  delete _sf_mesh;
  delete _all_medial_spheres;
  delete _mmesh;
}

void OPT_RPD::release_pointers() {
  _rpd3d.release_pointers();
  _tet_mesh = nullptr;
  _sf_mesh = nullptr;
  _all_medial_spheres = nullptr;
  _mmesh = nullptr;
}

void OPT_RPD::init(const TetMesh* _tet_mesh, const SurfaceMesh* _sf_mesh,
                   const Parameter* _params,
                   std::vector<MedialSphere>* _all_medial_spheres,
                   MedialMesh* _mmesh) {
  this->_tet_mesh = _tet_mesh;
  this->_sf_mesh = _sf_mesh;
  this->_all_medial_spheres = _all_medial_spheres;
  this->_mmesh = _mmesh;
  _rpd3d.init(_tet_mesh, _sf_mesh, _params, _all_medial_spheres);
  // set alpha, lambdas
  set_lbfgs_params();
  this->_particle_projection_cache.clear();
}

void OPT_RPD::set_file_name_no_ext(std::string _name_no_ext) {
  this->name_no_ext = _name_no_ext;
}

std::string OPT_RPD::get_save_folder_path(
    const std::string& name_no_ext) const {
  std::string folder_path = "../out/" + name_no_ext + "/sph_itr";
  // printf("folder_path: %s\n", folder_path.c_str());
  return folder_path;
}

Vector3 OPT_RPD::get_sphere_center(const int sphere_id) {
  return Vector3(this->_spheres_itr(sphere_id * this->dim),
                 this->_spheres_itr(sphere_id * this->dim + 1),
                 this->_spheres_itr(sphere_id * this->dim + 2));
}

Vector3 OPT_RPD::get_sphere_center(const Eigen::VectorXd& X,
                                   const int sphere_id) {
  return Vector3(X(sphere_id * this->dim), X(sphere_id * this->dim + 1),
                 X(sphere_id * this->dim + 2));
}

void OPT_RPD::save_spheres_itr_file(const Eigen::VectorXd& spheres_itr,
                                    const std::string folder_path,
                                    const std::string filename,
                                    RUN_TYPE run_type, int num_rpd_itr) {
  assert(!folder_path.empty() && !filename.empty());
  create_dir(folder_path);
  std::string type_name = "_";
  if (run_type == RUN_TYPE::PROJ)
    type_name = "_proj";
  else if (run_type == RUN_TYPE::PARTICLE_MMESH)
    type_name = "_particle";
  else if (run_type == RUN_TYPE::PARTICLE_KNN)
    type_name = "_particleknn";
  else
    assert(false);
  std::string sphere_path = folder_path + "/sph_" + filename + "itr" +
                            std::to_string(num_rpd_itr) + type_name + ".sph";
  int n_site = spheres_itr.size() / dim;
  std::fstream file;
  file.open(sphere_path, std::ios_base::out);
  file << 4 << " " << n_site << std::endl;
  for (int i = 0; i < n_site; i++) {
    file << std::setiosflags(std::ios::fixed) << std::setprecision(15)
         << spheres_itr(i * dim + 0) << " " << spheres_itr(i * dim + 1) << " "
         << spheres_itr(i * dim + 2) << " " << spheres_itr(i * dim + 3);
    file << " " << this->_all_medial_spheres->at(i).type;
    file << " " << 0;
    file << std::endl;
  }
  file.close();
  printf("saved .sph file %s\n", sphere_path.c_str());
}

// save this->_all_medial_spheres to this->_spheres_itr
void OPT_RPD::load_spheres_from_all() {
  int num_spheres = this->_all_medial_spheres->size();
  this->_spheres_itr.resize(num_spheres * dim);
  for (int i = 0; i < num_spheres; ++i) {
    const auto& msphere = this->_all_medial_spheres->at(i);
    this->_spheres_itr(i * dim) = msphere.center[0];
    this->_spheres_itr(i * dim + 1) = msphere.center[1];
    this->_spheres_itr(i * dim + 2) = msphere.center[2];
    this->_spheres_itr(i * dim + 3) = msphere.radius;
  }
}

void OPT_RPD::load_sphere_centers_old_from_all() {
  int num_spheres = this->_all_medial_spheres->size();
  this->_sphere_centers_old.clear();
  this->_sphere_centers_old.resize(num_spheres);
  for (int i = 0; i < num_spheres; ++i) {
    const auto& msphere = this->_all_medial_spheres->at(i);
    this->_sphere_centers_old.at(i) = msphere.center;
  }
}

// save this->_spheres_itr to this->_all_medial_spheres
void OPT_RPD::save_spheres_to_all() {
  if (this->_spheres_itr.size() != this->_all_medial_spheres->size() * dim) {
    printf("spheres_itr size: %d, all_medial_spheres size: %d\n",
           this->_spheres_itr.size(), this->_all_medial_spheres->size());
  }
  assert(this->_spheres_itr.size() == this->_all_medial_spheres->size() * dim);
  int num_spheres = this->_all_medial_spheres->size();
  GEO::parallel_for(0, num_spheres, [&](size_t i) {
    auto& msphere = this->_all_medial_spheres->at(i);
    msphere.center[0] = this->_spheres_itr(i * dim);
    msphere.center[1] = this->_spheres_itr(i * dim + 1);
    msphere.center[2] = this->_spheres_itr(i * dim + 2);
    msphere.radius = this->_spheres_itr(i * dim + 3);
  });
}

// save X to both this->_spheres_itr and this->_all_medial_spheres
void OPT_RPD::save_spheres_to_all(const Eigen::VectorXd& X) {
  // save X to this->_spheres_itr
  assert(X.size() == this->_spheres_itr.size());
  GEO::parallel_for(0, X.size(),
                    [&](size_t i) { this->_spheres_itr(i) = X(i); });
  this->save_spheres_to_all();
}

void OPT_RPD::set_lbfgs_params() {
  this->params.alpha_print = 1.0;  // only for printing
  // this is to make particle gradient less close to 0
  // when close to 0 too soon, it would stop optmize
  this->params.lambda_particle = 1E5;
  this->params.gamma = 0.8;                 // odt
  this->params.particle_kernel_sigma = -1;  // sigma in [Zhong2013]
  std::cout << "[LBFGS Params] "
            << "particle:" << to_string(this->params.lambda_particle)
            << ", gamma:" << to_string(this->params.gamma) << std::endl;
}

void OPT_RPD::reset_loss() {
  this->loss.lossParticle = 0.0;
  this->loss.lossTotal = 0.0;
}

void OPT_RPD::update_particle_projection_gradient_cache(bool is_debug) {
  printf("[Particle] Cashing gradient for particle projection...\n");
  // compute projection gradient or tangent
  int num_spheres = this->_all_medial_spheres->size();
  this->_particle_projection_cache.clear();
  this->_particle_projection_cache.resize(num_spheres);

  auto run_thread = [&](int i) {
    auto& msphere = this->_all_medial_spheres->at(i);
    assert(msphere.id == i);
    get_particle_projection_gradient_or_tangent(
        msphere, this->_tet_mesh->feature_edges, this->_rpd3d,
        this->_particle_projection_cache[i], is_debug);
  };
  // for (int i = 0; i < num_spheres; i++) run_thread(i);
  GEO::parallel_for(0, num_spheres, [&](int i) { run_thread(i); });
}

// Note:
// 1. will reset mmesh
// 2. will update this->_all_medial_spheres and this->_spheres_itr from X
//    by calling this->save_spheres_to_all()
// 3. sample NOT based on RPD, but small sphere around the center
void OPT_RPD::calculate_spheres_samples(const Eigen::VectorXd& X) {
  const clock_t start_t = clock();
  assert(X.size() != 0);
  // for RPD3D_Wrapper::cluster_sphere_type_using_samples_fids()
  this->save_spheres_to_all(X);
  _rpd3d.update_pcell_samples_cuda(*this->_sf_mesh, false /*is_sample_rpd*/);

  // clean mmesh if exists
  if (this->_mmesh) this->_mmesh->clear();

  std::cout << "[Timer] calculate_spheres_samples took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// Note:
// 1. will reset mmesh
// 2. will update this->_all_medial_spheres and this->_spheres_itr from X
//    by calling this->save_spheres_to_all()
// 3. do NOT run clustering here
void OPT_RPD::calculate_rpd_rvd_samples(const Eigen::VectorXd& X) {
  const clock_t start_t = clock();
  assert(X.size() != 0);
  // for RPD3D_Wrapper::cluster_sphere_type_using_samples_fids()
  this->save_spheres_to_all(X);
  auto spheres_given = convert2std(X);
  if (!this->is_rpd) {
    printf("[CAL_RPD] set to RVD..\n");
    // set raidus = 1 for RVD
    _rpd3d.reset_radii(spheres_given, 1.0f);  // use RVDs
  }
  _rpd3d.calculate_given_spheres(spheres_given);
  _rpd3d.update_pcell_samples_cuda(*this->_sf_mesh, true /*is_sample_rpd*/);

  // clean mmesh if exists
  if (this->_mmesh) this->_mmesh->clear();

  std::cout << "[Timer] calculate_rpd_rvd_samples took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// Note:
// 1. will reset mmesh
// 2. will update this->_all_medial_spheres and this->_spheres_itr from X
//    by calling this->save_spheres_to_all()
void OPT_RPD::calculate_rpd_rvd_and_cluster_samples(const Eigen::VectorXd& X) {
  const clock_t start_t = clock();
  assert(X.size() != 0);
  // for RPD3D_Wrapper::cluster_sphere_type_using_samples_fids()
  this->save_spheres_to_all(X);
  auto spheres_given = convert2std(X);
  if (!this->is_rpd) {
    printf("[CAL_RPD] set to RVD..\n");
    // set raidus = 1 for RVD
    _rpd3d.reset_radii(spheres_given, 1.0f);  // use RVDs
  }
  _rpd3d.calculate_given_spheres(spheres_given);
  _rpd3d.update_pcell_samples_cuda(*this->_sf_mesh, true /*is_sample_rpd*/);
  _rpd3d.cluster_sphere_type_using_samples_fids(
      *this->_tet_mesh, *this->_sf_mesh, false /*is_debug*/);

  // clean mmesh if exists
  if (this->_mmesh) this->_mmesh->clear();

  std::cout << "[Timer] calculate_rpd_rvd_and_cluster_samples took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// update medial mesh radii to cloest distance to surface
void OPT_RPD::update_mspheres_radii(const SurfaceMesh& sf_mesh) {
  this->save_spheres_to_all();
  auto run_parallel = [&](int i) {
    auto& msphere = this->_all_medial_spheres->at(i);
    if (msphere.is_deleted) return;
    if (msphere.is_on_corner()) return;
    double sq_dist = sf_mesh.aabb_wrapper.get_sq_dist_to_sf(msphere.center);
    msphere.radius = sqrt(sq_dist);
    this->_spheres_itr(i * this->dim + 3) = msphere.radius;
  };
  GEO::parallel_for(0, this->_all_medial_spheres->size(), run_parallel);
  this->load_spheres_from_all();
}

// compute medial mesh
void OPT_RPD::calculate_mmesh() {
  const clock_t start_t = clock();
  // update this->_all_medial_spheres from this->_spheres_itr
  this->save_spheres_to_all();
  // compute medial mesh
  this->_rpd3d.update_spheres_power_cells(false /*is_compute_se_sfids*/);
  this->_mmesh->clear();
  this->_mmesh->genearte_medial_spheres(*this->_all_medial_spheres);
  this->_mmesh->generate_medial_edges(*this->_sf_mesh,
                                      false /*is_compute_common_diff_sfids*/);
  this->_mmesh->generate_medial_faces();
  std::cout << "[Timer] calculate_mmesh took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// compute medial mesh and mark same sheet
void OPT_RPD::calculate_mmesh_and_mark_same_sheet() {
  const clock_t start_t = clock();
  calculate_mmesh();
  // cluster is_same_sheet
  this->_rpd3d.cluster_medges_same_sheet(*this->_tet_mesh, *this->_sf_mesh,
                                         *this->_mmesh, false /*is_debug*/);
  this->_rpd3d.cluster_mfaces_same_sheet(*this->_tet_mesh, *this->_sf_mesh,
                                         *this->_mmesh, false /*is_debug*/);
  std::cout << "[Timer] calculate_mmesh_and_mark_same_sheet took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// compute only medial medges and mark same sheet
void OPT_RPD::calculate_only_medges_and_mark_same_sheet() {
  const clock_t start_t = clock();
  // update this->_all_medial_spheres from this->_spheres_itr
  this->save_spheres_to_all();
  // compute medial mesh
  this->_rpd3d.update_spheres_power_cells(false /*is_compute_se_sfids*/);
  this->_mmesh->clear();
  this->_mmesh->genearte_medial_spheres(*this->_all_medial_spheres);
  this->_mmesh->generate_medial_edges(*this->_sf_mesh,
                                      false /*is_compute_common_diff_sfids*/);
  this->_rpd3d.cluster_medges_same_sheet(*this->_tet_mesh, *this->_sf_mesh,
                                         *this->_mmesh, false /*is_debug*/);
  std::cout << "[Timer] calculate_only_medges_and_mark_same_sheet took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

void print_matrix(const Matrix4& A) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      printf("%f ", A(i, j));
    }
    printf("\n");
  }
}

// using sphere-shrinking algo
bool OPT_RPD::shrink_one_sphere_wrapper(MedialSphere& msphere, bool is_debug) {
  if (is_debug) printf("shrinking sphere %d\n", msphere.id);
  MedialSphere new_msphere = msphere;
  // if contains cc_lines
  if (!msphere.tan_cc_lines.empty()) {
    const auto one_cc_line = msphere.tan_cc_lines[0];
    if (update_new_concave_sphere(*this->_sf_mesh,
                                  this->_tet_mesh->feature_edges,
                                  one_cc_line.tan_point, one_cc_line.id_fe,
                                  SphereType::T_2_c, new_msphere, is_debug)) {
      msphere = new_msphere;
      return true;
    }
  }

  // if not contains cc_lines
  double sq_dist;
  v2int v2fid_chosen;
  if (!msphere.tan_planes.empty()) {
    const auto& one_tan_plane = msphere.tan_planes[0];
    v2fid_chosen.first = one_tan_plane.tan_point;
    v2fid_chosen.second = one_tan_plane.fid;
  } else {
    v2fid_chosen.second = this->_sf_mesh->aabb_wrapper.get_nearest_point_on_sf(
        msphere.center, v2fid_chosen.first, sq_dist);
  }
  if (!update_msphere_given_v2fid(*this->_sf_mesh, *this->_tet_mesh,
                                  v2fid_chosen, new_msphere.id, new_msphere,
                                  true /*is_merge_to_ce*/, false /*is_debug*/))
    return false;
  // replace only successfully projected spheres
  msphere = new_msphere;
  return true;
}

bool OPT_RPD::is_sphere_TN_and_contains_cc_lines_wrapper(const int sphere_id) {
  if (!is_sphere_TN_wrapper(sphere_id)) return false;
  if (!is_sphere_contains_cc_lines_wrapper(sphere_id)) return false;
  return true;
}

bool OPT_RPD::is_sphere_contains_cc_lines_wrapper(const int sphere_id) {
  assert(sphere_id < this->_rpd3d.sphere_clusters.size());
  assert(sphere_id < this->_all_medial_spheres->size());
  auto& msphere = this->_all_medial_spheres->at(sphere_id);
  const auto& clusters = this->_rpd3d.sphere_clusters.at(sphere_id);
  int num_cluster_contains_ce = 0;
  for (const auto& cluster : clusters) {
    if (!cluster.ce_line_ids.empty()) {
      num_cluster_contains_ce++;
    }
  }
  if (num_cluster_contains_ce > 0 || msphere.tan_cc_lines.size() > 0)
    return true;
  return false;
}

bool OPT_RPD::is_sphere_SE_wrapper(const int sphere_id) {
  const auto& msphere = this->_all_medial_spheres->at(sphere_id);
  if (msphere.is_on_se()) return true;
  return false;
}

bool OPT_RPD::is_sphere_TN_wrapper(const int sphere_id) {
  int num_clusters = this->_rpd3d.get_msphere_num_clusters(sphere_id);
  if (num_clusters > 2) return true;
  return false;
}

bool OPT_RPD::is_sphere_UNK_wrapper(const int sphere_id) {
  assert(sphere_id < this->_rpd3d.sphere_pcell_samples.size());
  assert(sphere_id < this->_all_medial_spheres->size());
  auto& msphere = this->_all_medial_spheres->at(sphere_id);
  if (msphere.type == SphereType::T_UNK || msphere.is_deleted ||
      this->_rpd3d.sphere_pcell_samples.at(sphere_id).empty())
    return true;
  return false;
}

// return:
// 1. true: sphere is projected
// 2. false: sphere is not projected, not change
bool OPT_RPD::project_one_sphere(const PROJ_TYPE& proj_type,
                                 const int sphere_id) {
  // reference
  auto& msphere = this->_all_medial_spheres->at(sphere_id);
  if (msphere.is_on_corner()) return false;
  bool is_thread_debug = false;
  // if (msphere.id == 2353) is_thread_debug = true;

  // if on se, then project to se
  // after ODT, so spheres can move as well
  if (proj_type.is_proj_SE && msphere.is_on_se()) {
    // update msphere.center
    int se_edge_id = this->_sf_mesh->aabb_wrapper.project_to_se(msphere.center);
    msphere.type = SphereType::T_1_2;
    msphere.radius = SCALAR_FEATURE_RADIUS;
    msphere.se_edge_id = se_edge_id;
    msphere.se_line_id = this->_tet_mesh->get_fl_id(se_edge_id);
    // printf(
    //     "[PROJ] sphere %d is on SE, project to SE se_edge_id %d/%d, "
    //     "radius %f\n",
    //     sphere_id, se_edge_id, msphere.se_line_id, msphere.radius);
    return true;
  }

  // using sphere-shrinking algo to project:
  // 1. type T_UNK
  assert(sphere_id < this->_rpd3d.sphere_pcell_samples.size());
  if (proj_type.is_proj_UNK && is_sphere_UNK_wrapper(sphere_id)) {
    return shrink_one_sphere_wrapper(msphere, is_thread_debug);
  }

  // 2. type T_2 && is_proj_T2 = true
  if (proj_type.is_proj_T2 && msphere.type == SphereType::T_2 &&
      msphere.tan_cc_lines.empty()) {
    return shrink_one_sphere_wrapper(msphere, is_thread_debug);
  }

  // project using sphere-optimizataion algo
  if (proj_type.is_proj_TN && is_sphere_TN_wrapper(sphere_id)) {
    return this->_rpd3d.project_one_TN_sphere(msphere, is_thread_debug);
  }

  // any sphere tangent to cc_lines
  if (proj_type.is_proj_cc &&
      is_sphere_contains_cc_lines_wrapper(sphere_id) > 0) {
    return this->_rpd3d.project_one_TN_sphere(msphere, is_thread_debug);
  }

  if (proj_type.is_proj_TN_cc &&
      is_sphere_TN_and_contains_cc_lines_wrapper(sphere_id)) {
    return this->_rpd3d.project_one_TN_sphere(msphere, is_thread_debug);
  }

  return false;
}

#include "shrinking.h"
// directly project spheres in this->_all_medial_spheres
void OPT_RPD::run_spheres_projection(const PROJ_TYPE& proj_type) {
  // compute RPD for updating sphere types
  this->calculate_rpd_rvd_and_cluster_samples(this->_spheres_itr);

  //------------------------------------------------------------------
  std::vector<int> is_sphere_projected(this->_all_medial_spheres->size(), 0);
  // project all spheres in this->_all_medial_spheres in multi-thread

  // run multi-thread
  GEO::parallel_for(0, this->_all_medial_spheres->size(), [&](int i) {
    is_sphere_projected[i] = project_one_sphere(proj_type, i);
  });
  // for (int i = 0; i < this->_all_medial_spheres->size(); i++)
  //   run_thread_project(i);

  // print number of sphere updated
  int num_projected = 0;
  for (int i = 0; i < is_sphere_projected.size(); i++) {
    if (is_sphere_projected[i] == 1) num_projected++;
  }
  printf("[PROJ] projected %d/%zu spheres\n", num_projected,
         this->_all_medial_spheres->size());
}

// ----------------------------------------------
// ---------------Particle Smoothing-------------
// ----------------------------------------------
void OPT_RPD::update_particle_kernel_helper() {
  // force to mark the same sheet
  this->calculate_mmesh_and_mark_same_sheet();

  // compute total area of mmesh faces on the same
  double mmesh_total_area = 0.f;
  for (const auto& mface : this->_mmesh->faces) {
    // // ignore if any medges is not on the same sheet
    // bool is_on_same_sheet = true;
    // for (const int meid : mface.edges_) {
    //   if (!this->_mmesh->edges[meid].is_on_same_sheet) {
    //     is_on_same_sheet = false;
    //     break;
    //   }
    // }
    if (!mface.is_on_same_sheet) {
      if (is_debug)
        printf("[ODT] face %d is not on the same sheet, continue\n", mface.fid);
      continue;
    }
    // collect mface area
    mmesh_total_area += GEO::Geom::triangle_area(
        this->_all_medial_spheres->at(mface.vertices_[0]).center,
        this->_all_medial_spheres->at(mface.vertices_[1]).center,
        this->_all_medial_spheres->at(mface.vertices_[2]).center);
  }
  double particle_kernel_const = 0.3f;  // c_sigma in [Zhong2013]
  this->params.particle_kernel_sigma =
      particle_kernel_const *
      std::sqrt(mmesh_total_area / this->_all_medial_spheres->size());

  printf("[Particle] updated particle_kernel_sigma %f, mmesh_total_area: %f\n",
         this->params.particle_kernel_sigma, mmesh_total_area);
}

// compute RPD and particle SQEM
void OPT_RPD::project_sheet_spheres_SQEM(Eigen::VectorXd& X) {
  printf("[PROJ_SQEM_KNN] project particle spheres KNN ...\n");
  // if (X.size() == 0 || X.size() / this->dim <
  // this->_all_medial_spheres->size())
  this->load_spheres_from_all();
  // else
  // this->save_spheres_to_all(X);
  //
  this->calculate_spheres_samples(X);
  this->update_particle_projection_gradient_cache(false /*is_debug*/);

  // update mspheres only
  std::vector<int> is_sphere_projected(this->_all_medial_spheres->size(), 0);
  auto run_thread_sphere_proj = [&](int sphere_id) {
    // is_debug = false;
    // if (sphere_id == 30) is_debug = true;
    auto& msphere = this->_all_medial_spheres->at(sphere_id);
    // also update deleted spheres
    // if (msphere.is_deleted) return;
    if (is_sphere_UNK_wrapper(sphere_id)) {
      PROJ_TYPE proj_type;
      proj_type.is_proj_UNK = true;
      // update this->_all_medial_spheres
      bool is_good = this->project_one_sphere(proj_type, sphere_id);
      const auto& new_msphere = this->_all_medial_spheres->at(sphere_id);
      if (is_debug) {
        printf("[PROJ_SQEM_KNN] sphere %d, type: %d, CC_or_UNK, is_good: %d\n",
               sphere_id, new_msphere.type, is_good);
      }
    } else if (this->_rpd3d.params->is_run_organic ||
               (this->_rpd3d.params->is_run_cad && msphere.is_on_sheet())) {
      assert(sphere_id < this->_all_medial_spheres->size());
      Vector4 mi_tmp(msphere.center[0], msphere.center[1], msphere.center[2],
                     msphere.radius);
      assert(sphere_id < this->_particle_projection_cache.size());
      project_particle_sphere(this->_particle_projection_cache[sphere_id],
                              mi_tmp, is_debug);
      // update this->_all_medial_spheres
      msphere.center[0] = mi_tmp[0];
      msphere.center[1] = mi_tmp[1];
      msphere.center[2] = mi_tmp[2];
      msphere.radius = mi_tmp[3];
      // sanity check on sphere radius
      double sq_dist =
          this->_sf_mesh->aabb_wrapper.get_sq_dist_to_sf(msphere.center);
      msphere.radius = sqrt(sq_dist);
      if (is_debug) {
        printf(
            "[PROJ_SQEM_KNN] sphere %d, type %d, NO_CC, using SQEM to proj, "
            "(%f,%f,%f,%f)->(%f,%f,%f,%f) \n",
            sphere_id, msphere.type, msphere.center[0], msphere.center[1],
            msphere.center[2], msphere.radius, mi_tmp[0], mi_tmp[1], mi_tmp[2],
            mi_tmp[3]);
      }
    } else {
      // do nothing for other types
      return;
    }
    is_sphere_projected[sphere_id] = 1;
  };
  GEO::parallel_for(0, this->_all_medial_spheres->size(),
                    [&](int sphere_id) { run_thread_sphere_proj(sphere_id); });
  // for (int i = 0; i < this->_all_medial_spheres->size(); i++)
  //   run_thread_sphere_proj(i);

  // update this->_spheres_itr
  this->load_spheres_from_all();
  // update radius just in case
  // this->update_mspheres_radii(*this->_sf_mesh);

  // print number of sphere updated
  int num_projected = 0;
  for (int i = 0; i < is_sphere_projected.size(); i++) {
    if (is_sphere_projected[i] == 1) num_projected++;
  }
  printf("[PROJ_SQEM_KNN] projected %d/%zu spheres\n", num_projected,
         this->_all_medial_spheres->size());
}

// compute RPD and particle SQEM
void OPT_RPD::project_particle_spheres_SQEM(Eigen::VectorXd& X) {
  printf("[PROJ_SQEM] project particle spheres ...\n");
  if (X.size() == 0)
    this->load_spheres_from_all();
  else
    this->save_spheres_to_all(X);
  //
  std::vector<int> is_sphere_projected(this->_all_medial_spheres->size(), 0);
  this->calculate_spheres_samples(X);
  this->update_particle_projection_gradient_cache(false /*is_debug*/);

  // update mspheres only
  auto run_thread_sphere_proj = [&](int sphere_id) {
    is_debug = false;
    if (sphere_id == 351) is_debug = true;
    auto& msphere = this->_all_medial_spheres->at(sphere_id);
    if (msphere.is_deleted) return;
    if (is_sphere_UNK_wrapper(sphere_id)) {
      PROJ_TYPE proj_type;
      proj_type.is_proj_UNK = true;
      // update this->_all_medial_spheres
      bool is_good = this->project_one_sphere(proj_type, sphere_id);
      const auto& new_msphere = this->_all_medial_spheres->at(sphere_id);
      if (is_debug) {
        printf("[Particle] sphere %d, type: %d, CC_or_UNK, is_good: %d\n",
               sphere_id, new_msphere.type, is_good);
      }
    } else {
      assert(sphere_id < this->_all_medial_spheres->size());
      Vector4 mi_tmp(msphere.center[0], msphere.center[1], msphere.center[2],
                     msphere.radius);
      assert(sphere_id < this->_particle_projection_cache.size());
      project_particle_sphere(this->_particle_projection_cache[sphere_id],
                              mi_tmp, is_debug);
      if (is_debug) {
        printf(
            "[Particle] sphere %d, type %d, NO_CC, using SQEM to proj, "
            "(%f,%f,%f,%f)->(%f,%f,%f,%f) \n",
            sphere_id, msphere.type, msphere.center[0], msphere.center[1],
            msphere.center[2], msphere.radius, mi_tmp[0], mi_tmp[1], mi_tmp[2],
            mi_tmp[3]);
      }
      // update this->_all_medial_spheres
      msphere.center[0] = mi_tmp[0];
      msphere.center[1] = mi_tmp[1];
      msphere.center[2] = mi_tmp[2];
      // msphere.radius = isnan(mi_tmp[3]) ? msphere.radius : mi_tmp[3];
      msphere.radius = mi_tmp[3];
    }
    is_sphere_projected[sphere_id] = 1;
  };
  GEO::parallel_for(0, this->_all_medial_spheres->size(),
                    [&](int sphere_id) { run_thread_sphere_proj(sphere_id); });
  // for (int i = 0; i < this->_all_medial_spheres->size(); i++)
  //   run_thread_sphere_proj(i);

  // update this->_spheres_itr
  this->load_spheres_from_all();

  // print number of sphere updated
  int num_projected = 0;
  for (int i = 0; i < is_sphere_projected.size(); i++) {
    if (is_sphere_projected[i] == 1) num_projected++;
  }
  printf("[PROJ_SQEM] projected %d/%zu spheres\n", num_projected,
         this->_all_medial_spheres->size());
}

void OPT_RPD::integrate_particle_KNN(Eigen::VectorXd& X, Eigen::VectorXd& g) {
  printf("[Particle] calling integrate particle knn ...\n");
  // needs to make sure mmesh exists
  assert(this->_mmesh != nullptr);

  // has called OPT_RPD::update_particle_kernel_helper() ahead
  assert(this->params.particle_kernel_sigma != -1);

  // find sphere neighbors
  std::vector<std::set<int>> site_adj_sites;
  this->_rpd3d.get_sites_KNN(site_adj_sites,
                             this->params.particle_kernel_sigma);

  // reset loss
  int num_spheres = X.size() / this->dim;
  std::vector<double> lossParticles_tmp(num_spheres, 0.f);
  // compute K neighbors for each particle i
  auto run_thread_particle_mmesh = [&](int sphere_id) {
    // is_debug = false;
    // if (sphere_id == 30) is_debug = true;
    // particle i
    const Vector3& msphere_center = get_sphere_center(X, sphere_id);
    const auto& msphere = this->_all_medial_spheres->at(sphere_id);
    // // if deleted, return
    // if (msphere.is_deleted) return;
    // // do not update corner, stay still
    // if (msphere.is_on_corner()) return;

    // Step 1: get valid neighbors j neigh_id
    assert(sphere_id < site_adj_sites.size());
    const auto& valid_msphere_neigh_ids = site_adj_sites.at(sphere_id);
    if (is_debug) {
      printf("sphere_id: %d, ", sphere_id);
      print_set<int>(valid_msphere_neigh_ids, "valid_msphere_neigh_ids");
    }
    if (valid_msphere_neigh_ids.empty()) {
      // may happens for corners and T_4 spheres
      if (is_debug)
        printf("[Particle] sphere %d has no valid neighbors in RPD\n",
               sphere_id);
      return;
    }

    // Step 2: loop all neighbors in RPD
    // for (int j = 0; j < valid_msphere_neigh_ids.size(); j++) {
    // int sphere_neigh_id = valid_msphere_neigh_ids.at(j);
    for (int sphere_neigh_id : valid_msphere_neigh_ids) {
      // particle j
      if (sphere_id == sphere_neigh_id) continue;
      const auto& msphere_neigh =
          this->_all_medial_spheres->at(sphere_neigh_id);
      double kernel = this->params.particle_kernel_sigma;
      double kernel_sq = std::pow(kernel, 2);
      const auto neigh_center = get_sphere_center(X, sphere_neigh_id);
      // get lambda_scale
      double lambda_scale = 1.f;
      // accumulate particle energy
      double length_sq = GEO::Geom::distance2(neigh_center, msphere_center);
      // // v1/v2: use [Zhong 2013]
      // double e_ij =
      //     lambda_scale * std::exp(-1.f * length_sq / (4.f * kernel_sq));
      // v3: use [Witkin 1994]
      double e_ij =
          lambda_scale * std::exp(-1.f * length_sq / (2.f * kernel_sq));
      lossParticles_tmp[sphere_id] += this->params.lambda_particle * e_ij;
      // accumulate gradient, the force applied on particle i by particle j
      // not grad_ij!!!!
      //
      // v1: [Zhong2013] Eqn22
      // Vector3 grad_ji = (neigh_center - msphere_center) * e_ij / (2.f *
      // kernel_sq);
      //
      // v2: we normalize the force for spheres on sheets
      // Vector3 grad_ji = GEO::normalize(neigh_center - msphere_center) *e_ij
      // / (2.f * kernel_sq);
      //
      // v3: use [Witkin 1994]
      Vector3 grad_ji =
          GEO::normalize(neigh_center - msphere_center) * e_ij / kernel_sq;
      // Vector3 grad_ji = (neigh_center - msphere_center) * e_ij / kernel_sq;
      //
      g(sphere_id * this->dim + 0) += this->params.lambda_particle * grad_ji[0];
      g(sphere_id * this->dim + 1) += this->params.lambda_particle * grad_ji[1];
      g(sphere_id * this->dim + 2) += this->params.lambda_particle * grad_ji[2];
      g(sphere_id * this->dim + 3) += 0.f;  // no r

      if (is_debug)
        printf(
            "[Particle] sphere %d processing knn neighbor %d, length: %f, "
            "kernel: %f, e_ij: %f, e_ij/(2.f*kernel_sq): %f, grad_ji: "
            "(%f,%f,%f), g: (%f,%f,%f) \n",
            sphere_id, sphere_neigh_id, std::sqrt(length_sq), kernel, e_ij,
            e_ij / (2.f * kernel_sq), grad_ji[0], grad_ji[1], grad_ji[2],
            g(sphere_id * this->dim + 0), g(sphere_id * this->dim + 1),
            g(sphere_id * this->dim + 2));
    }

    // NOTE:
    // if g is too small, then no need to project, no need to move at all
    // otherwise g.norm() may be NAN, affects the LBFGS
    if (std::abs(g(sphere_id * this->dim + 0)) <= SCALAR_ZERO_4 &&
        std::abs(g(sphere_id * this->dim + 1)) <= SCALAR_ZERO_4 &&
        std::abs(g(sphere_id * this->dim + 2)) <= SCALAR_ZERO_4) {
      g(sphere_id * this->dim + 0) = 0.f;
      g(sphere_id * this->dim + 1) = 0.f;
      g(sphere_id * this->dim + 2) = 0.f;
      g(sphere_id * this->dim + 3) = 0.f;
      return;
    }

    // Step 3: project g
    Vector3 g_tmp(g(sphere_id * this->dim + 0), g(sphere_id * this->dim + 1),
                  g(sphere_id * this->dim + 2));
    assert(!this->_particle_projection_cache.empty());
    project_particle_gradient(this->_particle_projection_cache[sphere_id],
                              g_tmp, is_debug);
    g(sphere_id * this->dim + 0) = g_tmp[0];
    g(sphere_id * this->dim + 1) = g_tmp[1];
    g(sphere_id * this->dim + 2) = g_tmp[2];
    g(sphere_id * this->dim + 3) = 0.f;
  };

  GEO::parallel_for(0, num_spheres,
                    [&](int i) { run_thread_particle_mmesh(i); });
  // for (int i = 0; i < num_spheres; i++) run_thread_particle_mmesh(i);

  this->loss.lossParticle = 0.f;
  for (int sphere_id = 0; sphere_id < num_spheres; sphere_id++)
    this->loss.lossParticle += lossParticles_tmp[sphere_id];
  this->loss.lossTotal += this->loss.lossParticle;
}

// must call RPD3D_Wrapper::cluster_all() ahead
void OPT_RPD::integrate_particle_knn_with_mmesh(Eigen::VectorXd& X,
                                                Eigen::VectorXd& g) {
  printf("[Particle] calling integrate particle knn with mmesh ...\n");
  // needs to make sure mmesh exists
  assert(this->_mmesh != nullptr);

  // has called OPT_RPD::update_particle_kernel_helper() ahead
  assert(this->params.particle_kernel_sigma != -1);

  // find sphere neighbors
  // std::vector<std::set<int>> site_adj_sites;
  // this->_rpd3d.get_sites_adj_sites(site_adj_sites);
  std::vector<std::set<int>> site_adj_sites;
  this->_rpd3d.get_sites_KNN(site_adj_sites,
                             this->params.particle_kernel_sigma);

  // reset loss
  int num_spheres = X.size() / this->dim;
  std::vector<double> lossParticles_tmp(num_spheres, 0.f);
  // compute K neighbors for each particle i
  auto run_thread_particle_mmesh = [&](int sphere_id) {
    // particle i
    const Vector3& msphere_center = get_sphere_center(X, sphere_id);
    const auto& msphere = this->_all_medial_spheres->at(sphere_id);
    // if deleted, return
    if (msphere.is_deleted) return;
    // do not update corner, stay still
    if (msphere.is_on_corner()) return;

    // Step 1: get valid neighbors j neigh_id
    assert(sphere_id < site_adj_sites.size());
    const auto& valid_msphere_neigh_ids = site_adj_sites.at(sphere_id);
    if (is_debug) {
      printf("sphere_id: %d, ", sphere_id);
      print_set<int>(valid_msphere_neigh_ids, "valid_msphere_neigh_ids");
    }
    if (valid_msphere_neigh_ids.empty()) {
      // may happens for corners and T_4 spheres
      if (is_debug)
        printf("[Particle] sphere %d has no valid neighbors in RPD\n",
               sphere_id);
      return;
    }

    // Step 2: loop all neighbors in RPD
    // for (int j = 0; j < valid_msphere_neigh_ids.size(); j++) {
    // int sphere_neigh_id = valid_msphere_neigh_ids.at(j);
    for (int sphere_neigh_id : valid_msphere_neigh_ids) {
      // particle j
      if (sphere_id == sphere_neigh_id) continue;
      const auto& msphere_neigh =
          this->_all_medial_spheres->at(sphere_neigh_id);
      double kernel = this->params.particle_kernel_sigma;
      double kernel_sq = std::pow(kernel, 2);
      const auto neigh_center = get_sphere_center(X, sphere_neigh_id);
      // get lambda_scale
      double lambda_scale = 1.f;
      // accumulate particle energy
      double length_sq = GEO::Geom::distance2(neigh_center, msphere_center);
      // // v1/v2: use [Zhong 2013]
      // double e_ij =
      //     lambda_scale * std::exp(-1.f * length_sq / (4.f * kernel_sq));
      // v3: use [Witkin 1994]
      double e_ij =
          lambda_scale * std::exp(-1.f * length_sq / (2.f * kernel_sq));
      lossParticles_tmp[sphere_id] += this->params.lambda_particle * e_ij;
      // accumulate gradient, the force applied on particle i by particle j
      // not grad_ij!!!!
      //
      // v1: [Zhong2013] Eqn22
      // Vector3 grad_ji = (neigh_center - msphere_center) * e_ij / (2.f *
      // kernel_sq);
      //
      // v2: we normalize the force for spheres on sheets
      // Vector3 grad_ji = GEO::normalize(neigh_center - msphere_center) *e_ij
      // / (2.f * kernel_sq);
      //
      // v3: use [Witkin 1994]
      Vector3 grad_ji =
          GEO::normalize(neigh_center - msphere_center) * e_ij / kernel_sq;
      //
      g(sphere_id * this->dim + 0) += this->params.lambda_particle * grad_ji[0];
      g(sphere_id * this->dim + 1) += this->params.lambda_particle * grad_ji[1];
      g(sphere_id * this->dim + 2) += this->params.lambda_particle * grad_ji[2];
      g(sphere_id * this->dim + 3) += 0.f;  // no r

      if (is_debug)
        printf(
            "[Particle] sphere %d processing knn neighbor %d, length: %f, "
            "kernel: %f, e_ij: %f, e_ij/(2.f*kernel_sq): %f, grad_ji: "
            "(%f,%f,%f), g: (%f,%f,%f) \n",
            sphere_id, sphere_neigh_id, std::sqrt(length_sq), kernel, e_ij,
            e_ij / (2.f * kernel_sq), grad_ji[0], grad_ji[1], grad_ji[2],
            g(sphere_id * this->dim + 0), g(sphere_id * this->dim + 1),
            g(sphere_id * this->dim + 2));
    }

    // NOTE:
    // if g is too small, then no need to project, no need to move at all
    // otherwise g.norm() may be NAN, affects the LBFGS
    if (std::abs(g(sphere_id * this->dim + 0)) <= SCALAR_ZERO_4 &&
        std::abs(g(sphere_id * this->dim + 1)) <= SCALAR_ZERO_4 &&
        std::abs(g(sphere_id * this->dim + 2)) <= SCALAR_ZERO_4) {
      g(sphere_id * this->dim + 0) = 0.f;
      g(sphere_id * this->dim + 1) = 0.f;
      g(sphere_id * this->dim + 2) = 0.f;
      g(sphere_id * this->dim + 3) = 0.f;
      return;
    }

    // Step 3: project g
    Vector3 g_tmp(g(sphere_id * this->dim + 0), g(sphere_id * this->dim + 1),
                  g(sphere_id * this->dim + 2));
    assert(!this->_particle_projection_cache.empty());
    project_particle_gradient(this->_particle_projection_cache[sphere_id],
                              g_tmp, is_debug);
    g(sphere_id * this->dim + 0) = g_tmp[0];
    g(sphere_id * this->dim + 1) = g_tmp[1];
    g(sphere_id * this->dim + 2) = g_tmp[2];
    g(sphere_id * this->dim + 3) = 0.f;
  };

  GEO::parallel_for(0, num_spheres,
                    [&](int i) { run_thread_particle_mmesh(i); });
  // for (int i = 0; i < num_spheres; i++) run_thread_particle_mmesh(i);

  this->loss.lossParticle = 0.f;
  for (int sphere_id = 0; sphere_id < num_spheres; sphere_id++)
    this->loss.lossParticle += lossParticles_tmp[sphere_id];
  this->loss.lossTotal += this->loss.lossParticle;
}

void OPT_RPD::run_particle_lbfgs_globally_KNN(int& num_itr_global,
                                              BGAL::_LBFGS& lbfgs,
                                              const std::string folder_path,
                                              const std::string name_no_ext) {
  assert(this->params.particle_kernel_sigma != -1);
  // LBFGS function
  std::function<double(Eigen::VectorXd&, Eigen::VectorXd&, int&)> fun_lbfgs =
      [&](Eigen::VectorXd& X, Eigen::VectorXd& g, int& rpd_itr_local) {
        const clock_t start_t = clock();
        // reset
        this->reset_loss();  // set losses to 0
        g.setZero();         // set gradients to 0

        //------------------------------------------------------------------
        // update sphere radii, since particle only cares about the center
        this->update_mspheres_radii(*this->_sf_mesh);
        // save spheres to debug
        // save_spheres_itr_file(X, folder_path, name_no_ext,
        //                       RUN_TYPE::PARTICLE_MMESH, num_itr_global);

        this->calculate_spheres_samples(X);
        this->update_particle_projection_gradient_cache(false /*is_debug*/);

        //------------------------------------------------------------------
        // integral inner-Particle energy
        // has called OPT_RPD::update_particle_kernel_helper() ahead
        assert(this->params.particle_kernel_sigma != -1);
        this->integrate_particle_KNN(X, g);
        this->_grad_itr = g;

        // // print g
        // double g_norm = 0.0f;
        // for (int i = 0; i < g.size() / this->dim; i++) {
        //   const auto& msphere = this->_all_medial_spheres->at(i);
        //   // if (this->is_debug)
        //   printf("[Particle] sphere %d, type %d, g: (%f,%f,%f,%f)\n",
        //          msphere.id, msphere.type, g(i * 4), g(i * 4 + 1),
        //          g(i * 4 + 2), g(i * 4 + 3));
        //   for (int j = 0; j < 4; j++) g_norm += g(i * 4 + j) * g(i * 4 +
        //   j); printf("[Particle] g_norm: %f\n", g_norm);
        // }
        // g_norm = std::sqrt(g_norm);
        std::cout << "[PARTICLE_MMESH SUM] Particle gradient norm " << g.norm()
                  << std::endl;

        std::cout << std::setprecision(7) << "[PARTICLE_MMESH SUM] lossTotal: "
                  << this->params.alpha_print * this->loss.lossTotal
                  << " , lossParticle: "
                  << this->params.alpha_print * this->loss.lossParticle
                  << std::endl;

        std::cout << "[Timer] Integrate PARTICLE_MMESH itr" << rpd_itr_local
                  << " took " << (float)(clock() - start_t) / CLOCKS_PER_SEC
                  << " seconds" << std::endl;

        // update iteration counts
        num_itr_global++;
        rpd_itr_local++;
        return this->loss.lossTotal;
      };

  printf("----------------------------------- PARTICLE_MMESH start!!!!\n");
  lbfgs.minimize(fun_lbfgs, this->_spheres_itr);
  printf("----------------------------------- PARTICLE_MMESH done!!!!\n");
}

void OPT_RPD::run_particle_lbfgs_globally_mmesh(int& num_itr_global,
                                                BGAL::_LBFGS& lbfgs,
                                                const std::string folder_path,
                                                const std::string name_no_ext) {
  assert(this->params.particle_kernel_sigma != -1);
  // LBFGS function
  std::function<double(Eigen::VectorXd&, Eigen::VectorXd&, int&)> fun_lbfgs =
      [&](Eigen::VectorXd& X, Eigen::VectorXd& g, int& rpd_itr_local) {
        const clock_t start_t = clock();
        // reset
        this->reset_loss();  // set losses to 0
        g.setZero();         // set gradients to 0

        //------------------------------------------------------------------
        // update sphere radii, since particle only cares about the center
        this->update_mspheres_radii(*this->_sf_mesh);
        // save spheres to debug
        // save_spheres_itr_file(X, folder_path, name_no_ext,
        //                       RUN_TYPE::PARTICLE_MMESH, num_itr_global);

        //------------------------------------------------------------------
        // re-compute RPD
        // 1. for handling T1 sphere if any exapnd the samples using one
        //    neighbor's samples run in multi-thread with is_expand_pcell =
        //    true.
        // 2. will update this->_all_medial_spheres from this->_spheres_itr
        this->calculate_rpd_rvd_samples(X);
        this->update_particle_projection_gradient_cache(false /*is_debug*/);

        //------------------------------------------------------------------
        // this->calculate_only_medges_and_mark_same_sheet();
        // integral inner-Particle energy
        // has called OPT_RPD::update_particle_kernel_helper() ahead
        assert(this->params.particle_kernel_sigma != -1);
        this->integrate_particle_knn_with_mmesh(X, g);
        this->_grad_itr = g;

        // // print g
        // double g_norm = 0.0f;
        // for (int i = 0; i < g.size() / this->dim; i++) {
        //   const auto& msphere = this->_all_medial_spheres->at(i);
        //   // if (this->is_debug)
        //   printf("[Particle] sphere %d, type %d, g: (%f,%f,%f,%f)\n",
        //          msphere.id, msphere.type, g(i * 4), g(i * 4 + 1),
        //          g(i * 4 + 2), g(i * 4 + 3));
        //   for (int j = 0; j < 4; j++) g_norm += g(i * 4 + j) * g(i * 4 +
        //   j); printf("[Particle] g_norm: %f\n", g_norm);
        // }
        // g_norm = std::sqrt(g_norm);
        std::cout << "[PARTICLE_MMESH SUM] Particle gradient norm " << g.norm()
                  << std::endl;

        std::cout << std::setprecision(7) << "[PARTICLE_MMESH SUM] lossTotal: "
                  << this->params.alpha_print * this->loss.lossTotal
                  << " , lossParticle: "
                  << this->params.alpha_print * this->loss.lossParticle
                  << std::endl;

        std::cout << "[Timer] Integrate PARTICLE_MMESH itr" << rpd_itr_local
                  << " took " << (float)(clock() - start_t) / CLOCKS_PER_SEC
                  << " seconds" << std::endl;

        // update iteration counts
        num_itr_global++;
        rpd_itr_local++;
        return this->loss.lossTotal;
      };

  printf("----------------------------------- PARTICLE_MMESH start!!!!\n");
  lbfgs.minimize(fun_lbfgs, this->_spheres_itr);
  printf("----------------------------------- PARTICLE_MMESH done!!!!\n");
}

////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////
void OPT_RPD::calculate_proj_SQEM_KNN(int& itr_cnt) {
  const clock_t start_t = clock();
  std::string folder_path = get_save_folder_path(this->name_no_ext);
  assert(this->_all_medial_spheres != nullptr);
  assert(!this->_all_medial_spheres->empty());
  //------------------------------------------------------------------
  // run RPD inside the function
  // will handle the sync of
  // this->_spheres_itr and this->_all_medial_spheres inside
  this->project_sheet_spheres_SQEM(this->_spheres_itr);

  this->load_spheres_from_all();
  // save_spheres_itr_file(this->_spheres_itr, folder_path, this->name_no_ext,
  //                       RUN_TYPE::PROJ, itr_cnt++);

  std::cout << "[CAL_PROJ_KNN] itr" << itr_cnt << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

void OPT_RPD::calculate_proj_SQEM(int& itr_cnt) {
  const clock_t start_t = clock();
  std::string folder_path = get_save_folder_path(this->name_no_ext);
  assert(this->_all_medial_spheres != nullptr);
  assert(!this->_all_medial_spheres->empty());
  if (this->_spheres_itr.size() == 0) this->load_spheres_from_all();

  //------------------------------------------------------------------
  // run RPD inside the function
  this->project_particle_spheres_SQEM(this->_spheres_itr);

  this->load_spheres_from_all();
  // save_spheres_itr_file(this->_spheres_itr, folder_path, this->name_no_ext,
  //                       RUN_TYPE::PROJ, itr_cnt++);

  std::cout << "[CAL_PROJ] itr" << itr_cnt << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

void OPT_RPD::calculate_proj(int& itr_cnt, const PROJ_TYPE& proj_type) {
  const clock_t start_t = clock();
  std::string folder_path = get_save_folder_path(this->name_no_ext);
  assert(this->_all_medial_spheres != nullptr);
  assert(!this->_all_medial_spheres->empty());
  if (this->_spheres_itr.size() == 0 ||
      this->_spheres_itr.size() < this->_all_medial_spheres->size())
    this->load_spheres_from_all();

  // directly update this->_all_medial_spheres
  this->run_spheres_projection(proj_type);

  this->load_spheres_from_all();
  // save_spheres_itr_file(this->_spheres_itr, folder_path, this->name_no_ext,
  //                       RUN_TYPE::PROJ, itr_cnt++);

  std::cout << "[CAL_PROJ] itr" << itr_cnt << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

void OPT_RPD::calculate_particle_lbfgs_globally(int& num_itr_global,
                                                const RUN_TYPE& run_type,
                                                const int num_particle_itr) {
  const clock_t start_t = clock();
  std::string folder_path = get_save_folder_path(this->name_no_ext);
  ///////////////////////////////////////////////////
  // Main logic
  //
  // LBFGS
  BGAL::_LBFGS lbfgs;
  lbfgs._parameter.max_linearsearch = 30;
  lbfgs._parameter.max_iteration = 5;
  lbfgs._parameter.is_show = true;
  lbfgs._parameter.rpd_itr = 0;
  if (num_particle_itr != -1) {
    printf("update lbfgs._parameter.max_iteration to: %d\n", num_particle_itr);
    lbfgs._parameter.max_iteration = num_particle_itr;
  }

  // save this->_all_medial_spheres to this->_spheres_itr
  this->load_spheres_from_all();
  this->set_lbfgs_params();  // will reset particle_kernel_sigma
  this->_is_two_sphere_same_type_cache.clear();

  // reset kernel
  // let optimize based on the new computed medial mesh
  if (this->params.particle_kernel_sigma == -1) update_particle_kernel_helper();

  // for debug
  // save old spheres before iterations
  this->load_sphere_centers_old_from_all();

  if (run_type == RUN_TYPE::PARTICLE_MMESH) {
    this->run_particle_lbfgs_globally_mmesh(num_itr_global, lbfgs, folder_path,
                                            this->name_no_ext);
  } else if (run_type == RUN_TYPE::PARTICLE_KNN) {
    this->run_particle_lbfgs_globally_KNN(num_itr_global, lbfgs, folder_path,
                                          this->name_no_ext);
  } else {
    printf("Unkown run_type: %d\n", run_type);
    assert(false);
  }

  // save back to _all_medial_spheres
  this->save_spheres_to_all();
  // save_spheres_itr_file(this->_spheres_itr, folder_path, this->name_no_ext,
  //                       RUN_TYPE::PARTICLE_MMESH, num_itr_global);

  std::cout << "[CAL_PARTICLE] itr" << num_particle_itr << " took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}
