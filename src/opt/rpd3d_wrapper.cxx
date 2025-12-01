#include "rpd3d_wrapper.h"

#include <chrono>

#include "shrinking.h"
#include "voronoi.h"
#include "voronoi_defs.h"

using timer = std::chrono::system_clock;

RPD3D_Wrapper::~RPD3D_Wrapper() { this->~RPD3D_GPU(); }

RPD3D_Wrapper::RPD3D_Wrapper(const TetMesh* _tet_mesh,
                             const SurfaceMesh* _sf_mesh,
                             const Parameter* _params,
                             std::vector<MedialSphere>* _all_medial_spheres) {
  init(_tet_mesh, _sf_mesh, _params, _all_medial_spheres);
}

void RPD3D_Wrapper::init(const TetMesh* _tet_mesh, const SurfaceMesh* _sf_mesh,
                         const Parameter* _params,
                         std::vector<MedialSphere>* _all_medial_spheres) {
  this->tet_mesh = _tet_mesh;
  this->sf_mesh = _sf_mesh;
  this->params = _params;
  this->all_medial_spheres = _all_medial_spheres;
  num_itr_rpd = 0;
  n_site = -1;
  site_k = -1;
  // BVH for pcell_samples
  load_bvh_triangles_from_mesh(*_sf_mesh);
  // from dist2meshAABBTree.h
  build_AABBTree_CPU(this->aabb_tri_vertices, this->aabb_triangles,
                     this->aabb_tree);
  printf("[Particle] build aabb_tree %d for %d triangles\n",
         this->aabb_tree.size(), this->aabb_triangles.size());
  // // print aabb_tree
  // for (int i = 0; i < this->aabb_tree.size(); i++) {
  //   printf("aabb_tree %d: min(%f,%f,%f) max(%f,%f,%f)\n", i,
  //          this->aabb_tree[i].box.min.x, this->aabb_tree[i].box.min.y,
  //          this->aabb_tree[i].box.min.z, this->aabb_tree[i].box.max.x,
  //          this->aabb_tree[i].box.max.y, this->aabb_tree[i].box.max.z);
  // }
}

void RPD3D_Wrapper::load_spheres(std::vector<float>& _spheres, bool is_rpd) {
  _spheres.clear();
  for (int i = 0; i < this->all_medial_spheres->size(); i++) {
    const auto& msphere = this->all_medial_spheres->at(i);
    _spheres.push_back(msphere.center[0]);
    _spheres.push_back(msphere.center[1]);
    _spheres.push_back(msphere.center[2]);
    if (is_rpd) {
      _spheres.push_back(msphere.radius);
    } else {
      _spheres.push_back(1.f);
    }
  }
}

void RPD3D_Wrapper::load_bvh_triangles_from_mesh(const GEO::Mesh& mesh) {
  this->aabb_tree.clear();
  this->aabb_tri_vertices.clear();
  this->aabb_tri_vertices.resize(mesh.vertices.nb());
  for (size_t i = 0; i < this->aabb_tri_vertices.size(); i++) {
    Vector3 v = mesh.vertices.point(i);
    this->aabb_tri_vertices[i][0] = v[0];
    this->aabb_tri_vertices[i][1] = v[1];
    this->aabb_tri_vertices[i][2] = v[2];
  }

  this->aabb_triangles.clear();
  this->aabb_triangles.resize(mesh.facets.nb());
  for (size_t i = 0; i < this->aabb_triangles.size(); i++) {
    this->aabb_triangles[i].v0 = mesh.facets.vertex(i, 0);
    this->aabb_triangles[i].v1 = mesh.facets.vertex(i, 1);
    this->aabb_triangles[i].v2 = mesh.facets.vertex(i, 2);
  }
}

void RPD3D_Wrapper::reset_radii(std::vector<float>& _spheres, double radius) {
  int dim = 4;
  for (int i = 0; i < _spheres.size() / dim; i++) {
    _spheres[i * dim + 3] = radius;
  }
}

const std::vector<float>& RPD3D_Wrapper::get_sites() const {
  return this->site;
}

void RPD3D_Wrapper::remove_reset_inactive_clusters(
    std::vector<SFCluster>& clusters) {
  if (clusters.empty()) return;
  clusters.erase(
      std::remove_if(clusters.begin(), clusters.end(),
                     [](const SFCluster& c) { return !c.is_active; }),
      clusters.end());
  // reset cluster id
  int num_clusters_after = 0;
  for (auto& c : clusters) c.id = num_clusters_after++;
}

static int get_first_active_cluster_id(const std::vector<SFCluster>& clusters,
                                       int cluster_id) {
  assert(cluster_id > -1 && cluster_id < clusters.size());
  int first_active_cluster_id = cluster_id;
  while (!clusters.at(first_active_cluster_id).is_active) {
    first_active_cluster_id =
        clusters.at(first_active_cluster_id).merge_to_cluters_id;
    assert(first_active_cluster_id > -1 &&
           first_active_cluster_id < clusters.size());
  }
  return first_active_cluster_id;
}

// taking SphereType::T_1_INF and SphereType::T_2_INF into account
int RPD3D_Wrapper::get_msphere_num_clusters(const int sphere_id) {
  assert(!this->sphere_clusters.empty());
  assert(sphere_id > -1 && sphere_id < this->sphere_clusters.size());
  int num_clusters = 0;
  const auto& msphere_clusters = this->sphere_clusters.at(sphere_id);
  const auto& msphere = this->all_medial_spheres->at(sphere_id);
  for (const auto& cluster : msphere_clusters) {
    assert(cluster.is_active);
    num_clusters++;
    // if SphereType::T_1_INF || SphereType::T_2_INF, add one more cluster
    if (msphere.type == SphereType::T_1_INF &&
        is_inf_more_than_T1(cluster.is_normal_inf))
      num_clusters++;
    else if (msphere.type == SphereType::T_2_INF &&
             is_inf_more_than_T2(cluster.is_normal_inf))
      num_clusters++;
  }
  return num_clusters;
}

// used for clusters from:
// 1. RPD3D_Wrapper::medge_clusters
// 1. RPD3D_Wrapper::mface_clusters
int RPD3D_Wrapper::get_num_clusters(const std::vector<SFCluster>& clusters) {
  // assert(!clusters.empty());
  if (clusters.empty()) return 0;
  int num_clusters = 0;
  for (const auto& c : clusters) {
    assert(c.is_active);
    int pid_sphere_id = c.pid_sphere_id;
    const auto& msphere = this->all_medial_spheres->at(pid_sphere_id);
    num_clusters++;
    // more strict, only conside the case when one sphere is INF
    if (msphere.type == SphereType::T_2_INF &&
        is_inf_more_than_T2(c.is_normal_inf))
      num_clusters++;
    else if (msphere.type == SphereType::T_1_INF &&
             is_inf_more_than_T1(c.is_normal_inf))
      num_clusters++;
  }
  return num_clusters;
}

void RPD3D_Wrapper::sort_spheres_ids_from_clusters(
    std::vector<int>& sphere_ids) {
  if (sphere_ids.empty()) return;
  auto is_a_smaller_b = [&](const int& aid, const int& bid) {
    int a_num_clusters = this->get_msphere_num_clusters(aid);
    int b_num_clusters = this->get_msphere_num_clusters(bid);
    if (a_num_clusters < b_num_clusters) return true;
    if (a_num_clusters > b_num_clusters) return false;
    // a and b have equal num of clusters
    const auto& A = this->all_medial_spheres->at(aid);
    const auto& B = this->all_medial_spheres->at(bid);
    // 1. let NON-corners front
    if (A.is_on_corner() && !B.is_on_corner()) return false;
    if (!A.is_on_corner() && B.is_on_corner()) return true;
    // 2. let non-SE sphere be at front
    if (A.is_on_se() && !B.is_on_se()) return false;
    if (!A.is_on_se() && B.is_on_se()) return true;
    // 3. let NON-INF front
    if (A.is_on_inf() && !B.is_on_inf()) return false;
    if (!A.is_on_inf() && B.is_on_inf()) return true;
    // 4. let smaller id front
    return aid < bid;
  };

  std::sort(sphere_ids.begin(), sphere_ids.end(), is_a_smaller_b);
  // print_set(sphere_ids, "sorted_sphere_ids");
}

#include "io_cuda.h"
void RPD3D_Wrapper::get_pcell_in_bfaces() {
  assert(!this->powercells.empty());
  // reset
  this->pcell_bface_points.clear();
  this->pcell_bfaces.clear();
  this->pcell_bface_sphere_ids.clear();
  this->pcell_bface_adj_sphere_ids.clear();

  std::vector<cfloat3> voro_points;
  std::vector<std::vector<unsigned>> all_voro_faces;
  std::vector<int> voro_faces_sites, voro_faces_adj_sites;
  for (auto& cc_trans : this->powercells) {
    if (!cc_trans.is_active_updated) cc_trans.reload_active();
    // printf("all_voro_faces size: %zu\n", all_voro_faces.size());
    // printf("voro_faces_sites size: %zu\n", voro_faces_sites.size());
    get_one_convex_cell_faces_const(
        cc_trans, voro_points, all_voro_faces, voro_faces_sites,
        voro_faces_adj_sites, true /*is_triangle*/, this->sf_mesh->facets.nb(),
        true /*is_boundary_only*/);
  }
  // save to RPD3D_Wrapper members
  this->pcell_bface_points = voro_points;
  for (const auto& voro_face : all_voro_faces) {
    assert(voro_face.size() == 3);
    this->pcell_bfaces.push_back({{voro_face[0], voro_face[1], voro_face[2]}});
  }
  this->pcell_bface_sphere_ids = voro_faces_sites;
  this->pcell_bface_adj_sphere_ids = voro_faces_adj_sites;
  // printf(
  //     "[RPD3D] pcell_bfaces size: %zu, "
  //     "this->pcell_bface_sphere_ids: %zu\n",
  //     this->pcell_bfaces.size(), this->pcell_bface_sphere_ids.size());
}

// [no use]
void RPD3D_Wrapper::get_pcell_bface_areas() {
  assert(!this->powercells.empty());
  assert(!this->pcell_bface_points.empty());
  assert(!this->pcell_bfaces.empty());
  assert(!this->pcell_bface_sphere_ids.empty());
  assert(!this->pcell_bface_adj_sphere_ids.empty());
  // reset [no use]
  this->sphere_pcell_adj_bface_area.clear();
  this->sphere_pcell_adj_bface_area.resize(this->all_medial_spheres->size());
  this->sphere_pcell_bface_area.clear();
  this->sphere_pcell_bface_area.resize(this->all_medial_spheres->size());

  auto get_triangle_area = [&](const Vector3& v1, const Vector3& v2,
                               const Vector3& v3) {
    Vector3 a = v2 - v1;
    Vector3 b = v3 - v1;
    return 0.5 * GEO::length(GEO::cross(a, b));
  };

  // not thread safe
  for (int bface_id = 0; bface_id < pcell_bfaces.size(); bface_id++) {
    int sphere_id = this->pcell_bface_sphere_ids.at(bface_id);
    int adj_sphere_id = this->pcell_bface_adj_sphere_ids.at(bface_id);
    aint3 one_fs = this->pcell_bfaces.at(bface_id);  // <v1,v2,v3>
    std::array<Vector3, 3> fs_vs;
    for (int i = 0; i < 3; i++) {
      fs_vs[i] = Vector3(this->pcell_bface_points.at(one_fs[i]).x,
                         this->pcell_bface_points.at(one_fs[i]).y,
                         this->pcell_bface_points.at(one_fs[i]).z);
    }
    double area = get_triangle_area(fs_vs[0], fs_vs[1], fs_vs[2]);
    sphere_pcell_bface_area[sphere_id] += area;
    auto& one_pcell_face_area = this->sphere_pcell_adj_bface_area.at(sphere_id);
    if (one_pcell_face_area.find(adj_sphere_id) == one_pcell_face_area.end())
      one_pcell_face_area[adj_sphere_id] = area;
    else
      one_pcell_face_area[adj_sphere_id] += area;
  }
}

// [no use]
void RPD3D_Wrapper::get_pcell_dual_faces() {
  this->pcell_dual_faces.clear();
  assert(all_medial_spheres != nullptr);
  for (auto& msphere : *all_medial_spheres) {
    // if (msphere.is_deleted) continue;
    if (msphere.pcell.cell_ids.empty()) continue;
    // [neigh_id_min, neigh_id_max] -> { set of cell_ids in one edge CC }
    for (const auto& pair : msphere.pcell.edge_cc_cells) {
      const aint2& neigh_min_max = pair.first;
      aint3 mf = {{msphere.id, neigh_min_max[0], neigh_min_max[1]}};
      std::sort(mf.begin(), mf.end());
      // each edgeCC dual to one medial face
      this->pcell_dual_faces.insert(mf);
    }
  }
  printf("[RPD3D] pcell_dual_faces size: %zu\n", this->pcell_dual_faces.size());
}

void RPD3D_Wrapper::get_one_pcell_neighbors(
    const int sphere_id, std::set<int>& one_pcell_neighbors) {
  one_pcell_neighbors.clear();
  const auto& msphere = this->all_medial_spheres->at(sphere_id);
  if (msphere.is_deleted) return;
  if (msphere.pcell.cell_ids.empty()) return;
  // dual to medial edges
  for (const auto& pair : msphere.pcell.facet_cc_cells) {
    const int& seed_neigh_id = pair.first;
    one_pcell_neighbors.insert(seed_neigh_id);
  }
}

double RPD3D_Wrapper::get_avg_volume(bool is_debug) const {
  assert(!site_cell_vol.empty());
  if (is_debug) printf("[RPD3D] volumes: ");
  double avg_vol = 0;
  for (int i = 0; i < site_cell_vol.size(); i++) {
    avg_vol += site_cell_vol.at(i);
    if (is_debug && i % 10 == 0) printf("%f, ", i, site_cell_vol.at(i));
  }
  avg_vol /= site_cell_vol.size();

  if (is_debug) printf("[RPD3D] avg_vol: %f\n", avg_vol);

  return avg_vol;
}

void get_proj_k_sf_fids(const SurfaceMesh& sf_mesh, const int k,
                        const adouble3& point, std::set<int>& proj_fids) {
  Vector3 p(point[0], point[1], point[2]);
  int fid = sf_mesh.aabb_wrapper.get_nearest_face_sf(p);
  get_k_ring_neighbors_no_cross(sf_mesh, sf_mesh.fe_sf_fs_pairs, fid, k,
                                proj_fids, true /*is_clear_cur*/,
                                false /*is_debug*/);
}

void merge_set_group(std::vector<std::set<int>>& cur_groups,
                     std::set<int>& given_set) {
  std::vector<int> inter;
  bool is_merged = false;
  for (auto& one_group : cur_groups) {
    set_intersection(one_group, given_set, inter);
    if (inter.empty()) continue;
    one_group.insert(given_set.begin(), given_set.end());
    is_merged = true;
    break;
  }
  if (is_merged) return;

  // merge
  cur_groups.push_back(given_set);
}

// use KNN
#include "knn.h"
void RPD3D_Wrapper::get_sites_KNN(std::vector<std::set<int>>& site_adj_sites,
                                  const double search_radius) {
  int K = 10;
  int num_spheres = this->all_medial_spheres->size();
  std::vector<Vector3> msphere_centers(num_spheres);
  std::vector<bool> is_point_on_extf(num_spheres, false);
  for (int i = 0; i < num_spheres; i++) {
    const auto& msphere = this->all_medial_spheres->at(i);
    msphere_centers[i] = msphere.center;
    is_point_on_extf[i] = msphere.is_on_extf();
  }

  // compute knn
  std::vector<std::vector<uint32_t>> mspheres_knn_indices;
  std::vector<std::vector<double>> mspheres_knn_dists;
  if (search_radius == -1) {
    // // compute knn given fixed K
    knn_search(msphere_centers, K, mspheres_knn_indices, mspheres_knn_dists,
               false /*is_debug*/);
  } else {
    // compute knn given radius
    double search_radius_sq = std::pow(search_radius, 2);
    double search_radius_se_sq = std::pow(search_radius, 2);
    knn_search_radius_sq_or_k(msphere_centers, is_point_on_extf,
                              search_radius_sq, search_radius_se_sq, K,
                              mspheres_knn_indices, mspheres_knn_dists,
                              false /*is_debug*/);
  }

  site_adj_sites.clear();
  site_adj_sites.resize(this->all_medial_spheres->size());
  assert(mspheres_knn_indices.size() == site_adj_sites.size());
  for (int i = 0; i < mspheres_knn_indices.size(); i++) {
    site_adj_sites[i].insert(mspheres_knn_indices[i].begin(),
                             mspheres_knn_indices[i].end());
  }
}

// use ConvexCellHost
void RPD3D_Wrapper::get_sites_adj_sites(
    std::vector<std::set<int>>& site_adj_sites) {
  site_adj_sites.clear();
  site_adj_sites.resize(this->all_medial_spheres->size());
  auto run_one_convex_cell_adj_sites = [&](const ConvexCellHost& cc_trans,
                                           std::set<int>& adj_sites) {
    // some clipping planes may not exist in tri but
    // we still sotre it, here is to filter those planes
    // if (!cc_trans.is_active_updated) cc_trans.reload_active();
    assert(cc_trans.is_active_updated);
    const std::vector<int>& active_clipping_planes =
        cc_trans.active_clipping_planes;

    FOR(plane, cc_trans.nb_p) {
      if (active_clipping_planes[plane] <= 0) continue;
      cint2 hp = cc_trans.clip_id2_const(plane);
      // only store halfplane
      if (hp.y == -1) continue;
      // do not clean adj_sites ahead
      adj_sites.insert(hp.x == cc_trans.voro_id ? hp.y : hp.x);
    }
  };

  for (auto& cc_trans : this->powercells) {
    assert(cc_trans.voro_id < this->all_medial_spheres->size());
    if (!cc_trans.is_active_updated) cc_trans.reload_active();
    run_one_convex_cell_adj_sites(cc_trans,
                                  site_adj_sites.at(cc_trans.voro_id));
  }
}

#include <random>
// Generates a uniformly random point on the unit sphere using Marsaglia method
Vector3 uniform_sample_sphere(std::mt19937& gen,
                              std::uniform_real_distribution<>& dist) {
  double x1, x2, s;
  do {
    x1 = 2.0 * dist(gen) - 1.0;
    x2 = 2.0 * dist(gen) - 1.0;
    s = x1 * x1 + x2 * x2;
  } while (s >= 1.0 || s == 0.0);

  double z = 1.0 - 2.0 * s;
  double factor = 2.0 * std::sqrt(1.0 - s);

  double x = x1 * factor;
  double y = x2 * factor;

  return Vector3(x, y, z);
}

// classify each sample direction (from sphere's  center)
// based on all neighbor directions (from sphere's center)
int classify_sample_by_neigh_dirs(const Vector3& sampleDir,
                                  const std::vector<v2int>& neighDirs) {
  if (neighDirs.empty()) return -1;
  // greater dot product, smaller the directions
  double maxDot = -std::numeric_limits<double>::infinity();
  int closestIndex = -1;
  for (size_t i = 0; i < neighDirs.size(); ++i) {
    double dotProduct = GEO::dot(sampleDir, neighDirs[i].first);
    if (dotProduct > maxDot) {
      maxDot = dotProduct;
      closestIndex = i;
    }
  }
  assert(closestIndex != -1);
  return neighDirs[closestIndex].second;
}

// NOT use pcell, use a small sphere around center as grid
#include <cmath>
void RPD3D_Wrapper::collect_nearby_sphere_samples_helper(bool is_sample_rpd,
                                                         bool is_debug) {
  int num_samples = 100;  // fixed for every sphere
  std::vector<std::set<int>> site_adj_sites;
  if (is_sample_rpd)
    this->get_sites_adj_sites(site_adj_sites);
  else
    this->get_sites_KNN(site_adj_sites);
  assert(site_adj_sites.size() == this->all_medial_spheres->size());

  this->sphere_pcell_samples.clear();
  this->sphere_pcell_samples.resize(this->all_medial_spheres->size());
  auto run_sphere_thread = [&](const int sphere_id) {
    is_debug = false;
    // if (sphere_id == 332) is_debug = true;

    auto& one_sphere_pcell_samples = this->sphere_pcell_samples.at(sphere_id);
    assert(sphere_id < this->all_medial_spheres->size());
    assert(sphere_id < site_adj_sites.size());
    const auto& msphere = this->all_medial_spheres->at(sphere_id);
    const auto& msphere_neighs_ids = site_adj_sites.at(sphere_id);
    std::vector<v2int> neigh_dirs;                       // for medges
    std::map<int, std::vector<v2int>> neigh_neigh_dirs;  // for mfaces
    std::set<int> inter;
    // get smallest distance from msphere to neighbors
    double small_radius = msphere_neighs_ids.empty()  // handle deleted spheres
                              ? msphere.radius
                              : DBL_MAX;  // also handle extf spheres
    for (const int nsid : msphere_neighs_ids) {
      if (nsid == sphere_id) continue;
      assert(nsid > -1 && nsid < this->all_medial_spheres->size());
      const auto& msphere_neigh = this->all_medial_spheres->at(nsid);
      neigh_dirs.push_back(
          {GEO::normalize(msphere_neigh.center - msphere.center), nsid});
      small_radius = std::min(
          GEO::length(msphere.center - msphere_neigh.center), small_radius);
      // for mface
      // find neighbor of (sphere_id, nsid)
      auto nsid_neighs_ids = site_adj_sites.at(nsid);  // copy
      nsid_neighs_ids.erase(nsid);
      nsid_neighs_ids.erase(sphere_id);
      set_intersection<int>(msphere_neighs_ids, nsid_neighs_ids, inter);
      for (const auto& n_nsid : inter) {
        const auto& msphere_neigh_neigh = this->all_medial_spheres->at(n_nsid);
        neigh_neigh_dirs[nsid].push_back(
            {GEO::normalize(msphere_neigh_neigh.center - msphere.center),
             n_nsid});
      }
    }
    small_radius *= 0.5;

    //
    bool deterministic = true;
    std::mt19937 gen(deterministic ? RAN_SEED : std::random_device{}());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (int i = 0; i < num_samples; ++i) {
      Vector3 dir = uniform_sample_sphere(gen, dist);
      Vector3 sample = msphere.center + dir * small_radius;
      PCELL_SAMPLE one_sample(sphere_id);
      one_sample.pos = sample;
      one_sample.weight = 1.f;
      // store the adj_sphere_ids by classifying
      // for medge neighbors
      int nidx = classify_sample_by_neigh_dirs(
          GEO::normalize(sample - msphere.center), neigh_dirs);
      if (nidx != -1) {
        one_sample.adj_sphere_ids.insert(nidx);
        // for mface negibhros
        if (neigh_neigh_dirs.find(nidx) != neigh_neigh_dirs.end()) {
          int nidx2 = classify_sample_by_neigh_dirs(
              GEO::normalize(sample - msphere.center),
              neigh_neigh_dirs.at(nidx));
          if (nidx2 != -1) one_sample.adj_sphere_ids.insert(nidx2);
        }
      }
      one_sphere_pcell_samples.push_back(one_sample);

      if (is_debug) {
        printf("sphere %d has sample %d with adj_sphere_ids: ", sphere_id, i);
        print_set<int>(one_sphere_pcell_samples.back().adj_sphere_ids);
      }
    }
  };
  GEO::parallel_for(0, this->all_medial_spheres->size(),
                    [&](int sphere_id) { run_sphere_thread(sphere_id); });
  // for (int i = 0; i < this->all_medial_spheres->size(); i++) {
  //   run_sphere_thread(i);
  // }
}

// helper function for update_pcell_samples_cuda()
// run in multi-thread
#include "Integral.h"
void RPD3D_Wrapper::collect_pcell_sphere_samples_helper(bool is_debug) {
  const clock_t start_t = clock();
  // update:
  // - RPD3D_Wrapper::pcell_bface_points
  // - RPD3D_Wrapper::pcell_bfaces
  // - RPD3D_Wrapper::pcell_bface_sphere_ids
  // - RPD3D_Wrapper::pcell_bface_adj_sphere_ids
  this->get_pcell_in_bfaces();
  assert(!this->pcell_bfaces.empty());

  ///////////////////
  // tmp storage for RPD3D_Wrapper::sphere_pcell_samples
  // used for multi-thread
  std::vector<std::vector<PCELL_SAMPLE>> pcell_bface_samples;
  pcell_bface_samples.resize(this->pcell_bfaces.size());
  auto run_bface_thread = [&](int bfid) {
    // <v1,v2,v3>
    aint3 one_fs = this->pcell_bfaces.at(bfid);
    std::array<adouble3, 3> fs_vs;
    for (int i = 0; i < 3; i++) {
      fs_vs[i] = {{this->pcell_bface_points.at(one_fs[i]).x,
                   this->pcell_bface_points.at(one_fs[i]).y,
                   this->pcell_bface_points.at(one_fs[i]).z}};
    }
    int sphere_id = this->pcell_bface_sphere_ids.at(bfid);
    // // for debug
    // is_debug = false;
    // if (sphere_id == 825) is_debug = true;

    // get samples
    // sample from pcell boundary faces
    std::vector<Eigen::Vector3d> samples_tmp;
    std::vector<double> weights_tmp;
    // try 1:
    // BGAL::Integral::get_triangle_integral_samples<int>(
    //     convert2eigen(fs_vs[0]), convert2eigen(fs_vs[1]),
    //     convert2eigen(fs_vs[2]), samples_tmp, weights_tmp);
    //
    // try 2:
    // ninwang: just add one face center as sample
    //          to make computation faster
    Eigen::Vector3d center =
        1. / 3. *
        (convert2eigen(fs_vs[0]) + convert2eigen(fs_vs[1]) +
         convert2eigen(fs_vs[2]));
    samples_tmp.push_back(center);
    weights_tmp.push_back(1.f);
    // add bface sampled points
    for (int i = 0; i < samples_tmp.size(); i++) {
      PCELL_SAMPLE one_pcell_sample(sphere_id);
      one_pcell_sample.pos = Vector3(samples_tmp.at(i)[0], samples_tmp.at(i)[1],
                                     samples_tmp.at(i)[2]);
      one_pcell_sample.weight = weights_tmp.at(i);
      one_pcell_sample.adj_sphere_ids.insert(
          this->pcell_bface_adj_sphere_ids.at(bfid));
      pcell_bface_samples.at(bfid).push_back(one_pcell_sample);
    }
  };  // end run_bface_thread()
  // for (int bfid = 0; bfid < this->pcell_bfaces.size(); bfid++) {
  //   run_bface_thread(bfid);
  // }
  GEO::parallel_for(0, this->pcell_bfaces.size(),
                    [&](int bfid) { run_bface_thread(bfid); });

  ///////////////////
  // this part is not thread-safe
  // init RPD3D_Wrapper::sphere_pcell_samples
  this->sphere_pcell_samples.clear();
  this->sphere_pcell_samples.resize(this->all_medial_spheres->size());
  std::map<int, std::vector<int>> bface_vid2adj_sphere_ids;
  for (int bfid = 0; bfid < this->pcell_bfaces.size(); bfid++) {
    // 1. gather samples by sphere_id from RPD3D_Wrapper::pcell_bface_samples
    auto& one_bface_samples = pcell_bface_samples.at(bfid);
    int sphere_id = this->pcell_bface_sphere_ids.at(bfid);
    auto& one_sphere_pcell_samples = this->sphere_pcell_samples.at(sphere_id);
    one_sphere_pcell_samples.insert(one_sphere_pcell_samples.end(),
                                    one_bface_samples.begin(),
                                    one_bface_samples.end());
    // 2.1. save boundary vertices of pcell bface
    //    all adjacent sphere ids, update bface_vid2adj_sphere_ids
    int adj_sphere_id = this->pcell_bface_adj_sphere_ids.at(bfid);
    aint3 one_fs = this->pcell_bfaces.at(bfid);  // <v1,v2,v3>
    for (int i = 0; i < 3; i++) {
      if (bface_vid2adj_sphere_ids.find(one_fs[i]) ==
          bface_vid2adj_sphere_ids.end()) {
        // save first element as current sphere_id
        // this sample may NOT be on the bface boundary, but its okay to add
        bface_vid2adj_sphere_ids[one_fs[i]].push_back(sphere_id);
      }
      // update adj_sphere_ids
      bface_vid2adj_sphere_ids[one_fs[i]].push_back(adj_sphere_id);
    }
  }
  // 2.2. add boundary vertices of pcell bface
  //    to RPD3D_Wrapper::sphere_pcell_samples
  for (const auto& pair : bface_vid2adj_sphere_ids) {
    int vid = pair.first;
    // <sphere_id, adj_sphere_id1, adj_sphere_id2, ...>
    std::vector<int> vid_adj_sphere_ids = pair.second;  // copy
    if (vid_adj_sphere_ids.size() < 3) continue;        // no need to add
    int sphere_id = vid_adj_sphere_ids[0];
    // for debug
    // is_debug = false;
    // if (sphere_id == 825) is_debug = true;

    // printf("sphere_id %d has pcell vid %d with", sphere_id, vid);
    // print_vec<int>(vid_adj_sphere_ids, "adj_sphere_ids");

    auto& one_sphere_pcell_samples = this->sphere_pcell_samples.at(sphere_id);
    // add boundary vertices
    PCELL_SAMPLE one_pcell_sample(sphere_id);
    one_pcell_sample.pos = Vector3(this->pcell_bface_points.at(vid).x,
                                   this->pcell_bface_points.at(vid).y,
                                   this->pcell_bface_points.at(vid).z);
    // try 1:
    // // these points has no weight, not sampled for integrals
    // one_pcell_sample.weight = 0.0f;
    // try 2:
    // these points are used for integral
    // if we only sample one center on triangle face
    one_pcell_sample.weight = 1.f;
    // skip the first element = sphere_id
    for (int i = 1; i < vid_adj_sphere_ids.size(); i++)
      one_pcell_sample.adj_sphere_ids.insert(vid_adj_sphere_ids.at(i));
    // update_pcell_sample_projs(one_pcell_sample, is_debug);
    one_sphere_pcell_samples.push_back(one_pcell_sample);
  }

  std::cout << "Collect pcell samples took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

void RPD3D_Wrapper::update_pcell_samples_cuda(const SurfaceMesh& sf_mesh,
                                              bool is_sample_rpd) {
  const clock_t start_t = clock();

  // 1. initialize RPD3D_Wrapper::sphere_pcell_samples
  // run in multi-thread
  if (is_sample_rpd)
    collect_pcell_sphere_samples_helper(is_debug);
  else
    collect_nearby_sphere_samples_helper(true, is_debug);

  assert(!this->sphere_pcell_samples.empty());
  assert(this->sphere_pcell_samples.size() == this->all_medial_spheres->size());

  // 2.1. prepare for CUDA kernel
  // collect query points and size
  assert(!this->aabb_tree.empty());
  assert(!this->aabb_tri_vertices.empty());
  assert(!this->aabb_triangles.empty());
  std::vector<int> sphere_query_size;  // size = #spheres
  std::vector<afloat3> query_points;
  for (int sphere_id = 0; sphere_id < this->sphere_pcell_samples.size();
       sphere_id++) {
    const auto& msphere = this->all_medial_spheres->at(sphere_id);
    const auto& sphere_pcell_samples = this->sphere_pcell_samples.at(sphere_id);
    sphere_query_size.push_back(sphere_pcell_samples.size());
    for (const auto& sample : sphere_pcell_samples) {
      query_points.push_back({sample.pos[0], sample.pos[1], sample.pos[2]});
    }
  }
  assert(sphere_query_size.size() == this->sphere_pcell_samples.size());

  // 2.2. call CUDA kernel
  std::vector<afloat3> proj_points;
  std::vector<int> proj_faceIds;
  std::vector<float> proj_lambda1;
  std::vector<float> proj_lambda2;
  std::vector<bool> is_inside;  // only used when is_sample_rpd == false
  if (is_sample_rpd)
    proj_points_on_triangles_with_lambdas_cuda(
        query_points, this->aabb_tri_vertices, this->aabb_triangles,
        this->aabb_tree, proj_points, proj_faceIds, proj_lambda1, proj_lambda2);
  else
    point_inside_and_project_cuda(query_points, this->aabb_tri_vertices,
                                  this->aabb_triangles, this->aabb_tree,
                                  proj_points, proj_faceIds, proj_lambda1,
                                  proj_lambda2, is_inside);
  assert(proj_points.size() == query_points.size());
  assert(proj_faceIds.size() == query_points.size());
  assert(proj_lambda1.size() == query_points.size());
  assert(proj_lambda2.size() == query_points.size());
  if (!is_sample_rpd) assert(is_inside.size() == query_points.size());

  // get the index of the FIRST point
  auto get_sphere_sample_first_idx = [&](int sphere_id) {
    int proj_points_idx = 0;
    for (int i = 0; i < sphere_id; i++)
      proj_points_idx += sphere_query_size.at(i);
    return proj_points_idx;
  };

  // 3. update RPD3D_Wrapper::sphere_pcell_samples
  auto update_projs = [&](int sphere_id) {
    // get the index of the FIRST point
    int proj_points_idx = get_sphere_sample_first_idx(sphere_id);
    // T1 sphere may have no sample
    if (proj_points_idx == query_points.size()) return;
    assert(proj_points_idx < query_points.size());
    const auto& msphere = this->all_medial_spheres->at(sphere_id);

    // update samples
    auto& one_sphere_pcell_samples = this->sphere_pcell_samples.at(sphere_id);
    // for (auto& one_sample : one_sphere_pcell_samples) {
    for (int id = 0; id < one_sphere_pcell_samples.size();
         id++, proj_points_idx++) {
      auto& one_sample = one_sphere_pcell_samples.at(id);
      one_sample.proj = convert_std2geo(proj_points.at(proj_points_idx));
      one_sample.proj_fid = proj_faceIds.at(proj_points_idx);
      adouble3 lambda = {-1, -1, -1};
      lambda[1] = proj_lambda1.at(proj_points_idx);
      lambda[2] = proj_lambda2.at(proj_points_idx);
      lambda[0] = 1.0 - lambda[1] - lambda[2];
      one_sample.proj_FE_id =
          this->sf_mesh->get_FE_given_fid_and_lambdas_if_any(
              this->tet_mesh->feature_edges, EdgeType::CE, one_sample.proj_fid,
              lambda, false /*is_debug*/);
      one_sample.proj_dist = GEO::length(one_sample.pos - one_sample.proj);
      // Note: pnj should NOT be the normal of surface,
      // it should be the vector from pos to proj.
      // for cases like concave edges, face normal will have a sharp change
      Vector3 pnj = get_direction(one_sample.pos, one_sample.proj);
      // NOTE: for outside sample, we keep them but reverse direction
      if (!is_sample_rpd && !is_inside[proj_points_idx])
        pnj = get_direction(one_sample.proj, one_sample.pos);
      // NOTE: for proj on cc_lines, use center to proj as direction
      if (one_sample.proj_FE_id > -1)
        pnj = get_direction(msphere.center, one_sample.proj);
      // NOTE: this is important for correct normals!!!!
      // when yj_geo already too close to the surface
      // pnj can be either 0 or -normal_sf or wrong,
      // so we use the sf normal instead
      if (one_sample.proj_dist <= SCALAR_1) {  // bbox is 10^3
        pnj = get_mesh_facet_normal(sf_mesh, one_sample.proj_fid);
      }
      one_sample.normal = pnj;

      // printf(
      //     "sphere_id %d, proj_fid %d, proj_FE_id %d, "
      //     "proj_dist %f, proj_lambda1 %f, proj_lambda2 %f\n",
      //     sphere_id, one_sample.proj_fid, one_sample.proj_FE_id,
      //     one_sample.proj_dist, lambda[1], lambda[2]);
    }
  };
  // for (int sphere_id = 0; sphere_id < this->sphere_pcell_samples.size();
  //      sphere_id++) {
  //   update_projs(sphere_id);
  // }
  GEO::parallel_for(0, this->sphere_pcell_samples.size(),
                    [&](int sphere_id) { update_projs(sphere_id); });

  std::cout << "Collect and Update pcell samples took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// helper function for merge_clusters_share_same_fid() and
// merge_sfclusters_using_KNN()
bool is_two_clusters_allow_to_merge(const TetMesh& tet_mesh,
                                    const SurfaceMesh& sf_mesh,
                                    const SFCluster& c1, const SFCluster& c2,
                                    const bool is_debug) {
#if FEATURE_ADJUST_SPHERE_TYPE_BY_CE
  // only merge clusters with common ce_line_ids if both exists
  // except for ce_line_ids listed in TetMesh::allow_to_merge_ce_line_ids
  bool is_allow_to_merge = true;
  std::set<int> c2_allow_merge_tmp;

  // when is_strict_merge is true, only merge clusters:
  // 1. both ce_line_ids are empty
  // 2. both ce_line_ids exists and share the common ce_line_ids
  if (is_strict_merge && ((c1.ce_line_ids.empty() && !c2.ce_line_ids.empty()) ||
                          (!c1.ce_line_ids.empty() && c2.ce_line_ids.empty())))
    return false;

  // allow to merge
  if (c1.ce_line_ids.empty() || c2.ce_line_ids.empty())
    return is_allow_to_merge;

  // only checking the first ce_line_id of c1
  int c1_ce_id = *c1.ce_line_ids.begin();
  if (c2.ce_line_ids.find(c1_ce_id) == c2.ce_line_ids.end()) {
    // c2 does not contain c1's ce_line_id
    is_allow_to_merge = false;
  }

  if (is_debug)
    printf("part1: checking c1: %d, c2: %d, is_allow_to_merge: %d\n", c1.id,
           c2.id, is_allow_to_merge);

  // loop c2.ce_line_ids, check if any refinement of allowing to merge
  for (const int& c2_ce_id : c2.ce_line_ids) {
    if (tet_mesh.allow_to_merge_ce_line_ids.find(c2_ce_id) !=
        tet_mesh.allow_to_merge_ce_line_ids.end()) {
      c2_allow_merge_tmp = tet_mesh.allow_to_merge_ce_line_ids.at(c2_ce_id);
      if (c2_allow_merge_tmp.find(c1_ce_id) != c2_allow_merge_tmp.end()) {
        is_allow_to_merge = true;
        break;
      }
    }
  }  // for c2.ce_line_ids

  if (is_debug)
    printf("part2: checking c1: %d, c2: %d, is_allow_to_merge: %d\n", c1.id,
           c2.id, is_allow_to_merge);

  return is_allow_to_merge;
#elif FEATURE_ADJUST_SPHERE_TYPE_BY_CE_NOT_ALLOW
  // merge clusters except
  // 1. ce_line_ids listed in TetMesh::not_allow_to_merge_ce_line_ids
  // 2. two clusters contain a fid pair in SurfaceMesh::fe_sf_fs_map_se_only
  bool is_allow_to_merge = true;

  // step 1: check ce_line_ids
  //
  // step 1.2: if a pair exists in TetMesh::not_allow_to_merge_ce_line_ids
  //            then NOT allow to merge
  if (!c1.ce_line_ids.empty() && !c2.ce_line_ids.empty()) {
    std::set<int> c1_c2_ce_ids;
    c1_c2_ce_ids.insert(c1.ce_line_ids.begin(), c1.ce_line_ids.end());
    c1_c2_ce_ids.insert(c2.ce_line_ids.begin(), c2.ce_line_ids.end());

    // check if c1.ce_line_ids is in TetMesh::not_allow_to_merge_ce_line_ids
    for (const int& one_ce_id : c1_c2_ce_ids) {
      if (tet_mesh.not_allow_to_merge_ce_line_ids.find(one_ce_id) ==
          tet_mesh.not_allow_to_merge_ce_line_ids.end())
        continue;
      auto& not_allow_merge_tmp =
          tet_mesh.not_allow_to_merge_ce_line_ids.at(one_ce_id);

      if (is_debug) {
        printf("[is_allow_to_merge] one_ce_id: %d: ", one_ce_id);
        print_set(not_allow_merge_tmp,
                  "[is_allow_to_merge] not_allow_merge_tmp");
      }

      if (has_intersection(not_allow_merge_tmp, c1_c2_ce_ids)) {
        is_allow_to_merge = false;
        break;
      }
    }  // for c1.ce_line_ids
  }

  // step 1.2:
  // if one ce_line_ids has intersection with another, then allow to merge
  if (!is_allow_to_merge) {
    std::set<int> inter;
    set_intersection(c1.ce_line_ids, c2.ce_line_ids, inter);
    if (!inter.empty()) is_allow_to_merge = true;
  }

  // step 2: check if cross SE
  if (!sf_mesh.fe_sf_fs_map_se_only.empty()) {
    // check one cluster is enough
    for (const int& c1_fid : c1.fids) {
      if (sf_mesh.fe_sf_fs_map_se_only.find(c1_fid) ==
          sf_mesh.fe_sf_fs_map_se_only.end())
        continue;
      // check if c2 contains the fid in fe_sf_fs_map_se_only
      int c2_fid_not_allow = sf_mesh.fe_sf_fs_map_se_only.at(c1_fid);
      if (c2.fids.find(c2_fid_not_allow) == c2.fids.end()) continue;
      is_allow_to_merge = false;
      if (is_debug)
        printf(
            "[is_allow_to_merge] checking two clusters (%d,%d), fid (%d,%d) "
            "not allow merge\n",
            c1.id, c2.id, c1_fid, c2_fid_not_allow);

      break;
    }
  }

  if (is_debug)
    printf(
        "[is_allow_to_merge] checking c1: %d, c2: %d, is_allow_to_merge: "
        "%d\n",
        c1.id, c2.id, is_allow_to_merge);

  return is_allow_to_merge;
#endif
  return true;
}

// merge clusters, making sure that each fid is in only one cluster.
void merge_clusters_share_same_fid(const TetMesh& tet_mesh,
                                   const SurfaceMesh& sf_mesh,
                                   std::vector<SFCluster>& clusters,
                                   std::map<int, int>& f2cluster_result,
                                   bool is_debug) {
  auto update_f2cluster_given =
      [&](const std::vector<SFCluster>& clusters,
          std::map<int, std::set<int>>& f2clusters_given) {
        f2clusters_given.clear();
        // build f2clusters_given from clusters
        for (const auto& c : clusters) {
          for (const int fid : c.fids) {
            f2clusters_given[fid].insert(c.id);
          }
        }
      };
  auto is_each_f_has_one_cluster =
      [&](const std::map<int, std::set<int>>& f2clusters_given) {
        for (const auto& f2c : f2clusters_given) {
          if (f2c.second.size() > 1) return false;
        }
        return true;
      };

  std::map<int, std::set<int>> f2clusters_given;
  update_f2cluster_given(clusters, f2clusters_given);
  if (is_debug) {
    // print f2clusters_given
    for (const auto& f2c : f2clusters_given) {
      printf("fid %d -> clusters: ", f2c.first);
      for (const int& c : f2c.second) {
        printf("%d ", c);
      }
      printf("\n");
    }
  }

  int num_old_clusters = clusters.size();
  while (!is_each_f_has_one_cluster(f2clusters_given)) {
    // merge all clusters within fid of f2clusters
    // active clusters should already be sorted by SFCluster::id
    for (auto& f2c : f2clusters_given) {
      int fid = f2c.first;
      const auto& cluster_ids = f2c.second;
      if (cluster_ids.size() <= 1) continue;  // do nothing

      // check if all clusters are allow to merge
      bool is_allow_to_merge = true;
      int merge_cluster_id = *cluster_ids.begin();
      for (auto it = std::next(cluster_ids.begin()); it != cluster_ids.end();
           ++it) {
        auto& merge_cluster = clusters.at(merge_cluster_id);
        auto& cur_cluster = clusters.at(*it);
        if (is_debug)
          printf("[merge_clusters_same_fid] fid %d checking cur_cluster %d\n",
                 fid, cur_cluster.id);
        // ninwnag: yes, not having ! here
        // if allow to merge, then do not update is_allow_to_merge
        if (is_two_clusters_allow_to_merge(tet_mesh, sf_mesh, cur_cluster,
                                           merge_cluster, is_debug))
          continue;
        // abort merging
        is_allow_to_merge = false;
        // disable the cluster with empty ce_line_ids
        // this if-else order matters
        if (!merge_cluster.ce_line_ids.empty() &&
            !cur_cluster.ce_line_ids.empty()) {
          // remove fid from merge_cluster and cur_cluster
          merge_cluster.fids.erase(fid);
          cur_cluster.fids.erase(fid);
        } else if (cur_cluster.ce_line_ids.empty()) {
          cur_cluster.is_active = false;
        } else if (merge_cluster.ce_line_ids.empty()) {
          merge_cluster.is_active = false;
          merge_cluster_id = cur_cluster.id;
        } else {
          assert(false);
        }

        if (is_debug) {
          printf(
              "[merge_clusters_same_fid] fid %d, not allow to merge, "
              "cur_cluster %d, merge_cluster %d, break\n",
              fid, cur_cluster.id, merge_cluster.id);
        }

        break;
      }  // for each cluster_ids

      if (!is_allow_to_merge) continue;
      assert(merge_cluster_id != -1);
      auto& merge_cluster = clusters.at(merge_cluster_id);
      for (auto it = cluster_ids.begin(); it != cluster_ids.end(); ++it) {
        auto& cur_cluster = clusters.at(*it);
        if (merge_cluster.id == cur_cluster.id) continue;
        merge_cluster.merge(cur_cluster);  // remains one cluster
        if (is_debug)
          printf(
              "[merge_clusters_same_fid] fid %d merged cur_cluster %d to "
              "merge_cluster %d\n",
              fid, cur_cluster.id, merge_cluster.id);
      }
    }  // for each f2c

    // remove inactive clusters
    RPD3D_Wrapper::remove_reset_inactive_clusters(clusters);
    // check f2cluster again
    update_f2cluster_given(clusters, f2clusters_given);
  }

  // update SFCluster::id after cleaning
  // store f2cluster_result from cleaned clusters
  // reset cluster id
  f2cluster_result.clear();
  int num_clusters_after = 0;
  for (auto& c : clusters) {
    c.id = num_clusters_after++;
    for (const int fid : c.fids) {
      f2cluster_result[fid] = c.id;
    }
  }

  if (is_debug) {
    printf("------- merge_clusters_share_same_fid clusters size: %zu->%zu\n",
           num_old_clusters, num_clusters_after);
    // print f2cluster_result
    for (const auto& f2c : f2cluster_result) {
      printf("fid %d -> cluster %d\n", f2c.first, f2c.second);
    }
    for (const auto& c : clusters) {
      c.print();
    }
  }
}

// helper function for merge_sfclusters_using_KNN_V2()
// merge two clusters first if they share a CE
void merge_sfclusters_same_concave_edge(const TetMesh& tet_mesh,
                                        const SurfaceMesh& sf_mesh,
                                        std::map<int, int>& f2cluster,
                                        std::vector<SFCluster>& clusters,
                                        bool is_debug) {
  if (sf_mesh.fe_sf_fs_map_ce_only.empty()) return;
  if (is_debug) printf("---- merge_sfclusters_same_concave_edge start...\n");
  // loop all fid pair in one CE
  for (const auto& pair : sf_mesh.fe_sf_fs_map_ce_only) {
    const int fid1 = pair.first;
    const int fid2 = pair.second;
    if (f2cluster.find(fid1) == f2cluster.end() ||
        f2cluster.find(fid2) == f2cluster.end())
      continue;
    const int c1_id = f2cluster.at(fid1);
    const int c2_id = f2cluster.at(fid2);
    if (c1_id == c2_id) continue;
    auto& c1 = clusters.at(c1_id);
    auto& c2 = clusters.at(c2_id);
    if (!c1.is_active || !c2.is_active) continue;
    if (is_debug)
      printf(
          "[merge_sfclusters_same_concave_edge] CE fid_pair (%d,%d) found in "
          "cluster (%d,%d)\n",
          fid1, fid2, c1.id, c2.id);
    if (is_two_clusters_allow_to_merge(tet_mesh, sf_mesh, c1, c2, is_debug)) {
      c1.merge(c2);
      if (is_debug)
        printf(
            "[merge_sfclusters_same_concave_edge] merged cluster %d to "
            "cluster %d\n",
            c2.id, c1.id);
    }
  }
  RPD3D_Wrapper::remove_reset_inactive_clusters(clusters);
  // reset f2cluster
  f2cluster.clear();
  for (const auto& c : clusters)
    for (const int fid : c.fids) f2cluster[fid] = c.id;
}

// helper function for merge_sfclusters_using_KNN_V2()
// merge two clusters first if they share a CE
void merge_sfclusters_same_concave_line(const TetMesh& tet_mesh,
                                        const SurfaceMesh& sf_mesh,
                                        std::map<int, int>& f2cluster,
                                        std::vector<SFCluster>& clusters,
                                        bool is_debug) {
  if (is_debug) printf("---- merge_sfclusters_same_concave_line start...\n");
  // concave line id -> {cluster ids}
  std::map<int, std::vector<int>> cl2clusters;
  for (const auto& c : clusters) {
    if (c.ce_line_ids.empty()) continue;
    for (const int cl_id : c.ce_line_ids) cl2clusters[cl_id].push_back(c.id);
  }

  for (const auto& pair : cl2clusters) {
    int cl_id = pair.first;
    auto& merge_cluster_ids = pair.second;
    if (merge_cluster_ids.size() < 2) continue;
    if (is_debug) {
      printf("[merge_sfclusters_same_concave_line] cl_id %d merge ", cl_id);
      print_vec(merge_cluster_ids, "merge_cluster_ids");
    }
    // get first active cluster id
    int first_cluster_id =
        get_first_active_cluster_id(clusters, merge_cluster_ids[0]);
    // merge merge_cluster_ids
    auto& first_cluster = clusters[first_cluster_id];
    for (int i = 1; i < merge_cluster_ids.size(); i++) {
      int tmp_cluster_id =
          get_first_active_cluster_id(clusters, merge_cluster_ids[i]);
      // already merged, skip
      if (tmp_cluster_id == first_cluster_id) continue;
      auto& tmp_cluster = clusters[tmp_cluster_id];
      if (is_two_clusters_allow_to_merge(tet_mesh, sf_mesh, first_cluster,
                                         tmp_cluster, is_debug)) {
        first_cluster.merge(tmp_cluster);
        if (is_debug)
          printf(
              "[merge_sfclusters_same_concave_line] merged cluster %d to "
              "cluster %d\n",
              tmp_cluster.id, first_cluster.id);
      }
    }  // for merge_cluster_ids
  }  // for cl2clusters
  RPD3D_Wrapper::remove_reset_inactive_clusters(clusters);
  // reset f2cluster
  f2cluster.clear();
  for (const auto& c : clusters)
    for (const int fid : c.fids) f2cluster[fid] = c.id;

  if (is_debug) {
    printf("[merge_sfclusters_same_concave_line] ------ after:\n");
    for (const auto& c : clusters) c.print();
  }
}

// complexity O(N)
// 1. if K == -1, then use KNN_CLUSTER as default
//    else use K as the number of KNN and compute on the fly
// 2. f2cluster <fid -> cluster_id> must be 1-1 mapping
// 3. this function also update f2cluster
void merge_sfclusters_using_KNN_V2(const TetMesh& tet_mesh,
                                   const SurfaceMesh& sf_mesh,
                                   std::map<int, int>& f2cluster,
                                   std::vector<SFCluster>& clusters,
                                   const int K, bool is_debug) {
  if (is_debug) printf("---- merge_sfclusters_using_KNN_V2 start...\n");
  if (clusters.empty()) return;
  // assert(!f2cluster.empty());
  if (f2cluster.empty()) {
    printf("[merge_sfclusters] ERROR: f2cluster is empty\n");
    assert(false);
  }

  // step 1: merge clusters first if they share a CE
  //         also update f2cluster
  // merge_sfclusters_same_concave_edge(tet_mesh, sf_mesh, f2cluster, clusters,
  //                                    is_debug);
  merge_sfclusters_same_concave_line(tet_mesh, sf_mesh, f2cluster, clusters,
                                     is_debug);

  // step 2: merge clusters by expanding the neighbors of fids
  //
  // stopping condition
  std::set<int> queue_visited_fids;
  std::queue<int> queue;
  for (auto& expand_cluster : clusters) {
    // reset queue and queue_visited_fids
    while (!queue.empty()) queue.pop();
    queue_visited_fids.clear();
    if (!expand_cluster.is_active) {
      assert(expand_cluster.merge_to_cluters_id != -1);
      continue;
    }

    // add all fids in expand_cluster to queue
    for (const int fid : expand_cluster.fids) queue.push(fid);

    // step 1: expand the cluster by expanding the neighbors of fids
    while (!queue.empty()) {
      int cur_fid = queue.front();
      queue.pop();
      if (queue_visited_fids.find(cur_fid) != queue_visited_fids.end())
        continue;
      queue_visited_fids.insert(cur_fid);
      auto& cur_cluster = clusters.at(f2cluster.at(cur_fid));
      assert(cur_cluster.is_active);
      if (is_debug)
        printf("[merge_sfclusters] --- from queue, cur_fid %d, cluster %d\n",
               cur_fid, cur_cluster.id);

      // step 2:
      // merge to expand_cluster if allowed
      if (cur_cluster.id != expand_cluster.id) {
        // add all fids in cur_cluster to queue_visited_fids
        for (const int fid : cur_cluster.fids) queue_visited_fids.insert(fid);
#if FEATURE_ADJUST_SPHERE_TYPE_BY_CE || \
    FEATURE_ADJUST_SPHERE_TYPE_BY_CE_NOT_ALLOW
        bool is_allow_to_merge = is_two_clusters_allow_to_merge(
            tet_mesh, sf_mesh, expand_cluster, cur_cluster, is_debug);
        if (is_debug)
          printf(
              "[merge_sfclusters] allowing to merge cluster %d->%d, "
              "is_allow_to_merge: %d\n",
              cur_cluster.id, expand_cluster.id, is_allow_to_merge);
        if (!is_allow_to_merge) continue;
#endif
        // merge and update
        if (is_debug)
          printf(
              "[merge_sfclusters] from queue, merge fid %d, cluster %d->%d\n",
              cur_fid, cur_cluster.id, expand_cluster.id);
        expand_cluster.merge(cur_cluster);
        for (int fid : cur_cluster.fids) f2cluster[fid] = expand_cluster.id;
      }

      // step 3:
      // push neighbors to queue
      std::set<int> fid_knn;
      if (K <= 0)  // use cached when K=KNN_CLUSTER
        sf_mesh.get_sf_fid_krings_no_cross_se_only(cur_fid, fid_knn);
      else  // compute KNN on the fly
        sf_mesh.collect_kring_neighbors_given_fid_se_only(K, cur_fid, fid_knn);
      if (is_debug) print_set(fid_knn, "[merge_sfclusters] fid_knn");
      for (const int nfid : fid_knn) {
        if (nfid == cur_fid) continue;
        if (f2cluster.find(nfid) == f2cluster.end()) continue;
        if (queue_visited_fids.find(nfid) != queue_visited_fids.end()) continue;
        const auto& neigh_cluster = clusters.at(f2cluster.at(nfid));  // copy
        // already inside the expand_cluster
        if (neigh_cluster.id == expand_cluster.id) continue;
        if (is_debug)
          printf("[merge_sfclusters] found nfid: %d neigh_cluster id: %d\n",
                 nfid, neigh_cluster.id);
        assert(neigh_cluster.is_active);
        // push to queue
        if (is_debug)
          printf("[merge_sfclusters] push nfid %d to queue, cluster %d\n", nfid,
                 neigh_cluster.id);
        queue.push(nfid);
      }  // for fid_knn
    }  // while queue
  }  // for clusters

  // step 2: remove inactive clusters
  RPD3D_Wrapper::remove_reset_inactive_clusters(clusters);
  if (is_debug) {
    printf("--- clusters after merging, size %zu\n", clusters.size());
    for (const auto& c : clusters) {
      if (c.is_active) c.print();
    }
  }
}

// this implementation is too complicated
// if K == -1, then use KNN_CLUSTER as default
// else use K as the number of KNN and compute on the fly
void merge_sfclusters_using_KNN(const TetMesh& tet_mesh,
                                const SurfaceMesh& sf_mesh,
                                const std::map<int, int>& f2cluster,
                                std::vector<SFCluster>& clusters, const int K,
                                bool is_debug) {
  if (is_debug) printf("---- merge_sfclusters_using_KNN start...\n");
  if (clusters.empty()) return;
  assert(!f2cluster.empty());

  auto get_active_cluster_id_loop = [&](int cluster_id) {
    assert(cluster_id < clusters.size() && cluster_id >= 0);
    // loop until find the active cluster
    while (!clusters.at(cluster_id).is_active) {
      assert(clusters.at(cluster_id).merge_to_cluters_id != -1);
      cluster_id = clusters.at(cluster_id).merge_to_cluters_id;
    }
    return cluster_id;
  };

  // stopping condition
  std::set<int> visited_fids;

  // step 1: merge fids in f2cluster using sf_fid's KNN
  int count = 0;
  while (visited_fids.size() < f2cluster.size()) {
    if (is_debug)
      printf("visited_fids: %zu/%zu\n", visited_fids.size(), f2cluster.size());
    for (const auto& f2c_pair : f2cluster) {
      const int fid = f2c_pair.first;
      visited_fids.insert(fid);
      int cluster_id = get_active_cluster_id_loop(f2c_pair.second);
      auto& one_cluster = clusters.at(cluster_id);
      assert(one_cluster.is_active);
      if (is_debug) {
        printf("checking fid %d, cluster %d ...\n", fid, one_cluster.id);
        one_cluster.print();
      }
      std::set<int> fid_knn;
      if (K <= 0)  // use cached when K=KNN_CLUSTER
        sf_mesh.get_sf_fid_krings_no_cross_se_only(fid, fid_knn);
      else  // compute KNN on the fly
        sf_mesh.collect_kring_neighbors_given_fid_se_only(K, fid, fid_knn);
      for (const int nfid : fid_knn) {
        if (nfid == fid) continue;
        if (f2cluster.find(nfid) == f2cluster.end()) continue;
        int ncluster_id = get_active_cluster_id_loop(f2cluster.at(nfid));
        if (is_debug)
          printf("found nfid: %d ncluster_id: %d\n", nfid, ncluster_id);
        // already merged to current cluster
        if (one_cluster.id == ncluster_id) continue;
        auto& neigh_cluster = clusters.at(ncluster_id);
        assert(neigh_cluster.is_active);
#if FEATURE_ADJUST_SPHERE_TYPE_BY_CE || \
    FEATURE_ADJUST_SPHERE_TYPE_BY_CE_NOT_ALLOW
        bool is_allow_to_merge = is_two_clusters_allow_to_merge(
            tet_mesh, sf_mesh, one_cluster, neigh_cluster, is_debug);
        if (is_debug)
          printf("allowing to merge cluster %d->%d, is_allow_to_merge: %d\n",
                 neigh_cluster.id, one_cluster.id, is_allow_to_merge);
        if (!is_allow_to_merge) continue;
#endif
        if (is_debug)
          printf("merge fid %d->%d, cluster %d->%d\n", nfid, fid,
                 neigh_cluster.id, one_cluster.id);
        one_cluster.merge(neigh_cluster);
        // if (is_debug) printf("visited_fids: %zu\n", visited_fids.size());
        // if (visited_fids.size() >= f2cluster.size()) break;
        count++;
      }
      // if (visited_fids.size() >= f2cluster.size()) break;
    }  // for f2cluster
  }  // while

  if (is_debug) {
    printf("----- count %d\n", count);
    printf("--- clusters after merging:\n");
    for (const auto& c : clusters) {
      if (c.is_active) c.print();
    }
  }

  // step 2: remove inactive clusters
  RPD3D_Wrapper::remove_reset_inactive_clusters(clusters);
  if (is_debug) printf("clusters size: %zu\n", clusters.size());
}

// update EXTF sphere's FL ids from pcells:
// 1. SE: MedialSphere::se_edge_id/se_line_id or
// 2. Corner: MedialSphere::corner_fls/corner_fes
//
// must call after RPD3D_GPU::update_spheres_power_cells()
void RPD3D_Wrapper::update_extf_sphere_FL_from_pcells() {
  printf("[RPD3D_Wrapper] calling update_extf_sphere_FL_from_pcells...\n");
  // run thread
  auto run_sphere_thread = [&](int sphere_id) {
    auto& msphere = this->all_medial_spheres->at(sphere_id);
    if (msphere.is_deleted) return;
    if (!msphere.is_on_extf()) return;
    if (msphere.is_on_se()) {
      // get se_edge_id/se_line_id
      // stored in [cell_id, lvid1, lvid2, fe_line_id, fe_id]
      for (const auto& se_lvds : msphere.pcell.se_covered_lvids) {
        msphere.se_line_id = se_lvds[3];
        msphere.se_edge_id = se_lvds[4];
        return;  // one is enough
      }
      return;
    }  // is_on_se

    if (msphere.is_on_corner()) {
      // get corner_fls/corner_fes
      // stored in [cell_id, lvid1, lvid2, fe_line_id, fe_id]
      for (const auto& se_lvids : msphere.pcell.se_covered_lvids) {
        msphere.corner_fls.insert(se_lvids[3]);
        msphere.corner_fes.insert(se_lvids[4]);
      }
      // stored in [cell_id, lvid1, lvid2, fe_line_id, fe_id]
      for (const auto& ce_lvids : msphere.pcell.ce_covered_lvids) {
        msphere.corner_fls.insert(ce_lvids[3]);
        msphere.corner_fes.insert(ce_lvids[4]);
      }
      return;
    }
  };  // run_sphere_thread

  int num_spheres = this->all_medial_spheres->size();
  GEO::parallel_for(0, num_spheres,
                    [&](int sphere_id) { run_sphere_thread(sphere_id); });
}

// private, run in parallel
void RPD3D_Wrapper::cluster_one_sphere_type_using_samples_fids(int sphere_id,
                                                               int K) {
  auto& msphere = this->all_medial_spheres->at(sphere_id);
  if (msphere.is_deleted) return;
  bool is_debug_thread = false;
  const auto& sphere_pcell_samples = this->sphere_pcell_samples.at(sphere_id);

  // if (sphere_id == 150)
  //   is_debug_thread = true;
  // else
  //   is_debug_thread = false;

  if (is_debug_thread)
    printf("[cluter_sphere] processing sphere_id %d\n", sphere_id);
  // ninwang: yes we should cluster se and corner spheres
  //          but not assign type to them
  //          otherwise connectivity to se/corner would be lost
  // if (msphere.is_on_se() || msphere.is_on_corner()) return;
  std::vector<SFCluster> clusters;
  std::map<int, int> f2cluster;
  // step 1: aggregate sample pids by their fids
  auto sample_fids = this->get_sphere_pcell_samples_attribute<int>(
      sphere_id, PCELL_SAMPLE::ATTR::PROJ_FID);
  if (sample_fids.empty()) return;
  int num_samples = sample_fids.size();
  for (int pid = 0; pid < num_samples; pid++) {
    int fid = sample_fids.at(pid);
    // fid not in f2cluster
    if (f2cluster.find(fid) == f2cluster.end()) {
      f2cluster[fid] = clusters.size();
      clusters.push_back(SFCluster(clusters.size(), fid, pid));

    } else {
      // insert pid to the existing cluster
      clusters[f2cluster[fid]].fids.insert(fid);
      clusters[f2cluster[fid]].pids.insert(pid);
    }
    auto& c = clusters[f2cluster[fid]];
    c.pid_sphere_id = sphere_id;
    // step 1.5: collect SFCluster::ce_line_ids by concave lines
    //
    // we store the CE id of each projected sample in
    // PCELL_SAMPLE::proj_FE_id
    const auto& one_pcell_sample = sphere_pcell_samples.at(pid);
    int ce_id = one_pcell_sample.proj_FE_id;
    if (ce_id == UNK_FACE) continue;
    int cl_id = this->tet_mesh->get_fl_id(ce_id);
    c.ce_line_ids.insert(cl_id);
  }
  if (is_debug_thread) {
    printf("[cluter_sphere] sphere_id %d has init clusters size: %zu\n",
           sphere_id, clusters.size());
    for (const auto& c : clusters) c.print();
  }

  // step 2: merge clusters
  if (is_debug_thread)
    printf("[cluter_sphere] ---- processing sphere_id %d, init clusters:%zu\n",
           sphere_id, clusters.size());
  merge_sfclusters_using_KNN_V2(*this->tet_mesh, *this->sf_mesh, f2cluster,
                                clusters, K, is_debug_thread);

#if FEATURE_ADJUST_SPHERE_TYPE_BY_FILTERING
  // NOTE: this feature would ignore some clusters with too few samples
  //       (if smaller than sample_rel_ratio)
  //
  // step 3: sort clusters by pids size in descending order
  std::sort(clusters.begin(), clusters.end(),
            [](const SFCluster& a, const SFCluster& b) {
              return a.pids.size() > b.pids.size();
            });
  int num_clusters = clusters.size();

  // step 4: check the sample percentage of each cluster,
  // if one cluster has too few samples, we should ignore this cluster,
  // ONLY handle when #cluster > 2
  // e.g. T_3 sphere is actually T_2
  double sample_rel_ratio = 0.03;
  if (num_clusters > 2) {
    int total_num_samples = 0;
    for (const auto& c : clusters) {
      total_num_samples += c.pids.size();
    }
    // if one cluster has too few samples, we should ignore this cluster
    int num_active_clusters = clusters.size();
    for (auto& c : clusters) {
      // // NOTE: if cluster contains a CE, then we should keep it
      // no matter how small the cluster is
      if (!c.ce_line_ids.empty()) {
        sample_rel_ratio = 0.01;
        // continue;
      }
      /////////
      if (c.pids.size() < total_num_samples * sample_rel_ratio) {
        c.is_active = false;
        num_active_clusters--;
      }
    }
    // if we only have one active cluster, make last two clusters active
    if (num_active_clusters < 2) {
      // we have sorted the clusters
      for (int i = num_active_clusters; i > num_active_clusters - 2; i--) {
        clusters.at(i).is_active = true;
      }
      num_active_clusters = 2;
    }
    // remove inactive clusters
    RPD3D_Wrapper::remove_reset_inactive_clusters(clusters);

    if (num_clusters != clusters.size()) {
      if (is_debug_thread)
        printf("[cluter_sphere] sphere_id %d, adjust clusters size: %zu->%zu\n",
               sphere_id, num_clusters, clusters.size());
      this->is_sphere_adj_type[sphere_id] = 1;
    }
  }
#endif

  // step 3:
  // for classifying SphereType::T_1_INF or SphereType::T_2_INF
  // 1. update the variance of each cluster's normals
  // 2. update the normals_svd_rank
  Eigen::JacobiSVD<Eigen::Matrix4d> _;
  std::vector<Vector3> cluster_normals;
  for (auto& c : clusters) {
    // cannot call RPD3D_Wrapper::get_sphere_clusters_attribute()
    // since we have not save the clusters
    cluster_normals.clear();
    for (const int& pid : c.pids) {
      const Vector3& normal = sphere_pcell_samples.at(pid).normal;
      cluster_normals.push_back(normal);
    }
    // check the variance of normals
    update_normal_variance(*(this->params), cluster_normals, c);
    // check the normals SVD rank
    if (clusters.size() == 1 || clusters.size() == 2) {
      c.normals_svd_rank =
          get_normals_SVD_rank(cluster_normals, _, false /*is_debug*/);
    }
  }

  // step 4: save and set type
  // reset cluster id
  int num_clusters_after = 0;
  for (auto& c : clusters) c.id = num_clusters_after++;
  this->sphere_clusters.at(sphere_id) = clusters;
  // do not assign type to se/corner spheres
  if (msphere.is_on_se() || msphere.is_on_corner()) return;

  // NOTE:
  // Normals around concave lines also have large variance,
  // so when cc_line exists, SFCluster::is_normal_inf is not reliable.
  // Here, we need to use SVD rank to help.
  //
  // Note: this rank is not that reliable
  //       some T1 sphere may be classified as T_1_INF
  //       some T2 sphere may be classified as T_2_INF
  //       but not in reverse
  msphere.type = SphereType::T_UNK;
  if (clusters.size() == 1 && is_inf_more_than_T1(clusters[0].is_normal_inf)) {
    if (clusters[0].ce_line_ids.empty())
      // case 1: when cc_line = empty
      msphere.type = SphereType::T_1_INF;
    else {
      // case 2: when cc_line != empty, check cluster rank
      // Note: this is not that reliable
      //       some T1 sphere may be classified as T_1_INF
      if (clusters[0].normals_svd_rank > 3) msphere.type = SphereType::T_1_INF;
      if (is_debug_thread)
        printf("sphere_id %d, has cluster 0 rank %d\n", sphere_id,
               clusters[0].normals_svd_rank);
      // NOTE: if cluster_num_rank == 2,
      // then this is a T1 sphere only tangent to a cc_line
    }
  } else if (clusters.size() == 2) {
    msphere.type = SphereType::T_2;
    // check if SphereType::T_2_INF
    for (int cid = 0; cid < clusters.size(); cid++) {
      const auto& c = clusters.at(cid);
      if (!is_inf_more_than_T2(c.is_normal_inf)) continue;
      // case 1: when cc_line = empty
      if (c.ce_line_ids.empty()) {
        msphere.type = SphereType::T_2_INF;  // on seams, not sheet
        break;
      } else {
        // case 2: when cc_line != empty
        // Note: this is not that reliable
        //       some T2 sphere may be classified as T_2_INF
        if (c.normals_svd_rank > 3) {
          msphere.type = SphereType::T_2_INF;
          if (is_debug_thread)
            printf("sphere_id %d, has cluster %d rank %d\n", sphere_id, cid,
                   c.normals_svd_rank);
          break;
        }
      }
    }
  } else if (clusters.size() > 3) {
    msphere.type = SphereType::T_4_MORE;
  } else if (clusters.size() > 2) {
    msphere.type = SphereType::T_3_MORE;
  } else {
    msphere.type = SphereType::T_UNK;
  }

  if (is_debug_thread)
    printf("[cluter_sphere] sphere_id %d final type: %d\n", sphere_id,
           msphere.type);
}

// will update tan_plane and tan_cc_line in the end
// by calling RPD3D_Wrapper::update_all_spheres_tangents_use_clusters()
void RPD3D_Wrapper::cluster_sphere_type_using_samples_fids(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh, bool is_debug) {
  int num_spheres = this->all_medial_spheres->size();
  // reset
  this->sphere_clusters.clear();
  this->sphere_clusters.resize(num_spheres);
  this->is_sphere_adj_type.resize(num_spheres, 0);

  ////////////////////////////////////////////////////////////////////////////
  const clock_t start_t = clock();
  // // this can be multi-threaded
  // for (int sphere_id = 0; sphere_id < num_spheres; sphere_id++) {
  //   run_sphere_thread(sphere_id);
  // }
  GEO::parallel_for(0, num_spheres, [&](int sphere_id) {
    cluster_one_sphere_type_using_samples_fids(sphere_id);
  });

  // print spheres should be sampled by clusters
  int num_spheres_adj_type = 0;
  for (int i = 0; i < this->is_sphere_adj_type.size(); i++) {
    if (this->is_sphere_adj_type[i] == 1) num_spheres_adj_type++;
  }
  printf("[cluter_sphere] adjusted %zu/%zu spheres type\n",
         num_spheres_adj_type, num_spheres);

  // // update sphere tangents based on the clusters
  this->update_all_spheres_tangents_use_clusters(false /*is_debug*/);
  printf("[cluter_sphere] updated sphere tangents\n");

  std::cout << "Clustering mspheres took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// For T1 spheres, use neighbors' samples to replace current samples
//
// NOTE: must call after update_pcell_samples_cuda() and
// cluster_sphere_type_using_samples_fids()
//
// DO NOT CHANGE -> RPD3D_Wrapper::sphere_clusters!!!
//
// will expand the following data structure:
// 1. RPD3D_Wrapper::sphere_pcell_samples;
void RPD3D_Wrapper::expand_pcell_samples_by_neighbors(
    const SurfaceMesh& sf_mesh, bool is_debug) {
  auto run_thread = [&](int sphere_id) {
    const auto& msphere = this->all_medial_spheres->at(sphere_id);
    if (msphere.is_deleted) return;
    if (msphere.is_on_se() || msphere.is_on_corner()) return;
    if (msphere.is_on_inf()) return;  // not expand if SphereType::T_1_INF
    auto& one_sphere_clusters = this->sphere_clusters.at(sphere_id);
    // only handle sphere with 1 cluster
    if (one_sphere_clusters.size() != 1) return;
    if (is_debug)
      printf("[ExpandT1] sphere_id %d has only 1 cluster, expand samples\n",
             sphere_id);

    // get pcell samples
    auto& one_pcell_samples = this->sphere_pcell_samples.at(sphere_id);
    int prev_sample_size = one_pcell_samples.size();
    // get mmesh neighbors
    std::set<int> neighbors;
    this->get_one_pcell_neighbors(sphere_id, neighbors);
    if (neighbors.empty()) return;

    // // use neighbors' samples to replace current samples
    // one_pcell_samples.clear()

    // use neighbors' samples to expand current samples
    //
    // loop all neighbors and expand just one
    // only expand samples from one neighbor
    int neigh_id = -1;
    for (const int nid : neighbors) {
      // do not consider T1 neighbor
      if (this->sphere_clusters.at(nid).size() <= 1) continue;
      const auto& neigh_sphere = this->all_medial_spheres->at(nid);
      if (neigh_sphere.is_deleted) continue;
      if (neigh_sphere.is_on_se() || neigh_sphere.is_on_corner()) continue;
      const auto& n_pcell_samples = this->sphere_pcell_samples.at(nid);
      // expand
      for (int i = 0; i < n_pcell_samples.size(); i++) {
        const auto& n_pcell_sample = n_pcell_samples.at(i);
        one_pcell_samples.push_back(n_pcell_sample);
      }
      // only expand samples from one neighbor
      neigh_id = nid;
      break;
      // if (is_debug) printf("sphere %d nid %d\n", sphere_id, nid);
    }
    if (neigh_id == -1) {
      // do nothing
      // assert(neigh_id != -1);
    }
    // if (is_debug)
    printf("[ExpandT1] sphere_id %d append neigh %d samples %d->%d\n",
           sphere_id, neigh_id, prev_sample_size, one_pcell_samples.size());
  };

  // for (int sphere_id = 0; sphere_id < this->all_medial_spheres->size();
  //      sphere_id++) {
  //   // if (sphere_id == 505)
  //   //   is_debug = true;
  //   // else
  //   //   is_debug = false;
  //   run_thread(sphere_id);
  // }

  GEO::parallel_for(0, this->all_medial_spheres->size(),
                    [&](int sphere_id) { run_thread(sphere_id); });
}

// NOT THREAD-SAFE!!!!!
// may call SFCluster::expand_fids() inside the function
// make a copy of v1_clusters and v2_clusters before calling this function!!!
void join_two_clusters(const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
                       bool is_expand_fids, std::vector<SFCluster>& v1_clusters,
                       std::vector<SFCluster>& v2_clusters,
                       std::vector<SFCluster>& join_clusters, bool is_debug) {
  if (is_debug) {
    printf("[join two clusters] start ...\n");
    printf("[join two clusters] v1_clusters: %zu, v2_clusters: %zu\n",
           v1_clusters.size(), v2_clusters.size());
  }
  assert(!v1_clusters.empty() && !v2_clusters.empty());
  auto remove_dup_fid_in_clusters = [](std::vector<SFCluster>& clusters) {
    std::vector<SFCluster> new_clusters;
    std::set<int> dup_fids, dup_tmp;
    // find all possible dup_fids, if exists in ANY two clusters
    for (auto& c1 : clusters) {
      for (auto& c1_tmp : clusters) {
        if (c1.id == c1_tmp.id) continue;
        set_intersection(c1.fids_expand, c1_tmp.fids_expand, dup_tmp);
        dup_fids.insert(dup_tmp.begin(), dup_tmp.end());
      }
    }
    // remove dup_fids from each cluster
    for (auto& c : clusters) {
      std::set<int> new_fids;
      std::set_difference(c.fids_expand.begin(), c.fids_expand.end(),
                          dup_fids.begin(), dup_fids.end(),
                          std::inserter(new_fids, new_fids.begin()));
      c.fids_expand = new_fids;
    }
  };

  // expand clusters if needed
  if (is_expand_fids) {
    for (auto& c : v1_clusters)
      if (c.fids_expand.empty()) c.expand_fids(sf_mesh, KNN_CLUSTER_EXPAND);
    for (auto& c : v2_clusters)
      if (c.fids_expand.empty()) c.expand_fids(sf_mesh, KNN_CLUSTER_EXPAND);
    // remove duplicated fids in fids_expand
    // to avoid expanding too much
    remove_dup_fid_in_clusters(v1_clusters);
    remove_dup_fid_in_clusters(v2_clusters);
    // re-add SFClsuter::fids to fids_expand
    for (auto& c : v1_clusters)
      c.fids_expand.insert(c.fids.begin(), c.fids.end());
    for (auto& c : v2_clusters)
      c.fids_expand.insert(c.fids.begin(), c.fids.end());
  }

  // sort clusters by their size
  std::vector<SFCluster>* v1_clusters_ptr = &v1_clusters;
  std::vector<SFCluster>* v2_clusters_ptr = &v2_clusters;
  if (v1_clusters.size() > v2_clusters.size()) {
    std::swap(v1_clusters_ptr, v2_clusters_ptr);
    if (is_debug) printf("[join two clusters] SWAP v1 and v2 !!!!\n");
  }
  // NOTE: v1 might has intersection with multiple v2, we need to merge all of
  // them. This usually happens at multiple concave lines (eg. model 1510)
  //
  // join clusters if one cluster in v1_clusters has non-empty intersection with
  // one cluster in v2_clusters then add the join into join_clusters
  //
  // step 1: find join clusters
  join_clusters.clear();
  std::set<int> join_fids;
  std::set<int> joint_v2_clusters;  // already joined
  for (const auto& c1 : *v1_clusters_ptr) {
    // to join all possible clusters in v2_clusters
    SFCluster join_c(join_clusters.size());
    for (const auto& c2 : *v2_clusters_ptr) {
      if (joint_v2_clusters.find(c2.id) != joint_v2_clusters.end()) continue;
      // step1: check if c1 and c2 can merge
      bool is_allow_to_merge =
          is_two_clusters_allow_to_merge(tet_mesh, sf_mesh, c1, c2, is_debug);
      if (is_debug)
        printf(
            "[join two clusters] two clusters (%d,%d), is_allow_to_merge: %d\n",
            c1.id, c2.id, is_allow_to_merge);
      if (!is_allow_to_merge) continue;

      // step 2: find the intersection of fids
      join_fids.clear();
      set_intersection(c1.fids, c2.fids, join_fids);
      if (join_fids.empty()) {
        // find the intersection of fids_expand
        set_intersection(c1.fids_expand, c2.fids_expand, join_fids);
      }
      if (join_fids.empty()) continue;

      if (is_debug) {
        printf("[join two clusters] join two clusters (%d,%d)\n", c1.id, c2.id,
               join_fids.size());
        print_set(join_fids, "join_fids");
      }

      // step 3: add to join cluster
      join_c.fids.insert(join_fids.begin(), join_fids.end());
      join_c.pid_sphere_id = c1.pid_sphere_id;
      // to avoid re-expanding
      join_c.fids_expand.insert(join_fids.begin(), join_fids.end());
      join_c.ce_line_ids.insert(c1.ce_line_ids.begin(), c1.ce_line_ids.end());
      join_c.ce_line_ids.insert(c2.ce_line_ids.begin(), c2.ce_line_ids.end());
      joint_v2_clusters.insert(c2.id);
    }  // for v2_clusters

    if (!join_c.fids.empty()) {
      join_clusters.push_back(join_c);
    }
  }  // for v1_clusters

  if (is_debug) {
    printf("===== v1_clusters: \n");
    for (const auto& c : *v1_clusters_ptr) c.print();
    printf("===== v2_clusters: \n");
    for (const auto& c : *v2_clusters_ptr) c.print();
    printf("===== join_clusters: \n");
    for (const auto& c : join_clusters) c.print();
  }
}

// update
// 1. RPD3D_Wrapper::medge_clusters
// 2. MedialEdge::is_on_same_sheet
// 3. MedialEdge::is_extf
void RPD3D_Wrapper::mark_one_medge_same_sheet_by_adj_bface(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
    const std::vector<std::vector<SFCluster>>& sphere_clusters,
    std::vector<std::vector<SFCluster>>& medge_clusters, MedialMesh& mmesh,
    int eid, bool is_debug) {
  assert(medge_clusters.size() > eid);
  auto& medge = mmesh.edges.at(eid);
  if (medge.is_deleted) return;
  if (is_debug)
    printf(
        "[mark_one_medge_same_sheet_by_adj_bface] marking medge (%d,%d) ..\n",
        medge.vertices_[0], medge.vertices_[1]);

  //////////////////////////////////////////////////////////////
  // check and mark MedialEdge::is_extf
  bool is_skip_SE_already_markd = false;
  const MedialSphere& A = mmesh.vertices->at(medge.vertices_[0]);
  const MedialSphere& B = mmesh.vertices->at(medge.vertices_[1]);
  if (A.is_on_extf() && B.is_on_extf()) {
    is_skip_SE_already_markd = true;
    if (is_two_mspheres_on_same_sl_including_corners(A, B)) {
      medge.is_extf = true;
      medge.is_on_same_sheet = true;
    }
    // NOTE: do not return, we still need to update
    // RPD3D_Wrapper::medge_clusters, for these two spheres
  }
  // is_debug = false;
  // if (A.id == 1103 && B.id == 1112 || A.id == 1112 && B.id == 1103)
  //   is_debug = true;

  if (is_debug)
    printf(
        "[mark_one_medge_same_sheet_by_adj_bface] medge: (%d,%d) checking...\n",
        A.id, B.id);

  //////////////////////////////////////////////////////////////
  // check and mark MedialEdge::is_on_same_sheet
  ////////////
  // find one sphere (A or B) with min number of clusters
  assert(A.id < sphere_clusters.size());
  assert(B.id < sphere_clusters.size());
  std::vector<int> sorted_sphere_ids = {A.id, B.id};
  sort_spheres_ids_from_clusters(sorted_sphere_ids);
  int min_AB_id = sorted_sphere_ids[0];
  int max_AB_id = sorted_sphere_ids[1];
  if (is_debug)
    printf(
        "[mark_one_medge_same_sheet_by_adj_bface] medge: (%d,%d), "
        "min_AB_id: %d, max_AB_id: %d\n",
        A.id, B.id, min_AB_id, max_AB_id);

  ////////////
  // get samples on pcell bface on both A and B,
  // cluster is natually inherited from RPD3D_Wrapper::sphere_clusters
  //
  // NOTE:
  // 1. do NOT use RPD3D_Wrapper::get_sphere_pcell_samples_attribute()
  // 2. do NOT use RPD3D_Wrapper::get_sphere_clusters_attribute()
  auto& AB_medge_clusters = medge_clusters.at(eid);
  for (const auto& min_AB_cluster : this->sphere_clusters.at(min_AB_id)) {
    // save a new cluster
    SFCluster cur_medge_cluster(AB_medge_clusters.size());
    cur_medge_cluster.pid_sphere_id = min_AB_id;
    cur_medge_cluster.is_normal_inf = min_AB_cluster.is_normal_inf;  // inherit
    // save filtered pids in the cluster
    for (const auto& min_AB_pid : min_AB_cluster.pids) {
      const auto& one_sample =
          this->sphere_pcell_samples.at(min_AB_id).at(min_AB_pid);
      // skip if not on both A and B
      if (one_sample.adj_sphere_ids.find(max_AB_id) ==
          one_sample.adj_sphere_ids.end())
        continue;
      cur_medge_cluster.fids.insert(one_sample.proj_fid);
      cur_medge_cluster.pids.insert(min_AB_pid);
      int cl_id = one_sample.proj_FE_id == UNK_FACE
                      ? -1
                      : tet_mesh.get_fl_id(one_sample.proj_FE_id);
      if (cl_id != -1) cur_medge_cluster.ce_line_ids.insert(cl_id);
    }
    // save, only when this cluster contains info on both A and B
    if (!cur_medge_cluster.fids.empty())
      AB_medge_clusters.push_back(cur_medge_cluster);
  }
  // default MedialEdge::is_on_same_sheet = false
  if (AB_medge_clusters.empty()) return;
  if (is_debug) {
    printf(
        "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d), init "
        "AB_medge_cluster: %zu\n",
        min_AB_id, max_AB_id, AB_medge_clusters.size());
    for (const auto& c : AB_medge_clusters) c.print();
  }

  // //////////////////
  // // check AB_medge_clusters with max_AB_cluster
  // // 1. any cluster in AB_medge_clusters must exist in max_AB_cluster
  // std::vector<int> inter;
  // for (auto& one_AB_c : AB_medge_clusters) {
  //   bool is_delete = true;
  //   for (const auto& max_AB_cluster : this->sphere_clusters.at(max_AB_id)) {
  //     set_intersection(one_AB_c.fids, max_AB_cluster.fids, inter);
  //     // if no fids in common, then check the ce_line_ids
  //     if (inter.empty())
  //       set_intersection(one_AB_c.ce_line_ids, max_AB_cluster.ce_line_ids,
  //                        inter);
  //     if (inter.empty() && is_debug)
  //       printf(
  //           "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d) one_AB_c
  //           "
  //           "%d no fids in common with max_AB_cluster %d\n",
  //           min_AB_id, max_AB_id, one_AB_c.id, max_AB_cluster.id);
  //     if (!inter.empty()) {
  //       is_delete = false;
  //       break;
  //     }
  //   }
  //   if (is_delete) one_AB_c.is_active = false;
  // }
  // RPD3D_Wrapper::remove_reset_inactive_clusters(AB_medge_clusters);
  // if (is_debug) {
  //   printf(
  //       "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d), merge1 "
  //       "AB_medge_cluster: %zu\n",
  //       min_AB_id, max_AB_id, AB_medge_clusters.size());
  //   for (const auto& c : AB_medge_clusters) c.print();
  // }

  // // 2. one-to-one mapping from AB_medge_clusters to max_AB_cluster
  // std::vector<int> same_AB_clusters;
  // for (const auto& max_AB_cluster : this->sphere_clusters.at(max_AB_id)) {
  //   same_AB_clusters.clear();
  //   for (auto& one_AB_c : AB_medge_clusters) {
  //     set_intersection(one_AB_c.fids, max_AB_cluster.fids, inter);
  //     if (!inter.empty()) same_AB_clusters.push_back(one_AB_c.id);
  //   }
  //   if (same_AB_clusters.size() < 2) continue;
  //   if (max_AB_cluster.is_normal_inf != SphereType::T_UNK) {
  //     // if max_AB_cluster is on INF,
  //     // same_AB_clusters no need to merge
  //     continue;
  //   }

  //   // merge clusters in same_AB_clusters
  //   auto& cur_AB_c = AB_medge_clusters.at(same_AB_clusters[0]);
  //   for (int i = 1; i < same_AB_clusters.size(); i++) {
  //     auto& tmp_AB_c = AB_medge_clusters.at(same_AB_clusters[i]);
  //     if (is_two_clusters_allow_to_merge(tet_mesh, sf_mesh, cur_AB_c,
  //     tmp_AB_c,
  //                                        false)) {
  //       cur_AB_c.merge(tmp_AB_c);
  //       if (is_debug)
  //         printf(
  //             "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d) merge "
  //             "AB_medge_clusters: %d -> %d\n",
  //             min_AB_id, max_AB_id, tmp_AB_c.id, cur_AB_c.id);
  //     }
  //   }
  //   RPD3D_Wrapper::remove_reset_inactive_clusters(AB_medge_clusters);
  // }
  // if (is_debug) {
  //   printf(
  //       "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d), after merge
  //       " "AB_medge_cluster: %zu\n ", min_AB_id, max_AB_id,
  //       AB_medge_clusters.size());
  //   for (const auto& c : AB_medge_clusters) c.print();
  // }

  // already mark MedialEdge::is_extf previously
  if (is_skip_SE_already_markd) {
    if (is_debug)
      printf(
          "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d) already "
          "marked as special SE, skip\n",
          min_AB_id, max_AB_id);
    return;
  }

  //////////////////
  // v1v2_join_clusters should be the same as smallest cluster
  int num_AB_medge_cluster = get_num_clusters(AB_medge_clusters);
  const int& A_num_clusters = get_msphere_num_clusters(A.id);
  const int& B_num_clusters = get_msphere_num_clusters(B.id);
  int min_cluster_size = std::min(A_num_clusters, B_num_clusters);
  if (is_debug)
    printf(
        "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d) "
        "min_cluster_size: %d, AB_medge_clusters: %zu, num_AB_medge_cluster: "
        "%d\n",
        min_AB_id, max_AB_id, min_cluster_size, AB_medge_clusters.size(),
        num_AB_medge_cluster);

  // special case: all 2 spheres are on the INTF
  if (min_cluster_size > 2 && num_AB_medge_cluster > 1) {
    if (is_debug)
      printf(
          "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d) is on the "
          "same sheet, all on INTF, min_cluster_size: %d\n",
          min_AB_id, max_AB_id, min_cluster_size);
    medge.is_on_same_sheet = true;
    return;
  }

  // normal case
  if (num_AB_medge_cluster == min_cluster_size) {
    if (is_debug)
      printf(
          "[mark_one_medge_same_sheet_by_adj_bface] medge (%d,%d) is on the "
          "same sheet, AB_medge_clusters: %zu, num_AB_medge_cluster: %d\n",
          min_AB_id, max_AB_id, AB_medge_clusters.size(), num_AB_medge_cluster);
    medge.is_on_same_sheet = true;
    return;
  }

  // default MedialFace::is_on_same_sheet = false
  return;
}

// update
// 1. RPD3D_Wrapper::mface_clusters
// 2. MedialFace::is_on_same_sheet
void RPD3D_Wrapper::mark_one_mface_same_sheet_by_adj_bface(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
    const std::vector<std::vector<SFCluster>>& sphere_clusters,
    std::vector<std::vector<SFCluster>>& mface_clusters, MedialMesh& mmesh,
    int fid, bool is_debug) {
  assert(mface_clusters.size() > fid);
  auto& mface = mmesh.faces.at(fid);
  if (mface.is_deleted) return;
  if (is_debug)
    printf(
        "[mark one mface same sheet by adj bface] marking mface "
        "(%d,%d,%d)...\n",
        mface.vertices_[0], mface.vertices_[1], mface.vertices_[2]);

  //////////////////////////////////////////////////////////////
  // return if any medges not on the same sheet
  // default, mface.is_on_same_sheet = false
  for (const int meid : mface.edges_) {
    if (!mmesh.edges.at(meid).is_on_same_sheet) return;
  }

  //////////////////////////////////////////////////////////////
  const MedialSphere& A = mmesh.vertices->at(mface.vertices_[0]);
  const MedialSphere& B = mmesh.vertices->at(mface.vertices_[1]);
  const MedialSphere& C = mmesh.vertices->at(mface.vertices_[2]);

  ////////////
  // find one sphere (A or B or C) with min number of clusters
  std::vector<int> sphere_id_clusters_sorted = {A.id, B.id, C.id};
  sort_spheres_ids_from_clusters(sphere_id_clusters_sorted);
  int A_num_clusters = get_msphere_num_clusters(A.id);
  int B_num_clusters = get_msphere_num_clusters(B.id);
  int C_num_clusters = get_msphere_num_clusters(C.id);
  if (is_debug)
    printf(
        "[mark one mface same sheet by adj bface] mface sorted (%d,%d,%d) "
        "num clusters (%d,%d,%d)\n",
        sphere_id_clusters_sorted[0], sphere_id_clusters_sorted[1],
        sphere_id_clusters_sorted[2], A_num_clusters, B_num_clusters,
        C_num_clusters);

  ////////////
  // get samples on pcell bface on both A and B,
  // cluster is natually inherited from RPD3D_Wrapper::sphere_clusters
  //
  // NOTE:
  // 1. do NOT use RPD3D_Wrapper::get_sphere_pcell_samples_attribute()
  // 2. do NOT use RPD3D_Wrapper::get_sphere_clusters_attribute()
  int min_ABC_id = sphere_id_clusters_sorted[0];
  auto& ABC_mface_clusters = mface_clusters.at(fid);
  for (const auto& min_ABC_cluster : this->sphere_clusters.at(min_ABC_id)) {
    // save a new cluster
    SFCluster cur_mface_cluster(ABC_mface_clusters.size());
    cur_mface_cluster.pid_sphere_id = min_ABC_id;
    cur_mface_cluster.is_normal_inf = min_ABC_cluster.is_normal_inf;  // inherit
    // save filtered pids in the cluster
    for (const auto& one_pid : min_ABC_cluster.pids) {
      const auto& one_sample =
          this->sphere_pcell_samples.at(min_ABC_id).at(one_pid);
      // skip if not on A and B and C
      if (one_sample.adj_sphere_ids.find(sphere_id_clusters_sorted[1]) ==
          one_sample.adj_sphere_ids.end())
        continue;
      if (one_sample.adj_sphere_ids.find(sphere_id_clusters_sorted[2]) ==
          one_sample.adj_sphere_ids.end())
        continue;
      // if (is_debug) {
      //   printf("sphere_id %d has pid %d with ", min_ABC_id, one_pid);
      //   print_set(one_sample.adj_sphere_ids, "adj_sphere_id");
      // }
      cur_mface_cluster.fids.insert(one_sample.proj_fid);
      cur_mface_cluster.pids.insert(one_pid);
      int cl_id = one_sample.proj_FE_id == UNK_FACE
                      ? -1
                      : tet_mesh.get_fl_id(one_sample.proj_FE_id);
      if (cl_id != -1) cur_mface_cluster.ce_line_ids.insert(cl_id);
    }
    // save, only when this cluster contains info on both A and B
    if (!cur_mface_cluster.fids.empty())
      ABC_mface_clusters.push_back(cur_mface_cluster);
  }
  if (ABC_mface_clusters.empty()) {
    // default, mface.is_on_same_sheet = false
    return;
  }
  if (is_debug)
    printf(
        "[mark one mface same sheet by adj bface] mface (%d,%d,%d), init "
        "ABC_mface_clusters: %zu\n",
        mface.vertices_[0], mface.vertices_[1], mface.vertices_[2],
        ABC_mface_clusters.size());

  //////////////////////////////////////////////////////////////
  // v1v2v3_join_clusters should be the same as smallest cluster
  int min_cluster_size =
      std::min({A_num_clusters, B_num_clusters, C_num_clusters});
  int num_mface_clusters = get_num_clusters(ABC_mface_clusters);

  if (is_debug)
    printf(
        "[mark one mface same sheet by adj bface] mface (%d,%d,%d), "
        "ABC_mface_clusters: %zu, num_mface_clusters: %d\n",
        mface.vertices_[0], mface.vertices_[1], mface.vertices_[2],
        ABC_mface_clusters.size(), num_mface_clusters);

  // special case: all 3 spheres are on the INTF, but on different INTFs
  if (min_cluster_size > 2 && num_mface_clusters > 1) {
    if (is_debug)
      printf(
          "[mark one mface same sheet by adj bface] mface (%d,%d,%d) is on "
          "the same sheet, all on INTF, min_cluster_size: %d, "
          "min_cluster_size: %d\n",
          mface.vertices_[0], mface.vertices_[1], mface.vertices_[2],
          min_cluster_size, min_cluster_size);
    mface.is_on_same_sheet = true;
    return;
  }

  // normal case
  if (num_mface_clusters == min_cluster_size) {
    if (is_debug)
      printf(
          "[mark one mface same sheet by adj bface] mface (%d,%d,%d) is on "
          "the same sheet, num_mface_clusters: %d, min_cluster_size: %d\n",
          mface.vertices_[0], mface.vertices_[1], mface.vertices_[2],
          num_mface_clusters, min_cluster_size);
    mface.is_on_same_sheet = true;
    return;
  }

  // default MedialFace::is_on_same_sheet = false
  return;
}

// update
// 1. RPD3D_Wrapper::mface_clusters
// 2. MedialFace::is_on_same_sheet
void RPD3D_Wrapper::mark_one_mface_same_sheet_by_adj_bface_KNN(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
    const std::vector<std::vector<SFCluster>>& sphere_clusters,
    std::vector<std::vector<SFCluster>>& mface_clusters, MedialMesh& mmesh,
    int fid, bool is_debug) {
  assert(mface_clusters.size() > fid);
  auto& mface = mmesh.faces.at(fid);
  if (mface.is_deleted) return;
  bool is_debug_thread = false;
  if (mface.vertices_[0] == 689 && mface.vertices_[1] == 694 &&
      mface.vertices_[2] == 851)
    is_debug_thread = true;

  if (is_debug_thread)
    printf("[mface_same_sheet_KNN] marking mface (%d,%d,%d)...\n",
           mface.vertices_[0], mface.vertices_[1], mface.vertices_[2]);

  //////////////////////////////////////////////////////////////
  // return if any medges not on the same sheet
  // default, mface.is_on_same_sheet = false
  for (const int meid : mface.edges_) {
    if (!mmesh.edges.at(meid).is_on_same_sheet) return;
  }

  //////////////////////////////////////////////////////////////
  auto& ABC_mface_clusters = mface_clusters.at(fid);
  for (int i = 0; i < mface.vertices_.size(); i++) {
    int sphere_A_id = mface.vertices_[i];
    int sphere_B_id = mface.vertices_[(i + 1) % 3];
    int sphere_C_id = mface.vertices_[(i + 2) % 3];
    // get all pids that shared by three spheres
    for (const auto& first_cluster : this->sphere_clusters.at(sphere_A_id)) {
      // save a new cluster
      SFCluster cur_mface_cluster(ABC_mface_clusters.size());
      cur_mface_cluster.pid_sphere_id = sphere_A_id;
      cur_mface_cluster.is_normal_inf = first_cluster.is_normal_inf;  // inherit
      // save filtered pids in the cluster
      for (const auto& one_pid : first_cluster.pids) {
        const auto& one_sample =
            this->sphere_pcell_samples.at(sphere_A_id).at(one_pid);
        // skip if not on A and B and C
        if (one_sample.adj_sphere_ids.find(sphere_B_id) ==
            one_sample.adj_sphere_ids.end())
          continue;
        if (one_sample.adj_sphere_ids.find(sphere_C_id) ==
            one_sample.adj_sphere_ids.end())
          continue;
        // if (is_debug_thread) {
        //   printf("sphere_id %d has pid %d with ", sphere_A_id, one_pid);
        //   print_set(one_sample.adj_sphere_ids, "adj_sphere_id");
        // }
        cur_mface_cluster.fids.insert(one_sample.proj_fid);
        cur_mface_cluster.pids.insert(one_pid);
        int cl_id = one_sample.proj_FE_id == UNK_FACE
                        ? -1
                        : tet_mesh.get_fl_id(one_sample.proj_FE_id);
        if (cl_id != -1) cur_mface_cluster.ce_line_ids.insert(cl_id);
      }
      // save, only when this cluster contains info on both A and B
      if (!cur_mface_cluster.fids.empty())
        ABC_mface_clusters.push_back(cur_mface_cluster);
    }
  }
  if (is_debug_thread) {
    printf("[mface_same_sheet_KNN] mface (%d,%d,%d) clusters:\n",
           mface.vertices_[0], mface.vertices_[1], mface.vertices_[2]);
    for (const auto& c : ABC_mface_clusters) c.print();
  }

  // merge clusters since they may come from three different spheres
  // merge all clusters
  std::map<int, int> f2cluster;
  merge_clusters_share_same_fid(*this->tet_mesh, *this->sf_mesh,
                                ABC_mface_clusters, f2cluster,
                                false /*is_debug*/);
  merge_sfclusters_using_KNN_V2(*this->tet_mesh, *this->sf_mesh, f2cluster,
                                ABC_mface_clusters, KNN_CLUSTER /*K*/,
                                false /*is_debug*/);

  if (is_debug_thread) {
    printf("[mface_same_sheet_KNN] mface (%d,%d,%d) merged clusters:\n",
           mface.vertices_[0], mface.vertices_[1], mface.vertices_[2]);
    for (const auto& c : ABC_mface_clusters) c.print();
  }

  if (ABC_mface_clusters.empty()) {
    // default, mface.is_on_same_sheet = false
    return;
  }
  if (is_debug_thread)
    printf(
        "[mface_same_sheet_KNN] mface (%d,%d,%d), init ABC_mface_clusters: "
        "%zu\n",
        mface.vertices_[0], mface.vertices_[1], mface.vertices_[2],
        ABC_mface_clusters.size());

  //////////////////////////////////////////////////////////////
  // v1v2v3_join_clusters should be the same as smallest cluster
  int A_num_clusters = get_msphere_num_clusters(mface.vertices_[0]);
  int B_num_clusters = get_msphere_num_clusters(mface.vertices_[1]);
  int C_num_clusters = get_msphere_num_clusters(mface.vertices_[2]);
  int min_cluster_size =
      std::min({A_num_clusters, B_num_clusters, C_num_clusters});
  int max_cluster_size =
      std::max({A_num_clusters, B_num_clusters, C_num_clusters});
  int num_mface_clusters = get_num_clusters(ABC_mface_clusters);

  if (is_debug_thread)
    printf(
        "[mface_same_sheet_KNN] mface (%d,%d,%d), ABC_mface_clusters: %zu, "
        "num_mface_clusters: %d, min_cluster_size: %d, max_cluster_size: %d\n",
        mface.vertices_[0], mface.vertices_[1], mface.vertices_[2],
        ABC_mface_clusters.size(), num_mface_clusters, min_cluster_size,
        max_cluster_size);

  // special case: all 3 spheres are on the INTF, but on different INTFs
  if (min_cluster_size > 2 && num_mface_clusters > 1) {
    if (is_debug_thread)
      printf(
          "[mface_same_sheet_KNN] mface (%d,%d,%d) is on the same sheet, all "
          "on INTF, min_cluster_size: %d, min_cluster_size: %d\n",
          mface.vertices_[0], mface.vertices_[1], mface.vertices_[2],
          min_cluster_size, min_cluster_size);
    mface.is_on_same_sheet = true;
    return;
  }

  // normal case
  if (num_mface_clusters == max_cluster_size ||
      num_mface_clusters == min_cluster_size) {
    if (is_debug_thread)
      printf(
          "[mface_same_sheet_KNN] mface (%d,%d,%d) is on the same sheet, "
          "num_mface_clusters: %d, min_cluster_size: %d\n",
          mface.vertices_[0], mface.vertices_[1], mface.vertices_[2],
          num_mface_clusters, min_cluster_size);
    mface.is_on_same_sheet = true;
    return;
  }

  // default MedialFace::is_on_same_sheet = false
  return;
}

// NOT THREAD-SAFE
// since we use expand in join_two_clusters()
void RPD3D_Wrapper::mark_one_adj_mfaces_same_sheet_by_adj_bface(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
    const std::vector<std::vector<SFCluster>>& mface_clusters,
    std::vector<std::vector<SFCluster>>& mface_adj_clusters, MedialMesh& mmesh,
    int fid, bool is_debug) {
  assert(sphere_clusters.size() == mmesh.vertices->size());
  assert(mface_clusters.size() == mmesh.faces.size());

  auto& mface = mmesh.faces.at(fid);
  if (mface.is_deleted) return;
  if (is_debug)
    printf(
        "[mark_one_adj_mfaces_same_sheet_by_adj_bface] marking mface "
        "(%d,%d,%d) adj_mfaces...\n",
        mface.vertices_[0], mface.vertices_[1], mface.vertices_[2]);

  //////////////////////////////////////////////////////////////
  auto m1_join_clusters = mface_clusters.at(fid);  // copy
  if (m1_join_clusters.empty()) return;
  // clear SFCluster::fids_expand, to expand again
  for (auto& c : m1_join_clusters) c.fids_expand.clear();
  std::vector<SFCluster> m1m2_join_clusters;
  for (int eid : mface.edges_) {
    const auto& medge = mmesh.edges.at(eid);
    if (medge.is_deleted) continue;
    if (is_debug)
      printf(
          "[mark_one_adj_mfaces_same_sheet_by_adj_bface] checking one medge %d "
          "(%d,%d)\n",
          eid, medge.vertices_[0], medge.vertices_[1]);
    if (!medge.is_on_same_sheet) continue;
    for (int adj_fid : medge.faces_) {
      if (adj_fid == mface.fid) continue;
      auto& adj_mface = mmesh.faces.at(adj_fid);
      if (adj_mface.is_deleted) continue;
      if (!adj_mface.is_on_same_sheet) continue;
      aint2 f1f2 = {fid, adj_fid};
      std::sort(f1f2.begin(), f1f2.end());
      int f1f2_idx =
          get_upper_tri_matrix_idx(f1f2[0], f1f2[1], mmesh.faces.size());
      // already updated
      if (mmesh.is_two_faces_on_the_same_sheet[f1f2_idx] != -1) continue;
      mmesh.is_two_faces_on_the_same_sheet[f1f2_idx] = 0;
      // join two clusters
      auto m2_join_clusters = mface_clusters.at(adj_fid);  // copy
      m1m2_join_clusters.clear();
      join_two_clusters(tet_mesh, sf_mesh, true /*is_expand_fids*/,
                        m1_join_clusters, m2_join_clusters, m1m2_join_clusters,
                        false /*is_debug*/);
      if (is_debug) {
        printf(
            "[mark_one_adj_mfaces_same_sheet_by_adj_bface] two mfaces "
            "(%d,%d), mface1 %d (%d,%d,%d), mface2 %d (%d,%d,%d) \n",
            f1f2[0], f1f2[1], mface.fid, mface.vertices_[0], mface.vertices_[1],
            mface.vertices_[2], adj_fid, adj_mface.vertices_[0],
            adj_mface.vertices_[1], adj_mface.vertices_[2]);
        printf("===== m1_join_clusters: \n");
        for (const auto& c : m1_join_clusters) c.print();
        printf("===== m2_join_clusters: \n");
        for (const auto& c : m2_join_clusters) c.print();
        printf("===== m1m2_join_clusters: \n");
        for (const auto& c : m1m2_join_clusters) c.print();
      }
      // assign if not empty
      if (!m1m2_join_clusters.empty()) {
        mface_adj_clusters[f1f2_idx] = m1m2_join_clusters;
      }

      // normal case: m1_join_clusters.size() == m2_join_clusters.size()
      int min_num_clusters =
          std::min(m1_join_clusters.size(), m2_join_clusters.size());
      if (min_num_clusters == m1m2_join_clusters.size()) {
        mmesh.is_two_faces_on_the_same_sheet[f1f2_idx] = 1;
        if (is_debug)
          printf(
              "[mark_one_adj_mfaces_same_sheet_by_adj_bface] two mfaces "
              "(%d,%d) is on the same sheet\n",
              f1f2[0], f1f2[1], adj_mface.vertices_[2]);
      } else {
        if (is_debug)
          printf(
              "[mark_one_adj_mfaces_same_sheet_by_adj_bface] two mfaces "
              "(%d,%d) is NOT on the same sheet\n",
              f1f2[0], f1f2[1], adj_mface.vertices_[2]);
      }
    }  // for adj_fid
  }  // for eid
}

void RPD3D_Wrapper::cluster_all(const TetMesh& tet_mesh,
                                const SurfaceMesh& sf_mesh, MedialMesh& mmesh,
                                bool is_debug) {
  this->cluster_medges_same_sheet(tet_mesh, sf_mesh, mmesh, false /*is_debug*/);
  this->cluster_mfaces_same_sheet(tet_mesh, sf_mesh, mmesh, false /*is_debug*/);
  this->cluster_mfaces_adj_same_sheet(tet_mesh, sf_mesh, mmesh,
                                      false /*is_debug*/);
  this->cluster_medges_type(tet_mesh, sf_mesh, mmesh, false /*is_debug*/);
}

// update
// 1. RPD3D_Wrapper::medge_clusters
// 2. MedialEdge::is_on_same_sheet
// 3. MedialEdge::is_extf
void RPD3D_Wrapper::cluster_medges_same_sheet(const TetMesh& tet_mesh,
                                              const SurfaceMesh& sf_mesh,
                                              MedialMesh& mmesh,
                                              bool is_debug) {
  if (this->sphere_clusters.empty()) {
    printf("[cluster_medges_same_sheet] sphere_clusters is empty\n");
    assert(false);
  }
  this->medge_clusters.clear();
  this->medge_clusters.resize(mmesh.edges.size());
  GEO::parallel_for(0, mmesh.edges.size(), [&](int eid) {
    this->mark_one_medge_same_sheet_by_adj_bface(
        tet_mesh, sf_mesh, this->sphere_clusters, this->medge_clusters, mmesh,
        eid, false /*is_debug*/);
  });

  // for (int eid = 0; eid < mmesh.edges.size(); eid++) {
  //   // this->mark_one_medge_same_sheet_by_adj_bface(
  //   //     tet_mesh, sf_mesh, this->sphere_clusters, this->medge_clusters,
  //   //     mmesh, eid, true /*is_debug*/);
  // }
}

void RPD3D_Wrapper::cluster_mfaces_same_sheet(const TetMesh& tet_mesh,
                                              const SurfaceMesh& sf_mesh,
                                              MedialMesh& mmesh,
                                              bool is_debug) {
  if (is_debug)
    printf("------------------ start cluster_mfaces_same_sheet ....\n");
  if (this->medge_clusters.empty()) {
    printf(
        "[cluster_mfaces_same_sheet] medge_clusters is empty, call "
        "cluster_medges_same_sheet() first\n");
    this->cluster_medges_same_sheet(tet_mesh, sf_mesh, mmesh, is_debug);
  }
  ////////////////////////////
  this->mface_clusters.clear();
  this->mface_clusters.resize(mmesh.faces.size());
  GEO::parallel_for(0, mmesh.faces.size(), [&](int fid) {
    this->mark_one_mface_same_sheet_by_adj_bface(
        tet_mesh, sf_mesh, this->sphere_clusters, this->mface_clusters, mmesh,
        fid, false /*is_debug*/);
    // this->mark_one_mface_same_sheet_by_adj_bface_KNN(
    //     tet_mesh, sf_mesh, this->sphere_clusters, this->mface_clusters,
    //     mmesh, fid, false /*is_debug*/);
  });
  // for (int f = 0; f < mmesh.faces.size(); f++) {
  //   this->mark_one_mface_same_sheet_by_adj_bface_KNN(
  //       tet_mesh, sf_mesh, this->sphere_clusters, this->mface_clusters,
  //       mmesh, f, false /*is_debug*/);
  // }
}

void RPD3D_Wrapper::cluster_mfaces_adj_same_sheet(const TetMesh& tet_mesh,
                                                  const SurfaceMesh& sf_mesh,
                                                  MedialMesh& mmesh,
                                                  bool is_debug) {
  if (is_debug)
    printf("------------------ start cluster_mfaces_adj_same_sheet ....\n");
  assert(mmesh.vertices != nullptr);
  assert(this->sphere_clusters.size() == mmesh.vertices->size());
  if (this->mface_clusters.empty()) {
    printf(
        "[cluster_mfaces_adj_same_sheet] mface_clusters is empty, call "
        "cluster_mfaces_same_sheet() first\n");
    this->cluster_mfaces_same_sheet(tet_mesh, sf_mesh, mmesh, is_debug);
  }
  ////////////////////////////
  this->mface_adj_clusters.clear();
  this->mface_adj_clusters.resize(
      get_upper_tri_matrix_size(mmesh.faces.size()));
  mmesh.is_two_faces_on_the_same_sheet.clear();
  mmesh.is_two_faces_on_the_same_sheet.resize(
      get_upper_tri_matrix_size(mmesh.faces.size()), -1);
  // GEO::parallel_for(0, mmesh.faces.size(), [&](int fid) {
  //   this->mark_one_adj_mfaces_same_sheet_by_adj_bface(
  //       tet_mesh, sf_mesh, this->mface_clusters, this->mface_adj_clusters,
  //       mmesh, fid, false /*is_debug*/);
  // });
  for (int f = 0; f < mmesh.faces.size(); f++) {
    // NOT THREAD-SAFE since we use expand in join_two_clusters()
    this->mark_one_adj_mfaces_same_sheet_by_adj_bface(
        tet_mesh, sf_mesh, this->mface_clusters, this->mface_adj_clusters,
        mmesh, f, false /*is_debug*/);
  }
}

// mark MedialEdge::is_intf and MedialEdge::is_extf
void RPD3D_Wrapper::mark_one_medge_type(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
    const std::vector<std::vector<SFCluster>>& mface_adj_clusters,
    MedialMesh& mmesh, int eid, bool is_debug) {
  assert(mmesh.edges.size() > eid);
  auto& medge = mmesh.edges.at(eid);
  if (medge.is_deleted) return;
  if (is_debug)
    printf("[mark_one_medge_type] marking medge type (%d,%d) ..\n",
           medge.vertices_[0], medge.vertices_[1]);

  //////////////////////////////////////////////////////////////
  // check all adjacent mfaces
  // count the number of mfaces NOT on the same sheet
  std::set<aint2> visited_two_fids_pairs;
  std::set<aint2> two_fids_not_on_the_same_sheet;
  for (int fid1 : medge.faces_) {
    if (mmesh.faces.at(fid1).is_deleted) continue;
    if (!mmesh.faces.at(fid1).is_on_same_sheet) continue;
    if (is_debug)
      printf("[mark_one_medge_type] medge (%d,%d) found fid1: %d\n",
             medge.vertices_[0], medge.vertices_[1], fid1);

    for (int fid2 : medge.faces_) {
      if (fid1 == fid2) continue;
      if (mmesh.faces.at(fid2).is_deleted) continue;
      if (!mmesh.faces.at(fid2).is_on_same_sheet) continue;
      aint2 f1f2 = {fid1, fid2};
      std::sort(f1f2.begin(), f1f2.end());
      if (visited_two_fids_pairs.find(f1f2) != visited_two_fids_pairs.end())
        continue;
      visited_two_fids_pairs.insert(f1f2);
      int f1f2_idx =
          get_upper_tri_matrix_idx(f1f2[0], f1f2[1], mmesh.faces.size());
      // check both adjacent mfaces are on the same sheet
      // and two spheres are intf spheres
      const auto& medge_s1 = this->all_medial_spheres->at(medge.vertices_[0]);
      const auto& medge_s2 = this->all_medial_spheres->at(medge.vertices_[1]);
      bool is_candidate = (medge_s1.is_on_intf() && medge_s2.is_on_intf()) ||
                          (medge_s1.is_on_intf() && medge_s2.is_on_corner()) ||
                          (medge_s1.is_on_corner() && medge_s2.is_on_intf());
      if (mmesh.is_two_faces_on_the_same_sheet[f1f2_idx] == 0 && is_candidate) {
        two_fids_not_on_the_same_sheet.insert(f1f2);
      }
      if (is_debug)
        printf(
            "[mark_one_medge_type] medge (%d,%d) found fid2: %d, "
            "mmesh.is_two_faces_on_the_same_sheet[f1f2_idx]: %d\n",
            medge.vertices_[0], medge.vertices_[1], fid2,
            mmesh.is_two_faces_on_the_same_sheet[f1f2_idx]);
    }  // for f2
  }  // for f1

  if (is_debug) {
    printf(
        "[mark_one_medge_type] medge (%d,%d) has "
        "two_fids_not_on_the_same_sheet size %zu: [",
        medge.vertices_[0], medge.vertices_[1],
        two_fids_not_on_the_same_sheet.size());
    // print two_fids_not_on_the_same_sheet
    for (const auto& f1f2 : two_fids_not_on_the_same_sheet) {
      printf("(%d,%d), ", f1f2[0], f1f2[1]);
    }
    printf("]\n");
  }

  // at least two adjacent mfaces (in aint2) are not on the same sheet
  // so this medge is on the INTF, as it is the intersection of two sheets
  if (two_fids_not_on_the_same_sheet.size() > 0) {
    medge.is_intf = true;
    if (is_debug)
      printf("[mark_one_medge_type] medge (%d,%d) is on the INTF\n",
             medge.vertices_[0], medge.vertices_[1]);
  } else {
    if (is_debug)
      printf("[mark_one_medge_type] medge (%d,%d) is NOT on the INTF\n",
             medge.vertices_[0], medge.vertices_[1]);
  }

  // check and mark EXTF
  const MedialSphere& A = mmesh.vertices->at(medge.vertices_[0]);
  const MedialSphere& B = mmesh.vertices->at(medge.vertices_[1]);
  if (!A.is_on_extf() || !B.is_on_extf()) return;
  if (is_two_mspheres_on_same_sl_including_corners(A, B)) {
    medge.is_extf = true;
  }
}

void RPD3D_Wrapper::cluster_medges_type(const TetMesh& tet_mesh,
                                        const SurfaceMesh& sf_mesh,
                                        MedialMesh& mmesh, bool is_debug) {
  if (is_debug) printf("------------------ start cluster_medges_type ....\n");
  GEO::parallel_for(0, mmesh.edges.size(), [&](int eid) {
    mark_one_medge_type(tet_mesh, sf_mesh, this->mface_adj_clusters, mmesh, eid,
                        false /*is_debug*/);
  });
  // for (int e = 0; e < mmesh.edges.size(); e++) {
  //   mark_one_medge_type(tet_mesh, sf_mesh, this->mface_adj_clusters, mmesh,
  //   e, true /*is_debug*/);
  // }
}

void RPD3D_Wrapper::get_one_pcell_in_tets(
    ConvexCellHost& cc_trans, std::vector<cfloat3>& voro_points,
    std::vector<std::array<int, 4>>& voro_tets,
    std::vector<int>& voro_tets_sites, std::vector<int>& voro_tets_cell_ids) {
  if (!cc_trans.is_pc_explicit) cc_trans.reload_pc_explicit();
  assert(cc_trans.is_pc_explicit);
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
        voro_tets_cell_ids.push_back(cc_trans.id);
      }
      lf++;
    }
  }
}

// Compute RPD for given spheres in 4D (given x,y,z,r)
// Note: not all spheres are valid during RT,
//       invalid spheres will have 0 neighbors (-1 in site_knn),
//       so GPU should be able to handle it.
//
// Note: this function will call RPD3D_Wrapper::reset_deleted_spheres()
//       to make all spheres non-deleted before calling RT
void RPD3D_Wrapper::calculate_given_spheres(
    const std::vector<float>& _spheres /*4D (x,y,z,r)*/, bool is_debug) {
  const clock_t start_t = clock();
  // make all spheres non-deleted
  // only let RT decide with sphere to delete
  this->reset_deleted_spheres();
  // clear all MedialSphere::pcell
  // this will be updated calling RPD3D_GPU::update_spheres_power_cells()
  this->clear_spheres_topos();

  // compute RT
  int num_spheres = _spheres.size() / 4;
  generate_RT_CGAL_given_spheres(*this->params, _spheres, this->rt, is_debug);
  if (is_debug) printf("generate_RT_CGAL_given_spheres done\n");
  // load site_knn
  std::vector<int> is_sphere_valid;
  this->site_k = get_RT_spheres_and_neighbors(
      num_spheres, this->rt, this->site_knn, is_sphere_valid, is_debug);
  assert(is_sphere_valid.size() == num_spheres);
  assert(this->site_k > 0);
  // update site, site_weights, site_flags
  this->load_spheres_to_sites(_spheres, is_sphere_valid);

  //////////////////////
  // compute parial RPD using CUDA
  // make sure load_tet_adj_info() was called ahead
  //
  // site_flags: store (sphere_N + 1ring)
  // site:       store (sphere_N + 1ring + 2ring)
  assert(!this->tet_mesh->v_adjs.empty() && !this->tet_mesh->e_adjs.empty() &&
         !this->tet_mesh->f_adjs.empty() && !this->tet_mesh->f_ids.empty());
  powercells.clear();
  powercells = compute_clipped_voro_diagram_GPU(
      this->num_itr_rpd, this->tet_mesh->tet_vertices,
      this->tet_mesh->tet_indices, this->tet_mesh->v2tets,
      this->tet_mesh->v_adjs, this->tet_mesh->e_adjs, this->tet_mesh->f_adjs,
      this->tet_mesh->f_ids, this->site, this->n_site, this->site_weights,
      this->site_flags, this->site_knn, this->site_k, site_cell_vol,
      true /*site_is_transposed*/, 1 /*nb_iter*/, num_spheres);
  if (is_debug) printf("compute compute_clipped_voro_diagram_GPU done \n");
  this->num_itr_rpd++;
  std::cout << "[RPD] calculate RPD given spheres took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

// clear MedialSphere::pcell
void RPD3D_Wrapper::clear_spheres_topos() {
  for (auto& msphere : *this->all_medial_spheres) {
    msphere.topo_clear();
  }
}

void RPD3D_Wrapper::reset_deleted_spheres() {
  for (auto& msphere : *this->all_medial_spheres) {
    msphere.is_deleted = false;
  }
}

// will update, using this->spheres
// 1. this->site
// 2. this->site_weights
// 3. this->site_flags
// 4. this->n_site
void RPD3D_Wrapper::load_spheres_to_sites(
    const std::vector<float>& spheres /*4D (x,y,z,r)*/,
    const std::vector<int>& is_sphere_valid) {
  this->site.clear();
  this->site_weights.clear();
  this->site_flags.clear();
  int dim = 4;
  this->n_site = spheres.size() / dim;
  // std::cout << "n_site: " << this->n_site << ", dim: " << dim << std::endl;
  assert(is_sphere_valid.size() == this->n_site);

  this->site.resize(this->n_site * 3);
  this->site_weights.resize(this->n_site);
  this->site_flags.resize(this->n_site);
  for (int i = 0; i < this->n_site; ++i) {
    this->site[i] = spheres.at(i * 4);
    this->site[i + this->n_site] = spheres.at(i * 4 + 1);
    this->site[i + (this->n_site << 1)] = spheres.at(i * 4 + 2);

    // add weight (sq_radius) info
    this->site_weights[i] = std::pow(spheres.at(i * 4 + 3), 2);

    // add flag for sphere
    this->site_flags[i] = SiteFlag::no_flag;
    if (is_sphere_valid[i] == 1) this->site_flags[i] = SiteFlag::is_selected;

    // printf("site %d has sphere: (%lf %lf %lf %lf) with flag %d \n", i,
    //        msphere.center[0], msphere.center[1], msphere.center[2],
    //        msphere.radius, site_flags[i]);
  }
}

#include "updating.h"
bool RPD3D_Wrapper::project_one_TN_sphere(MedialSphere& msphere,
                                          bool is_debug) {
  if (is_debug) {
    printf("projecting sphere %d...\n", msphere.id);
    msphere.print_tan_planes();
    msphere.print_tan_cc_lines();
  }

  MedialSphere new_msphere = msphere;
  iter_params itr_params;
  if (this->params->is_run_cad) {  // CAD
    // Note:
    // this is more strict than is_break_use_energy_over_sq_radius = true
    itr_params.is_break_use_energy_over_sq_radius = false;
    itr_params.break_threshold = 0.5;
  }
  bool is_good = iterate_sphere((*this->sf_mesh), (*this->sf_mesh).aabb_wrapper,
                                (*this->sf_mesh).fe_sf_fs_pairs,
                                (*this->tet_mesh).feature_edges, new_msphere,
                                is_debug /*is_debug*/, itr_params);
  // for debug, set it to be true
  if (is_debug) is_good = true;
  if (!is_good) return false;
  if (is_debug)
    printf(
        "[PROJ] sphere %d type %d, projected (%f,%f,%f,%f) -> "
        "(%f,%f,%f,%f)\n",
        msphere.id, msphere.type, msphere.center[0], msphere.center[1],
        msphere.center[2], msphere.radius, new_msphere.center[0],
        new_msphere.center[1], new_msphere.center[2], new_msphere.radius);

  // replace msphere with new_mspher
  msphere = new_msphere;
  return true;
}

// update TangentPlane/TangentConcaveLine
void RPD3D_Wrapper::update_one_sphere_tangents_given_clusters(
    const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
    const std::vector<SFCluster>& given_clusters, MedialSphere& msphere,
    bool is_debug) const {
  // we may adding new spheres
  // assert(this->sphere_pcell_samples.size() ==
  // this->all_medial_spheres->size());
  if (given_clusters.empty()) return;

  double filter_threshold = 0.15;
  int sphere_id = msphere.id;
  msphere.tan_planes.clear();
  msphere.tan_cc_lines.clear();
  if (is_debug) printf("[SPH_TAN] handling sphere %d...\n", sphere_id);

  // loop each cluster
  std::map<int, int> saved_cl_ids;
  for (const auto& cluster : given_clusters) {
    if (is_debug) {
      printf("[SPH_TAN] handling sphere %d cluster: %d\n", sphere_id,
             cluster.id);
      cluster.print();
    }

    // get sphere_id, either A or B
    assert(cluster.pid_sphere_id != -1);
    int pid_sphere_id = cluster.pid_sphere_id;
    assert(pid_sphere_id < this->sphere_pcell_samples.size());
    const auto& sphere_pcell_samples =
        this->sphere_pcell_samples.at(pid_sphere_id);

    // PART1: add TangentPlane or TangentConcaveLine
    int min_pid = -1;
    double min_dist_to_sphere = DBL_MAX;
    // find the sample closest to the sphere center
    for (const int& pid : cluster.pids) {
      const auto& one_pcell_sample = sphere_pcell_samples.at(pid);
      const Vector3& sample_proj = one_pcell_sample.proj;
      const Vector3& sample_normal = one_pcell_sample.normal;
      double dist = GEO::length(msphere.center +
                                msphere.radius * sample_normal - sample_proj);
      if (dist < min_dist_to_sphere) {
        min_dist_to_sphere = dist;
        min_pid = pid;
      }
    }  // for each sample
    assert(min_pid != -1);
    const auto& min_pcell_sample = sphere_pcell_samples.at(min_pid);
    const Vector3& sample_proj = min_pcell_sample.proj;
    const Vector3& sample_normal = min_pcell_sample.normal;
    const int min_ce_id = min_pcell_sample.proj_FE_id;
    if (min_ce_id == -1) {
      // save the tangent plane
      int fid = min_pcell_sample.proj_fid;
      TangentPlane tan_pl(sf_mesh, sample_normal, sample_proj, fid);
      msphere.tan_planes.push_back(tan_pl);
      if (is_debug)
        printf(
            "[SPH_TAN] sphere %d add tangent plane with min_pid: %d from "
            "pid_sphere_id: %d, fid: %d\n",
            sphere_id, min_pid, pid_sphere_id, fid);
    } else {
      // save the tangent concave line
      const auto& ce = tet_mesh.feature_edges.at(min_ce_id);
      TangentConcaveLine tan_ce(sf_mesh, ce);
      tan_ce.tan_point = sample_proj;
      tan_ce.normal = sample_normal;
      msphere.tan_cc_lines.push_back(tan_ce);
      if (is_debug)
        printf(
            "[SPH_TAN] sphere %d add tangent concave line with min_pid: %d "
            "from pid_sphere_id: %d, min_ce_id: %d, cl_id: %d\n",
            sphere_id, min_pid, pid_sphere_id, min_ce_id, ce.get_fl_id());
    }

    // PART2: add TangentPlane for INF
    //        pick the point normal further away from min_pcell_sample
    if (msphere.is_on_inf() && is_inf_more_than_T1(cluster.is_normal_inf)) {
      double second_min_pid = -1;
      double second_score = -std::numeric_limits<double>::infinity();
      for (const int& pid : cluster.pids) {
        const auto& one_pcell_sample = sphere_pcell_samples.at(pid);
        double dist =
            GEO::length(min_pcell_sample.proj - one_pcell_sample.proj);
        double normal_diff =
            normal_angle_diff(min_pcell_sample.normal, one_pcell_sample.normal);
        double score = dist + normal_diff;
        if (score > second_score) {
          second_min_pid = pid;
          second_score = score;
        }
      }  // for pids
      assert(second_min_pid != -1);
      const auto& selected_sample = sphere_pcell_samples.at(second_min_pid);
      TangentPlane tan_pl(sf_mesh, selected_sample.normal, selected_sample.proj,
                          selected_sample.proj_fid);
      msphere.tan_planes.push_back(tan_pl);
    }

    // PART3: add other potential TangentConcaveLine
    //        each cluster may contains multiple concave lines
    if (!cluster.ce_line_ids.empty()) {
      std::vector<int> pid_per_cl, ce_id_per_cl;
      std::vector<double> min_dist_per_cl;
      // find the CE sample closest to the sphere center
      for (const int& pid : cluster.pids) {
        const auto& one_pcell_sample = sphere_pcell_samples.at(pid);
        int cur_per_idx = pid_per_cl.size();
        int ce_id = one_pcell_sample.proj_FE_id;
        if (ce_id == -1) continue;
        if (ce_id == min_ce_id) continue;  // already saved
        int cl_id = tet_mesh.get_fl_id(ce_id);
        assert(cluster.ce_line_ids.find(cl_id) != cluster.ce_line_ids.end());
        if (saved_cl_ids.find(cl_id) != saved_cl_ids.end()) {
          cur_per_idx = saved_cl_ids.at(cl_id);
        } else {
          saved_cl_ids[cl_id] = cur_per_idx;
          pid_per_cl.push_back(-1);
          ce_id_per_cl.push_back(-1);
          min_dist_per_cl.push_back(DBL_MAX);
        }
        const Vector3& sample_proj = one_pcell_sample.proj;
        const Vector3& sample_normal = one_pcell_sample.normal;
        double dist = GEO::length(msphere.center +
                                  msphere.radius * sample_normal - sample_proj);
        if (dist < min_dist_per_cl[cur_per_idx]) {
          pid_per_cl[cur_per_idx] = pid;
          min_dist_per_cl[cur_per_idx] = dist;
          ce_id_per_cl[cur_per_idx] = ce_id;
        }
      }  // for each sample
      assert(pid_per_cl.size() == min_dist_per_cl.size());
      assert(pid_per_cl.size() == ce_id_per_cl.size());
      // save tangent info
      for (int i = 0; i < pid_per_cl.size(); i++) {
        int pid = pid_per_cl.at(i);
        int ce_id = ce_id_per_cl.at(i);
        double min_dist_to_ce = min_dist_per_cl.at(i);
        double radius_threshold = msphere.radius * filter_threshold;
        if (is_debug)
          printf(
              "[SPH_TAN] pid %d, ce_id %d, min_dist_to_ce: %f, "
              "radius_threshold: %f\n",
              pid, ce_id, min_dist_to_ce, radius_threshold);
        assert(pid != -1 && ce_id != -1);
        // NOTE: sometimes spheres are way too far away from CE
        //       including this tangent info may allow interate_sphere()
        //       generate duplicated results.
        if (min_dist_to_ce >= radius_threshold) continue;
        const Vector3& sample_proj = sphere_pcell_samples.at(pid).proj;
        const Vector3& sample_normal = sphere_pcell_samples.at(pid).normal;
        // const Vector3& sample_normal =
        //     GEO::normalize(sample_proj - msphere.center);
        const auto& ce = tet_mesh.feature_edges.at(ce_id);
        TangentConcaveLine tan_ce(sf_mesh, ce);
        tan_ce.tan_point = sample_proj;
        tan_ce.normal = sample_normal;
        msphere.tan_cc_lines.push_back(tan_ce);
        if (is_debug)
          printf(
              "[SPH_TAN] sphere %d add tangent ce_id: %d, cl_id: %d from "
              "pid_sphere_id: %d\n",
              sphere_id, ce_id, ce.get_fl_id(), pid_sphere_id);
      }
    }
  }  // for each cluster
}

// update TangentPlane/TangentConcaveLine
// must call after RPD3D_Wrapper::cluster_sphere_type_using_samples_fids()
void RPD3D_Wrapper::update_all_spheres_tangents_use_clusters(
    bool is_debug) const {
  assert(sphere_clusters.size() == this->all_medial_spheres->size());

  auto run_thread_sphere = [&](int sphere_id) {
    auto& msphere = this->all_medial_spheres->at(sphere_id);
    if (msphere.is_deleted) return;
    if (msphere.is_on_se() || msphere.is_on_corner()) return;
    if (sphere_clusters.at(sphere_id).empty()) return;
    this->update_one_sphere_tangents_given_clusters(
        *this->tet_mesh, *this->sf_mesh, sphere_clusters.at(sphere_id), msphere,
        is_debug);
  };

  // for (int sphere_id = 0; sphere_id < this->all_medial_spheres->size();
  //      sphere_id++) {
  //   run_thread_sphere(sphere_id);
  // }
  GEO::parallel_for(0, this->all_medial_spheres->size(),
                    [&](int sphere_id) { run_thread_sphere(sphere_id); });
}

// compute mmesh inside
// update Parameters::mmesh_avg_len
void RPD3D_Wrapper::compute_and_update_params_mmesh_avg_length(
    MedialMesh& mmesh) {
  // if (mmesh.vertices == nullptr) {
  //   this->update_spheres_power_cells(false /*is_compute_se_sfids*/);
  //   mmesh.clear();
  //   mmesh.genearte_medial_spheres(*this->all_medial_spheres);
  //   mmesh.generate_medial_edges(*this->sf_mesh,
  //                               false /*is_compute_common_diff_sfids*/);
  // }
  printf("[compute_mmesh_avg_length] mmesh has %zu edges\n",
         mmesh.edges.size());
  assert(mmesh.vertices != nullptr);
  // compute average edge length
  this->mmesh_avg_len = 0;
  for (int meid = 0; meid < mmesh.edges.size(); meid++) {
    const MedialEdge& medge = mmesh.edges.at(meid);
    const MedialSphere& A = mmesh.vertices->at(medge.vertices_[0]);
    const MedialSphere& B = mmesh.vertices->at(medge.vertices_[1]);
    double edge_len = GEO::length(A.center - B.center);
    this->mmesh_avg_len += edge_len;
  }
  this->mmesh_avg_len /= mmesh.edges.size();
  this->mmesh_avg_len *= 1.5;
  printf("[compute_mmesh_avg_length] update mmesh_avg_len %f\n",
         this->mmesh_avg_len);
}

void RPD3D_Wrapper::eval_mmesh_medge_length(MedialMesh& mmesh, bool is_debug) {
  printf("[eval_medges] mmesh has %zu edges\n", mmesh.edges.size());
  assert(mmesh.vertices != nullptr);
  this->is_medge_add_new_sphere.clear();
  this->is_medge_add_new_sphere.resize(mmesh.edges.size(), 0);
  // make sure to cluster medge ahead
  this->cluster_medges_same_sheet(*this->tet_mesh, *this->sf_mesh, mmesh, false
                                  /*is_debug*/);

  // must update the Parameter::mmesh_avg_len
  // in update_params_mmesh_avg_length()
  if (this->mmesh_avg_len == -1)
    compute_and_update_params_mmesh_avg_length(mmesh);
  assert(this->mmesh_avg_len > -1);

  // parallel
  auto eval_medge_len = [&](int meid) {
    assert(meid > -1 && meid < mmesh.edges.size());
    const MedialEdge& medge = mmesh.edges.at(meid);
    const MedialSphere& A = mmesh.vertices->at(medge.vertices_[0]);
    const MedialSphere& B = mmesh.vertices->at(medge.vertices_[1]);

    bool is_debug_thread = false;
    // if (A.id == 450 && B.id == 687 || A.id == 687 && B.id == 450)
    //   is_debug_thread = true;

    // we only consider medge on the same sheet
    if (!medge.is_on_same_sheet) return;

    double edge_len = GEO::length(A.center - B.center);
    if (edge_len > this->mmesh_avg_len) {
      this->is_medge_add_new_sphere[meid] = 1;
    }
    if (is_debug_thread)
      printf(
          "[eval_medges] medge %d (%d,%d) len %f<%f need to add new sphere, "
          "A id: %d, B id: %d\n",
          meid, medge.vertices_[0], medge.vertices_[1], edge_len,
          this->mmesh_avg_len, A.id, B.id);
  };

  // for (int meid = 0; meid < mmesh.edges.size(); meid++) {
  //   eval_medge_len(meid);
  // }
  GEO::parallel_for(0, mmesh.edges.size(),
                    [&](int meid) { eval_medge_len(meid); });

  int num_not_block = 0;
  for (int meid = 0; meid < mmesh.edges.size(); meid++) {
    if (this->is_medge_add_new_sphere[meid] > 0) num_not_block++;
  }
  printf("[eval_medges] total %d medges do not block\n", num_not_block);
}

void RPD3D_Wrapper::add_new_sphere_after_eval_medge_length(
    const MedialMesh& mmesh, bool is_debug) {
  auto update_sphere_radius = [&](MedialSphere& msphere) {
    double sq_dist =
        this->sf_mesh->aabb_wrapper.get_sq_dist_to_sf(msphere.center);
    msphere.radius = std::sqrt(sq_dist);
  };

  int num_add_sphere = 0;
  std::set<int> processed_sphere_ids;
  for (int meid = 0; meid < mmesh.edges.size(); meid++) {
    const MedialEdge& medge = mmesh.edges.at(meid);
    const auto& A_id = medge.vertices_[0];
    const auto& B_id = medge.vertices_[1];
    const auto& A = this->all_medial_spheres->at(medge.vertices_[0]);
    const auto& B = this->all_medial_spheres->at(medge.vertices_[1]);
    // no need to add new sphere, or no need to handle here
    if (this->is_medge_add_new_sphere.at(meid) < 1) continue;
    // if added, then skip, do not add too many spheres
    // if (processed_sphere_ids.find(A_id) != processed_sphere_ids.end() ||
    //     processed_sphere_ids.find(B_id) != processed_sphere_ids.end())
    //   continue;
    if (is_debug)
      printf(
          "[medge_add_spheres] medge %d (%d,%d), is_medge_add_new_sphere: %d\n",
          meid, medge.vertices_[0], medge.vertices_[1],
          this->is_medge_add_new_sphere.at(meid));

    // add a new sphere in the center of A and B
    Vector3 new_center = 0.5 * (A.center + B.center);
    double new_radius = 0.5 * (A.radius + B.radius);
    MedialSphere new_msphere(this->all_medial_spheres->size(), new_center,
                             new_radius, SphereType::T_2);
    update_sphere_radius(new_msphere);
    this->all_medial_spheres->push_back(new_msphere);
    num_add_sphere++;
    // // only marked processed when success
    // processed_sphere_ids.insert(A_id);
    // processed_sphere_ids.insert(B_id);
  }
  printf("[medge_add_spheres] medges added %d/%zu new spheres\n",
         num_add_sphere, this->all_medial_spheres->size());
}

// must call RPD3D_Wrapper::cluster_sphere_type_using_samples_fids() ahead
void RPD3D_Wrapper::eval_mmesh_cand_medge_not_same_sheet(MedialMesh& mmesh,
                                                         bool is_debug) {
  assert(!this->sphere_clusters.empty());
  assert(mmesh.vertices != nullptr);
  this->is_medge_add_new_sphere.clear();
  this->is_medge_add_new_sphere.resize(mmesh.edges.size(), 0);
  this->is_msphere_add_new_sphere.clear();
  this->is_msphere_add_new_sphere.resize(this->all_medial_spheres->size(),
                                         std::vector<v2int>());
  // make sure to cluster medge ahead
  this->cluster_medges_same_sheet(*this->tet_mesh, *this->sf_mesh, mmesh, false
                                  /*is_debug*/);
  assert(!this->medge_clusters.empty());

  // parallel
  // step 1: check if T4 need to add new T2
  auto eval_cand_mspheres = [&](int msphere_id) {
    assert(msphere_id < this->is_msphere_add_new_sphere.size());
    const auto& msphere = this->all_medial_spheres->at(msphere_id);
    if (msphere.is_on_corner()) return;
    int num_clusters = get_msphere_num_clusters(msphere_id);
    if (num_clusters < 4) return;
    bool is_thread_debug = false;
    // if (msphere_id == 203) is_thread_debug = true;
    bool is_add = false;
    double dist_threshold = msphere.radius * 0.15;
    // loop tangent planes
    for (const auto& tan_pl : msphere.tan_planes) {
      double dist = std::abs(msphere.radius -
                             GEO::length(msphere.center - tan_pl.tan_point));
      if (dist <= dist_threshold) continue;
      if (is_thread_debug)
        printf(
            "[eval_mspheres] msphere_id %d, tan_pl fid %d, dist %f >"
            "dist_threshold: %f\n",
            msphere_id, tan_pl.fid, dist, dist_threshold);
      // need to add new T2 sphere,
      this->is_msphere_add_new_sphere[msphere_id].push_back(
          {tan_pl.tan_point, tan_pl.fid});
      is_add = true;
    }
    // loop tangent concave lines
    for (const auto& one_cc_line : msphere.tan_cc_lines) {
      double dist = std::abs(
          msphere.radius - GEO::length(msphere.center - one_cc_line.tan_point));
      if (dist <= dist_threshold) continue;
      if (is_thread_debug)
        printf(
            "[eval_mspheres] msphere_id %d, one_cc_line id_fe %d, dist %f >"
            "dist_threshold: %f\n",
            msphere_id, one_cc_line.id_fe, dist, dist_threshold);
      // need to add new T2 sphere on CE
      int ce_id_code =
          ss_params::encode_fid(one_cc_line.id_fe, true /*is_on_ce*/);
      this->is_msphere_add_new_sphere[msphere_id].push_back(
          {one_cc_line.tan_point, ce_id_code});
      is_add = true;
    }
    if (is_thread_debug && is_add)
      printf(
          "[eval_mspheres] sphere %d need to add new spheres, radius %f, "
          "dist_threshold: %f\n",
          msphere_id, msphere.radius, dist_threshold);
  };
  GEO::parallel_for(0, this->all_medial_spheres->size(),
                    [&](int msphere_id) { eval_cand_mspheres(msphere_id); });

  // parallel
  // step 2: check if two T2 need to add new INTF
  auto eval_cand_medges = [&](int meid) {
    assert(meid > -1 && meid < mmesh.edges.size());
    const MedialEdge& medge = mmesh.edges.at(meid);
    const MedialSphere& A = mmesh.vertices->at(medge.vertices_[0]);
    const MedialSphere& B = mmesh.vertices->at(medge.vertices_[1]);

    bool is_debug_thread = false;
    // if (A.id == 450 && B.id == 687 || A.id == 687 && B.id == 450)
    //   is_debug_thread = true;

    if (medge.is_on_same_sheet) return;
    // do not consider spheres
    // 1: for concave lines
    std::set<int> A_cl_ids, B_cl_ids;
    const auto& A_clusters = this->sphere_clusters.at(A.id);
    for (const auto& c : A_clusters) {
      if (c.ce_line_ids.empty()) continue;
      A_cl_ids.insert(c.ce_line_ids.begin(), c.ce_line_ids.end());
    }
    const auto& B_clusters = this->sphere_clusters.at(B.id);
    for (const auto& c : B_clusters) {
      if (c.ce_line_ids.empty()) continue;
      B_cl_ids.insert(c.ce_line_ids.begin(), c.ce_line_ids.end());
    }
    // // 1.1. only one crosses concave lines
    // if (!A_cl_ids.empty() && B_cl_ids.empty()) return;
    // if (A_cl_ids.empty() && !B_cl_ids.empty()) return;
    if (is_debug_thread) {
      printf("[eval_medges] A sphere %d contains concave lines %d\n", A.id,
             A_cl_ids.size());
      printf("[eval_medges] B sphere %d contains concave lines %d\n", B.id,
             B_cl_ids.size());
    }
    // 1.2. both cross the same concave line
    if (!A_cl_ids.empty() && !B_cl_ids.empty()) {
      std::vector<int> shared_cl_ids;
      set_intersection(A_cl_ids, B_cl_ids, shared_cl_ids);
      if (!shared_cl_ids.empty()) return;
      if (is_debug_thread)
        printf(
            "[eval_medges] A sphere %d and B sphere %d share concave lines "
            "%d\n",
            A.id, B.id, shared_cl_ids.size());
    }

    // do not consider sphere
    // 2: for INTF
    if (A.is_on_intf() && B.is_on_intf()) return;

    if (!A.is_on_intf() && !B.is_on_intf())
      // T2-T2 case, add new INTF
      this->is_medge_add_new_sphere[meid] = 1;
    else
      // T2-T3 case
      this->is_medge_add_new_sphere[meid] = 2;

    if (is_debug_thread)
      printf(
          "[eval_medges] medge %d (%d,%d) need to add new sphere, "
          "A id: %d, B id: %d\n",
          meid, medge.vertices_[0], medge.vertices_[1], A.id, B.id);
  };

  // for (int meid = 0; meid < mmesh.edges.size(); meid++) {
  //   eval_cand_medges(meid);
  // }
  GEO::parallel_for(0, mmesh.edges.size(),
                    [&](int meid) { eval_cand_medges(meid); });

  int num_not_block = 0;
  for (int meid = 0; meid < mmesh.edges.size(); meid++) {
    if (this->is_medge_add_new_sphere[meid] > 0) num_not_block++;
  }
  printf("[eval_medges] total %d medges do not block\n", num_not_block);
}

int RPD3D_Wrapper::add_T4_new_spheres_T2_helper(const int sphere_id,
                                                bool is_debug) {
  // // not true, we may already added new sphere
  // assert(this->is_msphere_add_new_sphere.size() ==
  //        this->all_medial_spheres->size());
  assert(sphere_id > -1 && sphere_id < this->is_msphere_add_new_sphere.size());
  const auto& msphere = this->all_medial_spheres->at(sphere_id);
  const auto& vec_point2fids = this->is_msphere_add_new_sphere.at(sphere_id);
  int num_sphere_added = 0;
  for (const v2int& v2fid_chosen : vec_point2fids) {
    // positive: fid, negative: ce_id
    bool is_point_on_ce = ss_params::is_point_on_ce(v2fid_chosen.second);
    int fid_or_ce_id =
        ss_params::decode_fid(v2fid_chosen.second, is_point_on_ce);
    if (is_debug)
      printf(
          "[add_T2] medge %d add new T2 sphere, "
          "v2fid_chosen fid: %d, is_point_on_ce: %d, fid_or_ce_id: %d\n",
          sphere_id, v2fid_chosen.second, is_point_on_ce, fid_or_ce_id);
    int new_sphere_id = this->all_medial_spheres->size();
    Vector3 new_normal = GEO::normalize(v2fid_chosen.first - msphere.center);
    MedialSphere new_msphere(
        new_sphere_id, v2fid_chosen.first, new_normal, fid_or_ce_id,
        is_point_on_ce ? SphereType::T_2_c : SphereType::T_2, 0
        /*itr_cnt*/);
    new_msphere.ss.set_p_fid(fid_or_ce_id, is_point_on_ce);
    if (is_point_on_ce)
      new_msphere.update_tan_cc_lines_from_ss_params(
          *this->sf_mesh, this->tet_mesh->feature_edges, true /*is_update_p*/,
          false /*is_update_q*/);
    if (!shrink_sphere(*this->sf_mesh, this->sf_mesh->aabb_wrapper,
                       this->sf_mesh->fe_sf_fs_pairs,
                       this->tet_mesh->feature_edges, new_msphere,
                       -1 /*itr_limit*/, false /*is_del_near_ce*/,
                       false /*is_del_near_se*/, false /*is_debug*/))
      continue;
    if (is_debug) printf("[add_T2] add new sphere %d\n", new_msphere.id);
    this->all_medial_spheres->push_back(new_msphere);
    num_sphere_added++;
  }
  if (is_debug)
    printf("[add_T2] sphere %d add num new spheres %d\n", sphere_id,
           num_sphere_added);
  return num_sphere_added;
}

// T2-T2 case
bool RPD3D_Wrapper::add_new_sphere_A_union_B_helper(const MedialMesh& mmesh,
                                                    const int meid,
                                                    bool is_debug) {
  assert(meid > -1 && meid < mmesh.edges.size());
  assert(!this->medge_clusters.empty() && meid < this->medge_clusters.size());

  // check if fids has intersections with clusters
  auto clusters_has_intersection =
      [&](const std::set<int> given_fids,
          const std::vector<SFCluster>& check_clusters) {
        std::vector<int> intersections;
        for (const auto& one_cluster : check_clusters) {
          set_intersection(given_fids, one_cluster.fids, intersections);
          if (!intersections.empty()) return true;
        }
        return false;
      };

  //////////////////////
  const MedialEdge& medge = mmesh.edges.at(meid);
  const auto& A = this->all_medial_spheres->at(medge.vertices_[0]);
  const auto& B = this->all_medial_spheres->at(medge.vertices_[1]);
  if (this->medge_clusters.at(meid).empty()) return false;
  if (is_debug)
    printf("[add_sphere_AunionB] medge %d (%d,%d) adding new INTF sphere\n",
           meid, medge.vertices_[0], medge.vertices_[1]);

  // find the given_clusters
  assert(A.id < this->sphere_clusters.size());
  assert(B.id < this->sphere_clusters.size());
  const auto medge_clusters = this->medge_clusters.at(meid);
  auto given_clusters = this->sphere_clusters.at(A.id);  // copy
  for (const auto& one_B_cluster : this->sphere_clusters.at(B.id)) {
    // filter given_clusters, save the one without any intersection
    if (clusters_has_intersection(one_B_cluster.fids, given_clusters)) continue;
    // filter medge_clusters, save the one without any intersection
    if (clusters_has_intersection(one_B_cluster.fids, medge_clusters)) continue;
    given_clusters.push_back(one_B_cluster);
  }
  // if (is_debug) {
  //   printf("[add_sphere_AunionB] new sphere has given_clusters:\n");
  //   for (const auto& c : given_clusters) c.print();
  // }

  bool is_debug_thread = false;
  // if (A.id == 529 && B.id == 967 || A.id == 967 && B.id == 529)
  //   is_debug_thread = true;

  // iterate sphere based on given_clusters
  Vector3 new_center = 0.5 * (A.center + B.center);
  double new_radius = 0.5 * (A.radius + B.radius);
  int sphere_id = this->all_medial_spheres->size();
  // do not use A.type, it may be on SE, but MedialSphere::se_edge_id is -1
  MedialSphere new_msphere(sphere_id, new_center, new_radius, SphereType::T_2);
  this->update_one_sphere_tangents_given_clusters(
      *this->tet_mesh, *this->sf_mesh, given_clusters, new_msphere,
      is_debug_thread /*is_debug*/);
  bool is_good =
      this->project_one_TN_sphere(new_msphere, is_debug_thread /*is_debug*/);
  // for debug, set it to be true
  if (is_debug_thread) is_good = true;
  if (!is_good) return is_good;
  if (is_debug)
    printf("[add_sphere_AunionB] add new sphere %d\n", new_msphere.id);
  this->all_medial_spheres->push_back(new_msphere);
  return is_good;
}

// T2-T3 case
bool RPD3D_Wrapper::add_new_sphere_A_union_partial_B_helper(
    const MedialMesh& mmesh, const int meid, bool is_debug) {
  assert(meid > -1 && meid < mmesh.edges.size());

  // check if fids has intersections with clusters
  auto clusters_has_intersection =
      [&](const std::set<int> given_fids,
          const std::vector<SFCluster>& check_clusters) {
        std::vector<int> intersections;
        for (const auto& one_cluster : check_clusters) {
          set_intersection(given_fids, one_cluster.fids, intersections);
          if (!intersections.empty()) return true;
        }
        return false;
      };

  //////////////////////
  const MedialEdge& medge = mmesh.edges.at(meid);
  const auto& A = this->all_medial_spheres->at(medge.vertices_[0]);
  const auto& B = this->all_medial_spheres->at(medge.vertices_[1]);
  if (this->medge_clusters.at(meid).empty()) return false;

  bool is_debug_thread = false;
  // if (A.id == 48 && B.id == 286 || A.id == 286 && B.id == 48)
  // if (A.id == 286 || B.id == 286) is_debug_thread = true;

  // find the given_clusters, load sphere with min clusters
  assert(A.id < this->sphere_clusters.size());
  assert(B.id < this->sphere_clusters.size());
  const auto& A_clusters = this->sphere_clusters.at(A.id);
  const auto& B_clusters = this->sphere_clusters.at(B.id);
  int min_sphere_id = A.id, max_sphere_id = B.id;
  if (get_num_clusters(B_clusters) < get_num_clusters(A_clusters)) {
    min_sphere_id = B.id;
    max_sphere_id = A.id;
  }
  if (is_debug)
    printf(
        "[add_sphere_A_union_partial_B] medge %d (%d,%d) adding new INTF "
        "sphere, min_sphere_id: %d, max_sphere_id: %d\n",
        meid, medge.vertices_[0], medge.vertices_[1], min_sphere_id,
        max_sphere_id);

  // new sphere
  Vector3 new_center = 0.5 * (A.center + B.center);
  double new_radius = std::max(A.radius, B.radius);
  int sphere_id = this->all_medial_spheres->size();
  // do not use A.type, it may be on SE, but MedialSphere::se_edge_id is -1
  MedialSphere new_msphere(sphere_id, new_center, new_radius, SphereType::T_2);
  auto given_clusters = this->sphere_clusters.at(min_sphere_id);  // copy
  for (const auto& one_max_cluster : this->sphere_clusters.at(max_sphere_id)) {
    // save and try a random one without any intersection
    if (clusters_has_intersection(one_max_cluster.fids, given_clusters))
      continue;
    given_clusters.push_back(one_max_cluster);
    if (is_debug) {
      printf("[add_sphere_A_union_partial_B] new sphere has given_clusters:\n");
      for (const auto& c : given_clusters) c.print();
    }

    // try to create new sphere
    new_msphere.tan_planes.clear();
    new_msphere.tan_cc_lines.clear();
    this->update_one_sphere_tangents_given_clusters(
        *this->tet_mesh, *this->sf_mesh, given_clusters, new_msphere,
        is_debug_thread /*is_debug*/);
    bool is_good =
        this->project_one_TN_sphere(new_msphere, is_debug_thread /*is_debug*/);
    if (!is_good) {
      given_clusters.pop_back();  // remove newly added one_max_cluster
      if (is_debug)
        printf(
            "[add_sphere_A_union_partial_B] medge %d (%d,%d) try one more time "
            "adding new INTF sphere\n",
            meid, medge.vertices_[0], medge.vertices_[1]);
      continue;  // try to add one more time
    }
    if (is_debug)
      printf("[add_sphere_A_union_partial_B] added new sphere %d\n",
             new_msphere.id);
    this->all_medial_spheres->push_back(new_msphere);
    return is_good;
  }
  return false;
}

void RPD3D_Wrapper::add_new_sphere_wrapper_after_eval(const MedialMesh& mmesh,
                                                      bool is_debug) {
  //////////////////////
  assert(this->is_msphere_add_new_sphere.size() ==
         this->all_medial_spheres->size());
  int num_add_sphere = 0;
  for (int msphere_id = 0; msphere_id < this->is_msphere_add_new_sphere.size();
       msphere_id++) {
    if (this->is_msphere_add_new_sphere[msphere_id].empty()) continue;
    num_add_sphere += add_T4_new_spheres_T2_helper(msphere_id, is_debug);
  }
  printf("[msphere_add_spheres] T4 added %d/%zu new spheres\n", num_add_sphere,
         this->all_medial_spheres->size());

  //////////////////////
  num_add_sphere = 0;
  std::set<int> processed_sphere_ids;
  for (int meid = 0; meid < mmesh.edges.size(); meid++) {
    const MedialEdge& medge = mmesh.edges.at(meid);
    const auto& A_id = medge.vertices_[0];
    const auto& B_id = medge.vertices_[1];
    // no need to add new sphere, or no need to handle here
    if (this->is_medge_add_new_sphere.at(meid) < 1) continue;
    // if (this->medge_clusters.at(meid).empty()) continue;
    // if added, then skip, do not add too many spheres
    if (processed_sphere_ids.find(A_id) != processed_sphere_ids.end() ||
        processed_sphere_ids.find(B_id) != processed_sphere_ids.end())
      continue;
    if (is_debug)
      printf(
          "[medge_add_spheres] medge %d (%d,%d), is_medge_add_new_sphere: %d\n",
          meid, medge.vertices_[0], medge.vertices_[1],
          this->is_medge_add_new_sphere.at(meid));

    bool is_good = false;
    if (this->is_medge_add_new_sphere.at(meid) == 1)
      is_good = add_new_sphere_A_union_B_helper(mmesh, meid, is_debug);
    else if (this->is_medge_add_new_sphere.at(meid) == 2)
      is_good = add_new_sphere_A_union_partial_B_helper(mmesh, meid, is_debug);
    if (is_good) {
      num_add_sphere++;
      // only marked processed when success
      processed_sphere_ids.insert(A_id);
      processed_sphere_ids.insert(B_id);
    }
  }
  printf("[medge_add_spheres] medges added %d/%zu new spheres\n",
         num_add_sphere, this->all_medial_spheres->size());
}

// simple version of sub-logic in
// RPD3D_Wrapper::update_sphere_clusters_variance_and_type()
void RPD3D_Wrapper::update_sphere_clusters_variance_and_type(
    const std::vector<PCELL_SAMPLE>& one_sphere_pcell_samples,
    std::vector<SFCluster>& clusters, MedialSphere& msphere) {
  // update cluster normal variance
  std::vector<Vector3> cluster_normals;
  for (auto& c : clusters) {
    // cannot call RPD3D_Wrapper::get_sphere_clusters_attribute()
    // since we have not save the clusters
    cluster_normals.clear();
    for (const int& pid : c.pids) {
      const Vector3& normal = one_sphere_pcell_samples.at(pid).normal;
      cluster_normals.push_back(normal);
    }
    // check the variance of normals
    update_normal_variance(*(this->params), cluster_normals, c);
  }

  msphere.type = SphereType::T_UNK;
  if (clusters.size() == 1 && is_inf_more_than_T1(clusters[0].is_normal_inf)) {
    msphere.type = SphereType::T_1_INF;
  } else if (clusters.size() == 2) {
    msphere.type = SphereType::T_2;
    // check if SphereType::T_2_INF
    for (int cid = 0; cid < clusters.size(); cid++) {
      const auto& c = clusters.at(cid);
      if (!is_inf_more_than_T2(c.is_normal_inf)) continue;
      msphere.type = SphereType::T_2_INF;  // on seams, not sheet
    }
  } else if (clusters.size() > 3) {
    msphere.type = SphereType::T_4_MORE;
  } else if (clusters.size() > 2) {
    msphere.type = SphereType::T_3_MORE;
  } else {
    msphere.type = SphereType::T_UNK;
  }
}

// private
void RPD3D_Wrapper::update_one_sheet_sphere_two_tangent(
    MedialSphere& msphere, std::vector<PCELL_SAMPLE>& one_sphere_pcell_samples,
    std::vector<SFCluster>& one_sphere_clusters) {
  msphere.tan_planes.clear();
  msphere.tan_cc_lines.clear();
  one_sphere_clusters.clear();
  if (msphere.is_deleted) return;
  bool is_debug_thread = false;
  // if (msphere.id == 9605) is_debug_thread = true;

  auto run_add_pcell_sample = [&](const Vector3& _proj, const Vector3& _normal,
                                  const int _fid) {
    PCELL_SAMPLE one_pcell_sample(msphere.id);
    one_pcell_sample.proj = _proj;
    one_pcell_sample.normal = _normal;
    one_pcell_sample.proj_fid = _fid;
    one_sphere_pcell_samples.push_back(one_pcell_sample);
  };

  auto run_add_sphere_clusters = [&](const int _fid, const int _pid) {
    SFCluster one_cluster(one_sphere_clusters.size(), _fid, _pid);
    one_cluster.pid_sphere_id = msphere.id;
    one_sphere_clusters.push_back(one_cluster);
  };

  double sq_dist = DBL_MAX;
  Vector3 nearest_p1;
  ///////////////////////////////////////////////////
  // handle SphereType::T_1_2, on sharp edge
  if (msphere.radius <= SCALAR_FEATURE_RADIUS) {
    int se_id = this->sf_mesh->aabb_wrapper.get_nearest_point_on_se(
        msphere.center, nearest_p1, sq_dist);
    assert(se_id > -1 && se_id < this->tet_mesh->feature_edges.size());
    const auto& fe = this->tet_mesh->feature_edges.at(se_id);
    run_add_pcell_sample(nearest_p1, fe.adj_normals[0], fe.adj_sf_fs_pair[0]);
    run_add_sphere_clusters(fe.adj_sf_fs_pair[0], 0 /*pid*/);
    run_add_pcell_sample(nearest_p1, fe.adj_normals[1], fe.adj_sf_fs_pair[1]);
    run_add_sphere_clusters(fe.adj_sf_fs_pair[1], 1 /*pid*/);
    TangentPlane tan_pl_p(*this->sf_mesh, fe.adj_normals[0], nearest_p1,
                          fe.adj_sf_fs_pair[0]);
    msphere.tan_planes.push_back(tan_pl_p);
    TangentPlane tan_pl_q(*this->sf_mesh, fe.adj_normals[1], nearest_p1,
                          fe.adj_sf_fs_pair[1]);
    msphere.tan_planes.push_back(tan_pl_q);
    msphere.type = SphereType::T_1_2;
    return;
  }

  ///////////////////////////////////////////////////
  // handle SphereType::T_2
  std::map<int, int> f2cluster;
  // tangent point p
  int p_fid = this->sf_mesh->aabb_wrapper.get_nearest_point_on_sf(
      msphere.center, nearest_p1, sq_dist);
  Vector3 p_normal = GEO::normalize(nearest_p1 - msphere.center);
  run_add_pcell_sample(nearest_p1, p_normal, p_fid);
  run_add_sphere_clusters(p_fid, 0 /*pid*/);
  f2cluster[p_fid] = one_sphere_clusters.back().id;

  // tangent point q
  int j = 0;
  Vector3 nearest_p2 = nearest_p1;
  Vector3 new_center = msphere.center;
  int q_fid = -1;
  while (q_fid == -1 || q_fid == p_fid ||
         GEO::distance(nearest_p2, nearest_p1) < SCALAR_1) {
    j++;
    Vector3 new_center =
        msphere.center - p_normal * msphere.radius * (SCALAR_ZERO_2 * j);
    q_fid = this->sf_mesh->aabb_wrapper.get_nearest_point_on_sf(
        new_center, nearest_p2, sq_dist);
    // printf("msphere %d, p_fid: %d q_fid: %d, j %d\n", msphere.id, p_fid,
    //        q_fid, j);
    if (j > 10) break;
  }
  assert(q_fid != -1);

  // only save q if the projection is close to the sphere
  double dist =
      std::abs(GEO::length(nearest_p2 - msphere.center) - msphere.radius);
  if (q_fid != p_fid) {
    p_normal = GEO::normalize(nearest_p2 - msphere.center);
    run_add_pcell_sample(nearest_p2, p_normal, q_fid);
    run_add_sphere_clusters(q_fid, 1 /*pid*/);
    f2cluster[q_fid] = one_sphere_clusters.back().id;
  }

  // update msphere's type
  if (one_sphere_clusters.size() == 1) {
    // save only 1 tangent plane
    TangentPlane tan_pl_p(*this->sf_mesh, p_normal, nearest_p1, p_fid);
    msphere.tan_planes.push_back(tan_pl_p);
  } else if (one_sphere_clusters.size() > 1) {
    // merge all clusters
    merge_sfclusters_using_KNN_V2(*this->tet_mesh, *this->sf_mesh, f2cluster,
                                  one_sphere_clusters, KNN_CLUSTER_VC /*K*/,
                                  is_debug_thread);
    // update tangents
    update_one_sphere_tangents_given_clusters(*this->tet_mesh, *this->sf_mesh,
                                              one_sphere_clusters, msphere,
                                              is_debug_thread);
  }
  update_sphere_clusters_variance_and_type(one_sphere_pcell_samples,
                                           one_sphere_clusters, msphere);
}

// private
void RPD3D_Wrapper::update_one_seam_or_junction_sphere_by_neighbors(
    const MedialMesh& mmesh, const std::set<int>& seam_spheres,
    const std::set<int>& junction_spheres, const MedialType& type,
    MedialSphere& msphere, std::vector<PCELL_SAMPLE>& one_sphere_pcell_samples,
    std::vector<SFCluster>& one_sphere_clusters) {
  if (msphere.is_deleted) return;
  if (msphere.edges_.empty()) return;
  assert(type == MedialType::SEAM || type == MedialType::JUNCTION);
  assert(this->sphere_clusters.size() == this->all_medial_spheres->size());
  bool is_debug_thread = false;
  // if (msphere.id == 366) is_debug_thread = true;
  // if (type == MedialType::JUNCTION) is_debug_thread = true;

  // get neighbor spheres
  std::set<int> neighbor_spheres;
  for (const auto& meid : msphere.edges_) {
    const auto& medge = mmesh.edges.at(meid);
    if (medge.is_deleted) continue;
    for (const auto& nvid : medge.vertices_) {
      if (nvid == msphere.id) continue;
      // find neighbors on sheet
      if (type == MedialType::SEAM &&
          (seam_spheres.find(nvid) != seam_spheres.end() ||
           junction_spheres.find(nvid) != junction_spheres.end()))
        continue;
      // // find neighbors on seam
      // if (type == MedialType::JUNCTION &&
      //     junction_spheres.find(nvid) != junction_spheres.end())
      //   continue;
      //
      // find neighbors on sheet, as well
      // to avoid two seams too close to each other
      if (type == MedialType::JUNCTION &&
          (seam_spheres.find(nvid) != seam_spheres.end() ||
           junction_spheres.find(nvid) != junction_spheres.end()))
        continue;
      neighbor_spheres.insert(nvid);
    }
  }
  if (is_debug_thread) {
    printf("[update_tangent] sphere %d, neighbor_spheres: ", msphere.id);
    print_set<int>(neighbor_spheres);
  }

  // update pcell_samples and clusters
  one_sphere_pcell_samples.clear();
  one_sphere_clusters.clear();
  std::set<int> new_pids;
  std::map<int, int> f2cluster, sample_old2new;
  for (const auto& nvid : neighbor_spheres) {
    const auto& nmsphere = this->all_medial_spheres->at(nvid);
    if (nmsphere.is_deleted) continue;
    // save neighbor samples
    sample_old2new.clear();
    const auto& nsamples = this->sphere_pcell_samples.at(nvid);
    for (int nsid = 0; nsid < nsamples.size(); nsid++) {
      const auto& nsample = nsamples.at(nsid);
      one_sphere_pcell_samples.push_back(nsample);
      one_sphere_pcell_samples.back().sphere_id = msphere.id;
      sample_old2new[nsid] = one_sphere_pcell_samples.size() - 1;
    }
    // save neighbor clusters, make a copy, not reference
    for (auto ncluster : this->sphere_clusters.at(nvid)) {  // copy
      // update all pids and pid_sphere_id in ncluster
      new_pids.clear();
      for (auto& pid : ncluster.pids) {
        if (sample_old2new.find(pid) == sample_old2new.end()) {
          printf(
              "[update_tangent] sphere %d, nsphere %d, type: %d nsamples: %d, "
              "ncluster.pids: %d, pid %d not found in sample_old2new\n",
              msphere.id, nvid, nmsphere.type, nsamples.size(),
              ncluster.pids.size(), pid);
          print_set<int>(ncluster.pids, "ncluster.pids");
          for (const auto& it : sample_old2new) {
            printf("pid %d -> %d\n", it.first, it.second);
          }
        }
        assert(sample_old2new.find(pid) != sample_old2new.end());
        new_pids.insert(sample_old2new.at(pid));
      }
      ncluster.pids = new_pids;
      ncluster.pid_sphere_id = msphere.id;

      // save ncluster
      bool is_store_new_cluster = true;
      for (const auto& fid : ncluster.fids) {
        if (f2cluster.find(fid) != f2cluster.end()) {
          // merge nclusters if multiple
          one_sphere_clusters.at(f2cluster.at(fid)).merge(ncluster);
          is_store_new_cluster = false;
          break;
        }
      }
      if (is_store_new_cluster) {
        one_sphere_clusters.push_back(ncluster);
        one_sphere_clusters.back().id = one_sphere_clusters.size() - 1;
        for (const auto& fid : one_sphere_clusters.back().fids) {
          f2cluster[fid] = one_sphere_clusters.back().id;
        }
      }
    }  // for clusters from neighbor spheres
  }  // for neighbor_spheres

  // merge all clusters
  merge_sfclusters_using_KNN_V2(*this->tet_mesh, *this->sf_mesh, f2cluster,
                                one_sphere_clusters, KNN_CLUSTER_VC /*K*/,
                                is_debug_thread);
  // update tangents
  update_one_sphere_tangents_given_clusters(*this->tet_mesh, *this->sf_mesh,
                                            one_sphere_clusters, msphere,
                                            is_debug_thread);
  // update cluster normal variance and type
  update_sphere_clusters_variance_and_type(one_sphere_pcell_samples,
                                           one_sphere_clusters, msphere);

  if (is_debug_thread) {
    printf("[update_tangent] sphere %d, type: %d, one_sphere_clusters: %zu\n",
           msphere.id, msphere.type, one_sphere_clusters.size());
    for (const auto& c : one_sphere_clusters) c.print();
  }
}

// public
// only update spheres as T2 spheres, only two tangent planes
void RPD3D_Wrapper::update_all_sheet_spheres_two_tangents() {
  this->sphere_clusters.clear();
  this->sphere_clusters.resize(this->all_medial_spheres->size());
  this->sphere_pcell_samples.clear();
  this->sphere_pcell_samples.resize(this->all_medial_spheres->size());
  // for (int i = 0; i < this->all_medial_spheres->size(); i++) {
  //   update_one_sheet_sphere_two_tangent(this->all_medial_spheres->at(i),
  //                                       this->sphere_pcell_samples[i],
  //                                       this->sphere_clusters[i]);
  // }
  // run in parallel
  GEO::parallel_for(0, this->all_medial_spheres->size(), [&](int i) {
    update_one_sheet_sphere_two_tangent(this->all_medial_spheres->at(i),
                                        this->sphere_pcell_samples[i],
                                        this->sphere_clusters[i]);
  });
}

// public
void RPD3D_Wrapper::update_spheres_tangents_after_tracing(
    const MedialMesh& mmesh) {
  const clock_t start_t = clock();
  this->sphere_clusters.clear();
  this->sphere_clusters.resize(this->all_medial_spheres->size());
  this->sphere_pcell_samples.clear();
  this->sphere_pcell_samples.resize(this->all_medial_spheres->size());

  // find all seam/junction spheres
  std::set<int> seam_spheres;
  std::set<int> junction_spheres;
  for (const auto& mstruct : mmesh.mstructure) {
    if (mstruct.type == MedialType::SEAM) {
      for (const auto& meid : mstruct.m_edge_ids) {
        const auto& medge = mmesh.edges.at(meid);
        for (const auto& vid : medge.vertices_) seam_spheres.insert(vid);
      }
    }
    if (mstruct.type == MedialType::JUNCTION) {
      for (const auto& mvid : mstruct.m_sphere_ids)
        junction_spheres.insert(mvid);
    }
  }

  // std::vector<int> visited_spheres(this->all_medial_spheres->size(), 0);
  // compute tangents of all medial spheres
  auto run_sheet_sphere = [&](const int sphere_id) {
    if (seam_spheres.find(sphere_id) != seam_spheres.end()) return;
    if (junction_spheres.find(sphere_id) != junction_spheres.end()) return;
    auto& msphere = this->all_medial_spheres->at(sphere_id);
    update_one_sheet_sphere_two_tangent(
        msphere, this->sphere_pcell_samples.at(sphere_id),
        this->sphere_clusters.at(sphere_id));
  };
  // for (const auto& mstruct : mmesh.mstructure)
  //   run_sheet_sphere(mstruct);
  GEO::parallel_for(0, this->all_medial_spheres->size(),
                    [&](int i) { run_sheet_sphere(i); });

  // compute tangents of spheres on seam
  auto run_seam_sphere = [&](const MedialStruct& mstruct) {
    if (mstruct.type != MedialType::SEAM) return;
    for (const auto& meid : mstruct.m_edge_ids) {
      const auto& medge = mmesh.edges.at(meid);
      for (int v = 0; v < medge.vertices_.size(); v++) {
        int vid = medge.vertices_[v];
        if (junction_spheres.find(vid) != junction_spheres.end()) continue;
        // if (visited_spheres[vid] == 1) continue;
        // visited_spheres[vid] = 1;
        // update the sphere
        auto& msphere = this->all_medial_spheres->at(vid);
        update_one_seam_or_junction_sphere_by_neighbors(
            mmesh, seam_spheres, junction_spheres, mstruct.type, msphere,
            this->sphere_pcell_samples.at(vid), this->sphere_clusters.at(vid));
      }
    }
  };
  // for (const auto& mstruct : mmesh.mstructure)
  //   run_seam_sphere(mstruct);
  GEO::parallel_for(0, mmesh.mstructure.size(),
                    [&](int i) { run_seam_sphere(mmesh.mstructure[i]); });

  // compute tangents of spheres on junction
  auto run_junction_sphere = [&](const MedialStruct& mstruct) {
    if (mstruct.type != MedialType::JUNCTION) return;
    for (const auto& vid : mstruct.m_sphere_ids) {
      auto& msphere = this->all_medial_spheres->at(vid);
      // if (visited_spheres[vid] == 1) continue;
      // visited_spheres[vid] = 1;
      update_one_seam_or_junction_sphere_by_neighbors(
          mmesh, seam_spheres, junction_spheres, mstruct.type, msphere,
          this->sphere_pcell_samples.at(vid), this->sphere_clusters.at(vid));
    }
  };
  // for (const auto& mstruct : mmesh.mstructure)
  //   run_junction_sphere(mstruct);
  GEO::parallel_for(0, mmesh.mstructure.size(),
                    [&](int i) { run_junction_sphere(mmesh.mstructure[i]); });
  std::cout << "[Timer] update_spheres_tangents_after_tracing took "
            << (float)(clock() - start_t) / CLOCKS_PER_SEC << " seconds"
            << std::endl;
}

#include "knn.h"
void remove_close_medial_spheres(std::vector<MedialSphere>& all_medial_spheres,
                                 double threshold) {
  double sqr_threshold = threshold * threshold;
  printf("[RemoveCloseSpheres] clean spheres with sqr_threshold: %lf\n",
         sqr_threshold);
  std::vector<Vector3> msphere_centers;
  for (const auto& msphere : all_medial_spheres)
    msphere_centers.push_back(msphere.center);
  std::vector<std::vector<uint32_t>> rets_index;
  std::vector<std::vector<double>> rets_dist_sqr;
  int K = 3;
  knn_search(msphere_centers, K, rets_index, rets_dist_sqr, false /*is_debug*/);
  bool is_debug = false;

  std::set<int> close_set;
  // for each non-deleted msphere, check if it is close to other neighbors,
  // if yes, then delete other neighbors
  for (int i = 0; i < all_medial_spheres.size(); i++) {
    auto& msphere = all_medial_spheres.at(i);
    if (msphere.is_deleted) continue;
    if (msphere.is_on_extf()) continue;

    // if (msphere.id == 533)
    //   is_debug = true;
    // else
    //   is_debug = false;

    if (is_debug) {
      printf("sphere %d: (%lf %lf %lf)\n", msphere.id, msphere.center[0],
             msphere.center[1], msphere.center[2]);
      for (int j = 0; j < K; j++) {
        int neighbor_id = rets_index[i][j];
        printf("neighbor %d: %d, dist: %lf\n", j, neighbor_id,
               rets_dist_sqr[i][j]);
      }
    }

    // first neighbor is itself
    // second neighbor is too close
    if (rets_dist_sqr[i][1] < sqr_threshold) {
      // check all neighbors
      for (int j = 1; j < K; j++) {
        if (rets_dist_sqr[i][j] < sqr_threshold) {
          int neighbor_id = rets_index[i][j];
          assert(neighbor_id > 0 && neighbor_id < all_medial_spheres.size());
          auto& neighbor_msphere = all_medial_spheres.at(neighbor_id);
          if (neighbor_msphere.is_deleted) continue;
          if (neighbor_msphere.is_on_extf()) {
            // delete msphere, not neighbor_msphere
            msphere.is_deleted = true;
            close_set.insert(msphere.id);
            if (is_debug) {
              printf("msphere %d delete itself, neighbor %d dist: %lf\n",
                     msphere.id, neighbor_id, rets_dist_sqr[i][j]);
            }
            continue;
          }

          if (is_debug) {
            printf("msphere %d delete neighbor %d with dist: %lf\n", msphere.id,
                   neighbor_id, rets_dist_sqr[i][j]);
          }
          neighbor_msphere.is_deleted = true;
          close_set.insert(neighbor_id);
        }
      }
    }
  }
  printf("[RemoveCloseSpheres] Removed closed spheres: %zu\n",
         close_set.size());
  print_set<int>(close_set, "close_set");
}

void remove_outside_medial_spheres(
    const SurfaceMesh& sf_mesh, std::vector<MedialSphere>& all_medial_spheres) {
  std::vector<bool> is_outside(all_medial_spheres.size(), false);
  auto run_thread = [&](int sphere_id) {
    auto& msphere = all_medial_spheres.at(sphere_id);
    if (msphere.is_deleted) return;
    if (msphere.is_on_extf()) return;
    double sq_dist;
    Vector3 closest_point;
    int fid = sf_mesh.aabb_wrapper.get_nearest_point_on_sf(
        msphere.center, closest_point, sq_dist);
    Vector3 dir = GEO::normalize(closest_point - msphere.center);
    Vector3 f_normal = get_mesh_facet_normal(sf_mesh, fid);
    if (is_vector_same_direction(f_normal, dir, 90.f)) return;
    msphere.is_deleted = true;
    is_outside[sphere_id] = true;
  };
  GEO::parallel_for(0, all_medial_spheres.size(),
                    [&](int sphere_id) { run_thread(sphere_id); });

  std::set<int> outside_set;
  for (int i = 0; i < all_medial_spheres.size(); i++) {
    if (is_outside[i]) outside_set.insert(i);
  }
  printf("[RemoveOutsideSpheres] Removed outside spheres: %zu\n",
         outside_set.size());
  print_set<int>(outside_set, "outside_set");
}

// remove deleted + SphereType::T_UNK medial spheres
// reassign MedialSphere::id
void clean_deleted_T1_medial_spheres(
    std::vector<MedialSphere>& all_medial_spheres) {
  std::vector<MedialSphere> all_medial_spheres_clean;
  for (auto& msphere : all_medial_spheres) {
    if (msphere.is_deleted || msphere.type == SphereType::T_UNK) continue;
    all_medial_spheres_clean.push_back(msphere);
    all_medial_spheres_clean.back().id = all_medial_spheres_clean.size() - 1;
    all_medial_spheres_clean.back().prev_id = msphere.id;
  }

  printf("clean_deleted_T1_medial_spheres: %zu->%zu\n",
         all_medial_spheres.size(), all_medial_spheres_clean.size());
  all_medial_spheres = all_medial_spheres_clean;
}

int get_normals_SVD_rank(const std::vector<Vector3>& normals,
                         Eigen::JacobiSVD<Eigen::Matrix4d>& svd,
                         bool is_debug) {
  if (is_debug) printf("[SVDRank] calling get_normals_SVD_rank ...\n");
  assert(!normals.empty());

  Eigen::MatrixXd N(normals.size(), 4);
  for (int n = 0; n < normals.size(); n++) {
    N.row(n) = Eigen::Vector4d(normals[n][0], normals[n][1], normals[n][2], 1);
  }
  Eigen::Matrix4d N_trans_N = N.transpose() * N;
  if (is_debug) printf("[SVDRank] N_trans_N pass the sanity check\n");

  // SVD, 4x4 matrix
  svd = Eigen::JacobiSVD<Eigen::Matrix4d>(
      N_trans_N, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // determine which singular values should be considered nonzero.
  svd.setThreshold(SCALAR_ZERO_2);
  int num_rank = svd.rank();

  if (is_debug) {
    printf("[SVDRank] SVD rank is %d\n", num_rank);
    std::cout << "Its singular values are:" << std::endl
              << svd.singularValues() << std::endl;
  }
  return num_rank;
}

// 1. INF_TYPE::NOT_INF: false,
// 2. INF_TYPE::INF_T1: normal_variance > SphereType_T2_VARIANCE.
//    Only for SphereType::T_1_INF
// 3. INF_TYPE::INF_T2: normal_variance > SphereType_INF_VARIANCE
//    Only for SphereType::T_2_INF
double update_normal_variance(const Parameter& params,
                              const std::vector<Vector3>& sample_normals,
                              SFCluster& c) {
  c.normal_variance = compute_normals_variance(sample_normals);
  if (c.normal_variance > params.SphereType_T1_INF_VARIANCE)
    c.is_normal_inf = INF_TYPE::INF_T1;
  if (c.normal_variance > params.SphereType_T2_INF_VARIANCE)
    c.is_normal_inf = INF_TYPE::INF_T2;
  return c.normal_variance;
}
