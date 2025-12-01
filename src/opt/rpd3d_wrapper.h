#pragma once

#include <Eigen/Dense>
#include <map>

#include "dist2meshAABBTree.h"
#include "input_types.h"
#include "medial_mesh.h"
#include "rpd_api.h"

#define IN_USE 1
#define NOT_IN_USE 0

#define FEATURE_ADJUST_SPHERE_TYPE_BY_FILTERING IN_USE
#define FEATURE_ADJUST_SPHERE_TYPE_BY_CE NOT_IN_USE
#define FEATURE_ADJUST_SPHERE_TYPE_BY_CE_NOT_ALLOW IN_USE
#define FEATURE_ADD_NEW_SPHERE_MFACE IN_USE

constexpr int KNN_CLUSTER_VC = 5;
constexpr int KNN_CLUSTER = 2;
constexpr int KNN_CLUSTER_EXPAND = 4;

// 1. INF_TYPE::NOT_INF: false,
// 2. INF_TYPE::INF_T1: normal_variance > SphereType_T2_VARIANCE.
//    Only for SphereType::T_1_INF
// 3. INF_TYPE::INF_T2: normal_variance > SphereType_INF_VARIANCE
//    Only for SphereType::T_2_INF
enum INF_TYPE { NOT_INF = -1, INF_T1 = 1, INF_T2 = 2 };
inline bool is_inf_more_than_T1(const INF_TYPE& inf_type) {
  return inf_type == INF_TYPE::INF_T1 || inf_type == INF_TYPE::INF_T2;
}
inline bool is_inf_more_than_T2(const INF_TYPE& inf_type) {
  return inf_type == INF_TYPE::INF_T2;
}

//----------------------------------------------------------------
inline std::vector<float> convert2std(const Eigen::VectorXd& X) {
  std::vector<float> sites4d;
  for (int i = 0; i < X.size(); i++) {
    sites4d.push_back(X(i));
  }
  return sites4d;
}
inline Eigen::Vector3d convert2eigen(const cfloat3& p) {
  return Eigen::Vector3d(p.x, p.y, p.z);
}
inline Eigen::Vector3d convert2eigen(const adouble3& p) {
  return Eigen::Vector3d(p[0], p[1], p[2]);
}
inline Eigen::Vector3d convert2eigen(const Vector3& p) {
  return Eigen::Vector3d(p.x, p.y, p.z);
}
inline Vector3 convert2geo(const Eigen::VectorXd& p) {
  return Vector3(p(0), p(1), p(2));
}

//----------------------------------------------------------------
//------------------------ MatStruct -----------------------------
//----------------------------------------------------------------
class SFCluster {
 public:
  SFCluster() { is_active = false; };
  SFCluster(int _id, int _fid, int _pid) : id(_id) {
    is_active = true;
    merge_to_cluters_id = -1;
    fids.insert(_fid);
    pids.insert(_pid);
    pid_sphere_id = -1;
    min_pid2dist.first = -1;
    min_pid2dist.second = DBL_MAX;
    normal_variance = -1;
    is_normal_inf = INF_TYPE::NOT_INF;
    normals_svd_rank = -1;
  };
  SFCluster(int _id, const std::set<int>& _fids) : id(_id) {
    is_active = true;
    merge_to_cluters_id = -1;
    fids = _fids;
    pid_sphere_id = -1;
    min_pid2dist.first = -1;
    min_pid2dist.second = DBL_MAX;
    normal_variance = -1;
    is_normal_inf = INF_TYPE::NOT_INF;
    normals_svd_rank = -1;
  };
  SFCluster(int _id) : id(_id) {
    is_active = true;
    merge_to_cluters_id = -1;
    pid_sphere_id = -1;
    min_pid2dist.first = -1;
    min_pid2dist.second = DBL_MAX;
    normal_variance = -1;
    is_normal_inf = INF_TYPE::NOT_INF;
    normals_svd_rank = -1;
  };
  SFCluster(int _id, const SFCluster& given) : id(_id) {
    is_active = true;
    merge_to_cluters_id = -1;
    fids.insert(given.fids.begin(), given.fids.end());
    pids.insert(given.pids.begin(), given.pids.end());
    if (given.pid_sphere_id != -1) pid_sphere_id = given.pid_sphere_id;
    min_pid2dist.first = -1;
    min_pid2dist.second = DBL_MAX;
    ce_line_ids.insert(given.ce_line_ids.begin(), given.ce_line_ids.end());
    normal_variance = -1;
    is_normal_inf = INF_TYPE::NOT_INF;
    normals_svd_rank = -1;
  };
  ~SFCluster() {};
  inline void merge(SFCluster& other) {
    fids.insert(other.fids.begin(), other.fids.end());
    pids.insert(other.pids.begin(), other.pids.end());
    ce_line_ids.insert(other.ce_line_ids.begin(), other.ce_line_ids.end());
    other.is_active = false;
    other.merge_to_cluters_id = id;
    normal_variance = -1;
    is_normal_inf = INF_TYPE::NOT_INF;
    normals_svd_rank = -1;
    if (other.min_pid2dist.second < min_pid2dist.second)
      min_pid2dist = other.min_pid2dist;
    pid_sphere_id = other.pid_sphere_id;
  }
  inline void expand_fids(const SurfaceMesh& sf_mesh, const int k = -1) {
    // expand fids by their neighbors, do not update ce_line_ids
    fids_expand.clear();
    fids_expand.insert(fids.begin(), fids.end());
    std::set<int> fid_knn;
    for (const int fid : fids) {
      fid_knn.clear();
      if (k == -1)  // use cached KNN_CLUSTER
        sf_mesh.get_sf_fid_krings_no_cross_se_only(fid, fid_knn);
      else
        sf_mesh.collect_kring_neighbors_given_fid_se_only(k, fid, fid_knn);
      fids_expand.insert(fid_knn.begin(), fid_knn.end());
    }

    // update fids_expand, remove those SE pairs if any
    if (!sf_mesh.fe_sf_fs_map_se_only.empty()) {
      std::set<int> fids_expand_tmp;
      for (const int fid : fids_expand) {
        if (sf_mesh.fe_sf_fs_map_se_only.find(fid) !=
            sf_mesh.fe_sf_fs_map_se_only.end()) {
          // if fid_not_allow is in fids_expand,
          // then remove both fid and fid_not_allow
          int fid_not_allow = sf_mesh.fe_sf_fs_map_se_only.at(fid);
          if (fids_expand.find(fid_not_allow) != fids_expand.end()) continue;
        }
        fids_expand_tmp.insert(fid);
      }
      fids_expand = fids_expand_tmp;
    }
  }
  inline void print() const {
    printf(
        "---SFCluster %d, is_active: %d, merge_to_cluters_id: %d, "
        "normal_variance: %f, is_normal_inf: %d, normals_svd_rank: %d\n",
        id, is_active, merge_to_cluters_id, normal_variance, is_normal_inf,
        normals_svd_rank);
    print_set<int>(fids, "fids");
    print_set<int>(fids_expand, "fids_expand");
    // print_set<int>(pids, "pids");
    printf("pid_sphere_id: %d, pids size: %zu, ", pid_sphere_id, pids.size());
    printf("min_pid2dist: (%d,%f)\n", min_pid2dist.first,
           min_pid2dist.first == -1 ? -1.f : min_pid2dist.second);
    print_set<int>(ce_line_ids, "ce_line_ids");
  }

 public:
  int id;
  bool is_active;
  int merge_to_cluters_id;  // when is_active == false, merge to merge_to_id
  // matching indices in SurfaceMesh
  std::set<int> fids;
  // expanded fids using k-ring neighbors
  // sample ids, matching RPD3D_Wrapper::sphere_pcell_samples
  // and PCELL_SAMPLE::sphere_id
  std::set<int> fids_expand;
  // store which sphere_id for storing pids
  int pid_sphere_id;
  // matching indices in RPD3D_Wrapper::sphere_pcell_samples
  std::set<int> pids;
  // pid whose proj has min dist to sphere
  // pid matches index in RPD3D_Wrapper::sphere_pcell_samples
  std::pair<int, double> min_pid2dist;
  // concave line ids
  std::set<int> ce_line_ids;
  // variance of sample normals,
  // from RPD3D_Wrapper::sphere_pcell_samples and
  // PCELL_SAMPLE::normal
  double normal_variance;
  // check the value of SFCluster::normal_variance
  // INF_TYPE::NOT_INF: false,
  // INF_TYPE::INF_T1: normal_variance > SphereType_T2_VARIANCE,
  // INF_TYPE::INF_T2: normal_variance > SphereType_INF_VARIANCE
  INF_TYPE is_normal_inf;
  // the rank of the SVD for normals in a cluster
  int normals_svd_rank;
};

class PCELL_SAMPLE {
 public:
  PCELL_SAMPLE(const int& _sphere_id) : sphere_id(_sphere_id) {};
  ~PCELL_SAMPLE() {};
  inline void verify() {
    assert(sphere_id > -1);
    assert(proj_fid > -1);
  }

 public:
  // mapping to each attribute
  enum ATTR {
    SPHERE_ID = 0,
    POS = 2,
    PLANE = 3,
    WEIGHT = 4,
    PROJ = 5,
    PROJ_FID = 6,
    PROJ_FE_ID = 7,
    PROJ_DIST = 8
  };

  int sphere_id = -1;
  // if on the boundary of pcell
  std::set<int> adj_sphere_ids;
  Vector3 pos;
  Vector3 normal;
  // used in Integral::get_tetra_integral_samples()
  double weight;
  /////////////////////////////////
  // after projection on surface
  Vector3 proj;
  // projected sf face id
  int proj_fid = -1;
  // projected FeatureEdge::id, -1 if not on FE
  int proj_FE_id = -1;
  double proj_dist = DBL_MAX;
};

class RPD3D_Wrapper : public RPD3D_GPU {
 public:
  RPD3D_Wrapper() {};
  ~RPD3D_Wrapper();
  RPD3D_Wrapper(const TetMesh* _tet_mesh, const SurfaceMesh* _sf_mesh,
                const Parameter* _params,
                std::vector<MedialSphere>* _all_medial_spheres);
  void init(const TetMesh* _tet_mesh, const SurfaceMesh* _sf_mesh,
            const Parameter* _params,
            std::vector<MedialSphere>* _all_medial_spheres);
  void load_spheres(std::vector<float>& _spheres, bool is_rpd);
  void reset_radii(std::vector<float>& _spheres, double radius);
  const std::vector<float>& get_sites() const;
  static void remove_reset_inactive_clusters(std::vector<SFCluster>& clusters);
  static int get_first_active_cluster_id(int cluster_id);

 public:
  // fetching a vector of PCELL_SAMPLE attribute for a given sphere_id
  // from RPD3D_Wrapper::sphere_pcell_samples, return a copy.
  //
  // 1. index matching RPD3D_Wrapper::sphere_pcell_samples
  // 2. filtered by given_adj_sphere_id if given != -1
  template <typename T>
  inline std::vector<T> get_sphere_pcell_samples_attribute(
      const int sphere_id, const PCELL_SAMPLE::ATTR attr,
      const std::set<int> given_adj_sphere_ids = std::set<int>()) const {
    assert(sphere_id < this->sphere_pcell_samples.size());
    std::vector<T> samples_attr;
    for (const auto& sample : this->sphere_pcell_samples.at(sphere_id)) {
      if (!given_adj_sphere_ids.empty()) {
        std::vector<int> inter;
        set_intersection(sample.adj_sphere_ids, given_adj_sphere_ids, inter);
        // not save if no full intersection
        if (inter.size() != given_adj_sphere_ids.size()) continue;
      }
      if constexpr (std::is_same_v<T, int>) {
        if (attr == PCELL_SAMPLE::ATTR::SPHERE_ID)
          samples_attr.push_back(sample.sphere_id);
        else if (attr == PCELL_SAMPLE::ATTR::PROJ_FID)
          samples_attr.push_back(sample.proj_fid);
        else if (attr == PCELL_SAMPLE::ATTR::PROJ_FE_ID)
          samples_attr.push_back(sample.proj_FE_id);
        else
          assert(false && "Invalid attribute type for int");
      } else if constexpr (std::is_same_v<T, double>) {
        if (attr == PCELL_SAMPLE::ATTR::WEIGHT)
          samples_attr.push_back(sample.weight);
        else if (attr == PCELL_SAMPLE::ATTR::PROJ_DIST)
          samples_attr.push_back(sample.proj_dist);
        else
          assert(false && "Invalid attribute type for double");
      } else if constexpr (std::is_same_v<T, Vector3>) {
        if (attr == PCELL_SAMPLE::ATTR::POS)
          samples_attr.push_back(sample.pos);
        else if (attr == PCELL_SAMPLE::ATTR::PLANE)
          samples_attr.push_back(sample.normal);
        else if (attr == PCELL_SAMPLE::ATTR::PROJ)
          samples_attr.push_back(sample.proj);
        else
          assert(false && "Invalid attribute type for Vector3");
      } else {
        assert(false && "Unsupported type for PCELL_SAMPLE attribute");
      }
    }
    return samples_attr;
  }

  // fetching a vector of PCELL_SAMPLE attribute for a given sphere_id
  // from RPD3D_Wrapper::sphere_clusters, return a copy.
  //
  // 1. index NOT matching RPD3D_Wrapper::sphere_pcell_samples
  // 2. filtered by given_adj_sphere_id if given != -1
  // 3. RPD3D_Wrapper::sphere_clusters are filterd by
  //    FEATURE_ADJUST_SPHERE_TYPE_BY_FILTERING if given
  // 4. RPD3D_Wrapper::sphere_clusters are filtered by given_cluster_id if given
  template <typename T>
  inline std::vector<T> get_sphere_clusters_attribute(
      const int sphere_id, const PCELL_SAMPLE::ATTR attr,
      const std::set<int> given_adj_sphere_ids = std::set<int>(),
      const int given_cluster_id = -1) const {
    assert(sphere_id < this->sphere_pcell_samples.size());
    assert(sphere_id < this->sphere_clusters.size());

    std::vector<T> samples_attr;
    const auto& all_clusters = this->sphere_clusters.at(sphere_id);
    for (int cluster_id = 0; cluster_id < all_clusters.size(); cluster_id++) {
      // filter by given_cluster_id
      if (given_cluster_id != -1 && cluster_id != given_cluster_id) continue;
      const auto& cluster = all_clusters.at(cluster_id);
      for (const auto& sample_id : cluster.pids) {
        const auto& sample =
            this->sphere_pcell_samples.at(sphere_id).at(sample_id);
        if (!given_adj_sphere_ids.empty()) {
          std::vector<int> inter;
          set_intersection(sample.adj_sphere_ids, given_adj_sphere_ids, inter);
          // not save if no full intersection
          if (inter.size() != given_adj_sphere_ids.size()) continue;
        }
        if constexpr (std::is_same_v<T, int>) {
          if (attr == PCELL_SAMPLE::ATTR::SPHERE_ID)
            samples_attr.push_back(sample.sphere_id);
          else if (attr == PCELL_SAMPLE::ATTR::PROJ_FID)
            samples_attr.push_back(sample.proj_fid);
          else if (attr == PCELL_SAMPLE::ATTR::PROJ_FE_ID)
            samples_attr.push_back(sample.proj_FE_id);
          else
            assert(false && "Invalid attribute type for int");
        } else if constexpr (std::is_same_v<T, double>) {
          if (attr == PCELL_SAMPLE::ATTR::WEIGHT)
            samples_attr.push_back(sample.weight);
          else if (attr == PCELL_SAMPLE::ATTR::PROJ_DIST)
            samples_attr.push_back(sample.proj_dist);
          else
            assert(false && "Invalid attribute type for double");
        } else if constexpr (std::is_same_v<T, Vector3>) {
          if (attr == PCELL_SAMPLE::ATTR::POS)
            samples_attr.push_back(sample.pos);
          else if (attr == PCELL_SAMPLE::ATTR::PLANE)
            samples_attr.push_back(sample.normal);
          else if (attr == PCELL_SAMPLE::ATTR::PROJ)
            samples_attr.push_back(sample.proj);
          else
            assert(false && "Invalid attribute type for Vector3");
        } else {
          assert(false && "Unsupported type for PCELL_SAMPLE attribute");
        }
      }
    }
    return samples_attr;
  }

 public:
  void get_sites_KNN(std::vector<std::set<int>>& site_adj_sites,
                     const double search_radius = -1);
  void get_sites_adj_sites(std::vector<std::set<int>>& site_adj_sites);
  int get_msphere_num_clusters(const int sphere_id);
  int get_num_clusters(const std::vector<SFCluster>& clusters);
  void get_pcell_dual_faces();  // [no use]
  void get_one_pcell_neighbors(const int sphere_id,
                               std::set<int>& one_pcell_neighbors);
  double get_avg_volume(bool is_debug) const;
  // [no use]
  // updating MedialSphere::pcell_samples_sf_fids_in_group
  void update_all_pcell_samples_sf_fids(bool is_debug = false);
  // update:
  // 1. MedialSphere::sphere_pcell_samples
  void update_pcell_samples_cuda(const SurfaceMesh& sf_mesh,
                                 bool is_sample_rpd);
  // should call after loading msphere from files, like .ma or .sph
  // since we do not store SE and CE info in those files
  void update_extf_sphere_FL_from_pcells();
  // should always call during computing the RPD
  void cluster_sphere_type_using_samples_fids(const TetMesh& tet_mesh,
                                              const SurfaceMesh& sf_mesh,
                                              bool is_debug = false);
  void expand_pcell_samples_by_neighbors(const SurfaceMesh& sf_mesh,
                                         bool is_debug = false);
  bool project_one_TN_sphere(MedialSphere& msphere, bool is_debug);
  /////////////////////////////////
  void cluster_all(const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
                   MedialMesh& mmesh, bool is_debug);
  /////////////////////////////////
  // helper functions for RPD3D_Wrapper::cluster_all()
  //
  // need to call ahead: RPD3D_Wrapper::cluster_sphere_type_using_samples_fids()
  void cluster_medges_same_sheet(const TetMesh& tet_mesh,
                                 const SurfaceMesh& sf_mesh, MedialMesh& mmesh,
                                 bool is_debug);
  void cluster_mfaces_same_sheet(const TetMesh& tet_mesh,
                                 const SurfaceMesh& sf_mesh, MedialMesh& mmesh,
                                 bool is_debug);
  void cluster_mfaces_adj_same_sheet(const TetMesh& tet_mesh,
                                     const SurfaceMesh& sf_mesh,
                                     MedialMesh& mmesh, bool is_debug);
  void cluster_medges_type(const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
                           MedialMesh& mmesh, bool is_debug);

  /////////////////////////////////
  void eval_mmesh_medge_length(MedialMesh& mmesh, bool is_debug);
  void add_new_sphere_after_eval_medge_length(const MedialMesh& mmesh,
                                              bool is_debug);

  /////////////////////////////////
  void eval_mmesh_cand_medge_not_same_sheet(MedialMesh& mmesh, bool is_debug);
  void add_new_sphere_wrapper_after_eval(const MedialMesh& mmesh,
                                         bool is_debug);

  /////////////////////////////////
  void update_all_sheet_spheres_two_tangents();
  void update_spheres_tangents_after_tracing(const MedialMesh& mmesh);

 public:
  void get_one_pcell_in_tets(ConvexCellHost& cc_trans,
                             std::vector<cfloat3>& voro_points,
                             std::vector<std::array<int, 4>>& voro_tets,
                             std::vector<int>& voro_tets_sites,
                             std::vector<int>& voro_tets_cell_ids);
  void calculate_given_spheres(
      const std::vector<float>& _spheres /*4D (x,y,z,r)*/,
      bool is_debug = false);

 private:
  void collect_nearby_sphere_samples_helper(bool is_sample_rpd, bool is_debug);
  void collect_pcell_sphere_samples_helper(bool is_debug);
  void sort_spheres_ids_from_clusters(std::vector<int>& sphere_ids);
  void get_pcell_in_bfaces();
  void get_pcell_bface_areas();
  void clear_spheres_topos();
  void reset_deleted_spheres();
  void load_spheres_to_sites(
      const std::vector<float>& _spheres /*4D (x,y,z,r)*/,
      const std::vector<int>& is_sphere_valid);
  void update_one_sphere_tangents_given_clusters(
      const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
      const std::vector<SFCluster>& given_clusters, MedialSphere& msphere,
      bool is_debug) const;

  void cluster_one_sphere_type_using_samples_fids(int sphere_id, int K = -1);
  // update TangentPlane/TangentConcaveLine
  // helper function for RPD3D_Wrapper::cluster_sphere_type_using_samples_fids()
  void update_all_spheres_tangents_use_clusters(bool is_debug) const;
  /////////////////////////////////
  // helper functions for RPD3D_Wrapper::cluster_all()
  void mark_one_medge_same_sheet_by_adj_bface(
      const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
      const std::vector<std::vector<SFCluster>>& sphere_clusters,
      std::vector<std::vector<SFCluster>>& medge_clusters, MedialMesh& mmesh,
      int eid, bool is_debug);
  void mark_one_mface_same_sheet_by_adj_bface(
      const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
      const std::vector<std::vector<SFCluster>>& sphere_clusters,
      std::vector<std::vector<SFCluster>>& mface_clusters, MedialMesh& mmesh,
      int fid, bool is_debug);
  void mark_one_mface_same_sheet_by_adj_bface_KNN(
      const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
      const std::vector<std::vector<SFCluster>>& sphere_clusters,
      std::vector<std::vector<SFCluster>>& mface_clusters, MedialMesh& mmesh,
      int fid, bool is_debug);
  void mark_one_adj_mfaces_same_sheet_by_adj_bface(
      const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
      const std::vector<std::vector<SFCluster>>& mface_clusters,
      std::vector<std::vector<SFCluster>>& mface_adj_clusters,
      MedialMesh& mmesh, int fid, bool is_debug);
  void mark_one_medge_type(
      const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
      const std::vector<std::vector<SFCluster>>& mface_adj_clusters,
      MedialMesh& mmesh, int eid, bool is_debug);
  /////////////////////////////////
  // helper functions for RPD3D_Wrapper::add_new_sphere_wrapper_after_eval
  int add_T4_new_spheres_T2_helper(const int sphere_id, bool is_debug);
  bool add_new_sphere_A_union_B_helper(const MedialMesh& mmesh, const int meid,
                                       bool is_debug);
  bool add_new_sphere_A_union_partial_B_helper(const MedialMesh& mmesh,
                                               const int meid, bool is_debug);
  /////////////////////////////////
  void update_one_sheet_sphere_two_tangent(
      MedialSphere& msphere,
      std::vector<PCELL_SAMPLE>& one_sphere_pcell_samples,
      std::vector<SFCluster>& one_sphere_clusters);
  void update_one_seam_or_junction_sphere_by_neighbors(
      const MedialMesh& mmesh, const std::set<int>& seam_spheres,
      const std::set<int>& junction_spheres, const MedialType& type,
      MedialSphere& msphere,
      std::vector<PCELL_SAMPLE>& one_sphere_pcell_samples,
      std::vector<SFCluster>& one_sphere_clusters);
  /////////////////////////////////
  // update Parameter::mmesh_avg_len
  void update_sphere_clusters_variance_and_type(
      const std::vector<PCELL_SAMPLE>& one_sphere_pcell_samples,
      std::vector<SFCluster>& clusters, MedialSphere& msphere);

 public:
  // --------- per tet
  // convert to tets
  // updated by calling get_pcell_in_bfaces();
  std::vector<cfloat3> pcell_bface_points;
  // save <v1,v2,v3> for each pcell boundary face
  std::vector<std::array<int, 3>> pcell_bfaces;
  std::vector<int> pcell_bface_sphere_ids;
  std::vector<int> pcell_bface_adj_sphere_ids;
  // --------- per sphere, merged by sphere_id
  // see function RPD3D_Wrapper::get_sphere_pcell_samples_attribute()
  // to fetch specific attribute for a given sphere_id
  std::vector<std::vector<PCELL_SAMPLE>> sphere_pcell_samples;
  // -------- [no use]
  // sphere_id: adj_sphere_id -> adjacent pcell face area
  std::vector<std::map<int, double>> sphere_pcell_adj_bface_area;
  std::vector<double> sphere_pcell_bface_area;
  //---------------------
  // updated by cluster_sphere_type_using_samples_fids()
  std::vector<std::vector<SFCluster>> sphere_clusters;
  // updated by RPD3D_Wrapper::cluster_medges_same_sheet()
  // to update MedialEdge::is_on_same_sheet
  std::vector<std::vector<SFCluster>> medge_clusters;
  // updated by RPD3D_Wrapper::cluster_mfaces_same_sheet()
  // to update MedialFace::is_on_same_sheet
  std::vector<std::vector<SFCluster>> mface_clusters;
  // (mfid_min, mfid_max) -> join clusters
  // RPD3D_Wrapper::mface_adj_clusters stored as a upper triangular matrix
  // with size n, given two sorted MedialFace:id indices (mfid_min, mfid_max),
  // we can get the index for the matrix by calling get_upper_tri_matrix_idx()
  std::vector<std::vector<SFCluster>> mface_adj_clusters;
  //---------------------
  // updated in RPD3D_Wrapper::cluster_sphere_type_using_samples_fids()
  // when feature FEATURE_ADJUST_SPHERE_TYPE_BY_FILTERING is in use
  // we want to adjust sphere type by removing clusters that contains too
  // few samples, so only dominant clusters samples are used.
  //
  //  0: should update by samples in RPD3D_Wrapper::sphere_pcell_samples
  //  1: should update by samples indicate in RPD3D_Wrapper::sphere_clusters
  std::vector<int> is_sphere_adj_type;

  // ---------------------
  // store if a candiate INTF medge block the volume
  // 0: not add
  // 1: not same sheet, T2-T2 case, add new INTF
  // 2: not same sheet, T2-T3 case, add new INTF
  std::vector<int> is_medge_add_new_sphere;
  // for T4 sphere where tangents are far from tangents
  std::vector<std::vector<v2int>> is_msphere_add_new_sphere;

  // ---------------------
  // convert to dual face
  std::set<aint3> pcell_dual_faces;  // [no use]

 private:
  std::vector<afloat3> aabb_tri_vertices;
  std::vector<AABBTriangle> aabb_triangles;
  std::vector<AABBTreeNode> aabb_tree;
  void load_bvh_triangles_from_mesh(const GEO::Mesh& mesh);

 private:
  double mmesh_avg_len = -1;
  void compute_and_update_params_mmesh_avg_length(MedialMesh& mmesh);
};

/////////////////////////////////
void remove_close_medial_spheres(std::vector<MedialSphere>& all_medial_spheres,
                                 double threshold);
void remove_outside_medial_spheres(
    const SurfaceMesh& sf_mesh, std::vector<MedialSphere>& all_medial_spheres);
void clean_deleted_T1_medial_spheres(
    std::vector<MedialSphere>& all_medial_spheres);

/////////////////////////////////
int get_normals_SVD_rank(const std::vector<Vector3>& normals,
                         Eigen::JacobiSVD<Eigen::Matrix4d>& svd, bool is_debug);

/////////////////////////////////
double update_normal_variance(const Parameter& params,
                              const std::vector<Vector3>& sample_normals,
                              SFCluster& c);