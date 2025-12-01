#pragma once

#include <Eigen/Dense>

#include "LBFGS.h"
#include "medial_mesh.h"
#include "particle.h"
#include "rpd3d_wrapper.h"
#include "rpd_api.h"

class OPT_RPD {
 public:
  class OPT_PARAM {
   public:
    double alpha_print;
    double lambda_particle;        // for particle + global/lbfgs
    double gamma;                  // odt + local
    double particle_kernel_sigma;  // sigma in [Zhong2013]
  };

  class OPT_LOSS {
   public:
    double lossParticle = 0.0;
    double lossTotal = 0.0;
  };

  enum RUN_TYPE {
    PROJ = 3,
    PARTICLE_MMESH = 7,
    PARTICLE_KNN = 8,
  };

  class PROJ_TYPE {
   public:
    PROJ_TYPE() {};
    bool is_proj_SE = false;
    bool is_proj_UNK = false;
    bool is_proj_T2 = false;
    bool is_proj_TN = false;
    bool is_proj_cc = false;
    bool is_proj_TN_cc = false;
  };

  enum ODT_INTEG_TYPE {
    FREE = 1,       // not consider boundaries
    EXTF_ONLY = 2,  // consider only extf neighbors only, differently
    EXTF_INTF = 3,  // consider only extf or only intf neighbors, differently
    IGN_EXTF_INTF = 4,  // ignore both extf and intf neighbors
    NO_USE = 5
  };

 public:
  OPT_RPD(bool _is_rpd, bool _debug) : is_rpd(_is_rpd), is_debug(_debug) {};
  ~OPT_RPD();
  void init(const TetMesh* _tet_mesh, const SurfaceMesh* _sf_mesh,
            const Parameter* _params,
            std::vector<MedialSphere>* _all_medial_spheres, MedialMesh* _mmesh);
  void set_file_name_no_ext(std::string _name_no_ext);

  // for debug
  std::vector<Vector3> _sphere_centers_old;  // 3D (x,y,z)
  void load_sphere_centers_old_from_all();

 public:
  void calculate_proj_SQEM(int& itr_cnt);
  void calculate_proj_SQEM_KNN(int& itr_cnt);
  void calculate_proj(int& itr_cnt, const PROJ_TYPE& proj_type);
  void calculate_particle_lbfgs_globally(int& itr_cnt, const RUN_TYPE& run_type,
                                         const int num_particle_itr);

  RPD3D_Wrapper* get_rpd3d() { return &_rpd3d; }
  void reset_is_rpd(bool _is_rpd) { this->is_rpd = _is_rpd; }
  bool get_is_rpd() const { return this->is_rpd; }
  Eigen::VectorXd get_grad_itr() const { return this->_grad_itr; }

 private:
  std::string get_save_folder_path(const std::string& name_no_ext) const;
  Vector3 get_sphere_center(const int sphere_id);
  Vector3 get_sphere_center(const Eigen::VectorXd& X, const int sphere_id);
  void save_spheres_itr_file(const Eigen::VectorXd& spheres_itr,
                             const std::string folder_path,
                             const std::string filename, RUN_TYPE run_type,
                             int num_rpd_itr);
  void load_spheres_from_all();
  void save_spheres_to_all();
  void save_spheres_to_all(const Eigen::VectorXd& X);

  bool is_sphere_TN_and_contains_cc_lines_wrapper(const int sphere_id);
  bool is_sphere_contains_cc_lines_wrapper(const int sphere_id);
  bool is_sphere_UNK_wrapper(const int sphere_id);
  bool is_sphere_TN_wrapper(const int sphere_id);
  bool is_sphere_SE_wrapper(const int sphere_id);

  void get_sphere_centers_old();
  void set_lbfgs_params();
  void reset_loss();
  // call everytime we run RPD again
  void update_particle_projection_gradient_cache(bool is_debug = false);
  void calculate_spheres_samples(const Eigen::VectorXd& X);
  void calculate_rpd_rvd_samples(const Eigen::VectorXd& X);
  void calculate_rpd_rvd_and_cluster_samples(const Eigen::VectorXd& X);
  void update_mspheres_radii(const SurfaceMesh& sf_mesh);
  void calculate_mmesh();
  void calculate_mmesh_and_mark_same_sheet();
  void calculate_only_medges_and_mark_same_sheet();
  bool shrink_one_sphere_wrapper(MedialSphere& new_msphere, bool is_debug);
  bool project_one_sphere(const PROJ_TYPE& proj_type, const int sphere_id);
  void run_spheres_projection(const PROJ_TYPE& proj_type = PROJ_TYPE());
  void project_sheet_spheres_SQEM(Eigen::VectorXd& X);
  void project_particle_spheres_SQEM(Eigen::VectorXd& X);

 private:
  // ----------------------------
  // for mmesh mfaces on the same sheet
  // does NOT consider is_ignore_intf/is_ignore_extf<
  std::set<aint3> mmesh_mface_same_sheet;
  // for 1ring area sum
  // taking is_ignore_intf/is_ignore_extf into account
  std::vector<double> spheres_1ring_sum;
  // for 1ring, sum over (area * circumcenter)
  // taking is_ignore_intf/is_ignore_extf into account
  std::vector<Vector3> spheres_1ring_ccenter;
  // for 1ring, sum over (area * (sphere_center - circumcenter))
  // equation（4.5) [Chen2011]
  // taking is_ignore_intf/is_ignore_extf into account
  std::vector<Vector3> spheres_1ring_sphere_ccenter;
  // for 1ring, sum over (area * (sphere_center - barycenter)) -- CPT
  // equation（4.19) [Chen2011]
  // taking is_ignore_intf/is_ignore_extf into account
  std::vector<Vector3> spheres_1ring_sphere_bcenter;
  // sphere_id -> 1ring mfaces <mf1, mf2, mf3> sorted
  // equation（4.23) [Chen2011]
  // taking is_ignore_intf/is_ignore_extf into account
  // may contains mface for medge as {-1, mf1, mf2}
  std::map<int, std::set<aint3>> spheres_to_1ring_mfaces;

  // ----------Particle Smoothing---------
  void update_particle_kernel_helper();
  void integrate_particle_KNN(Eigen::VectorXd& X, Eigen::VectorXd& g);
  void integrate_particle_knn_with_mmesh(Eigen::VectorXd& X,
                                         Eigen::VectorXd& g);
  void run_particle_lbfgs_globally_KNN(int& num_itr_global, BGAL::_LBFGS& lbfgs,
                                       const std::string folder_path,
                                       const std::string name_no_ext);
  void run_particle_lbfgs_globally_mmesh(int& num_itr_global,
                                         BGAL::_LBFGS& lbfgs,
                                         const std::string folder_path,
                                         const std::string name_no_ext);
  std::map<int, std::set<int>> spheres_1ring_neighbors;

 private:
  std::string name_no_ext;
  bool is_rpd;
  bool is_debug;
  int dim = 4;  // 4D (x,y,z,r)
  const TetMesh* _tet_mesh;
  const SurfaceMesh* _sf_mesh;
  std::vector<MedialSphere>* _all_medial_spheres;
  RPD3D_Wrapper _rpd3d;
  MedialMesh* _mmesh;
  Eigen::VectorXd _spheres_itr;  // 4D (x,y,z,r)
  Eigen::VectorXd _grad_itr;     // 4D (x,y,z,r)
  OPT_PARAM params;
  OPT_LOSS loss;
  std::map<aint2, bool> _is_two_sphere_same_type_cache;
  // storing the gradient projection
  // updated by update_particle_projection_gradient_cache()
  std::vector<ParticleProjection> _particle_projection_cache;
};
