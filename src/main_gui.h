#ifndef H_MAIN_GUI_H
#define H_MAIN_GUI_H

#include <geogram/mesh/mesh.h>

#include <chrono>
#include <cmath>

#include "eval/mesh_eval.h"
#include "input_types.h"
#include "medial_mesh.h"
#include "medial_sphere.h"
#include "opt/opt_rpd.h"
#include "opt/rpd3d_wrapper.h"
#include "rpd_api.h"
#include "triangulation.h"
#include "voronoi_defs.h"

class MainGuiWindow {
 public:
  // saver
  std::string name_no_ext = "";
  std::string sub_folder_name = "sph_itr";

 private:
  static MainGuiWindow* instance_;
  static constexpr double point_radius_rel = 0.002;
  static constexpr double edge_radius_rel = 0.0008;
  using timer = std::chrono::system_clock;

 private:
  // non-pointers
  bool site_is_transposed;
  std::vector<float> site;
  std::vector<float> site_weights;
  std::vector<uint> site_flags;
  std::vector<int> site_knn;
  std::vector<float> site_cell_vol;
  std::vector<Vector3> pc_vertices;  // (non-restricted) powercell vertices
  std::set<aint2> checked_mmesh_edges;

  // pointers
  Parameter* params = nullptr;  // not const, update site_k
  // eg. const int * == int const * (pointer to const int)
  TetMesh* tet_mesh = nullptr;  // not const, init_corner_spheres() will update
  const SurfaceMesh* sf_mesh = nullptr;
  const std::set<int>* spheres_to_fix = nullptr;
  std::vector<MedialSphere>* all_medial_spheres = nullptr;  // not const
  //   std::vector<ConvexCellHost>* cells_to_show = nullptr;     // not const
  MedialMesh* mmesh = nullptr;           // not const
  RegularTriangulationNN* rt = nullptr;  // not const
  RPD3D_Wrapper* rpd3d = nullptr;        // not const
  OPT_RPD* opt_rpd = nullptr;            // not const
  SphereTopK* sphere_topk = nullptr;     // not const

  // For GEO FIX
  std::vector<double> fix_geo_samples;
  std::vector<float> fix_geo_samples_dist2mat;
  std::vector<aint3> fix_geo_samples_clostprim;

 public:
  void set_params(Parameter& _params);
  void set_tet_mesh(TetMesh& _tet_mesh);
  void set_tet_adj_info(const std::map<int, std::set<int>>& _v2tets,
                        const std::map<int, std::set<int>>& _tet_vs2sf_fids,
                        const std::map<aint4, int>& _tet_vs_lfs2tvs_map);
  void set_sf_mesh(const SurfaceMesh& _sf_mesh);
  void set_all_medial_spheres(std::vector<MedialSphere>& _all_medial_spheres);
  void set_rt(RegularTriangulationNN& _rt);
  void set_medial_mesh(MedialMesh& _mmesh);
  // call after set_tet_mesh(), set_sf_mesh(), reset_params()
  void set_rpd3d(RPD3D_Wrapper* _rpd3d);
  void set_opt_rpd(OPT_RPD* _opt_rpd);
  void set_sphere_topk(SphereTopK& _sphere_topk);

 public:
  MainGuiWindow();
  ~MainGuiWindow();  // implemented in main_gui.cpp

  // show polyscope window
  void compute_rpd_partial_init();
  void compute_rpd();
  void update_rpd_after_partial();
  void show(bool is_compute_rpd = true);
  static void callbacks();

  // main iteration functions
  int num_itr_global = 0;
  int num_sphere_added = 0;

  // fix
  int num_topo_itr = 0;
  int run_topo_fix_itr(const SurfaceMesh& sf_mesh, const TetMesh& tet_mesh,
                       std::vector<MedialSphere>& all_medial_spheres,
                       std::vector<ConvexCellHost>& cells_to_show,
                       bool is_debug = false);
  int num_mmesh_itr = 0;
  void run_mmesh_generator(const SurfaceMesh& sf_mesh, MedialMesh& mmesh,
                           std::vector<MedialSphere>& all_medial_spheres,
                           bool is_compute_common_diff_sfids = false,
                           bool is_trace_mstruc = false, bool is_debug = false);
  int num_geo_itr = 0;
  int run_fix_geo_itr(const SurfaceMesh& sf_mesh, const TetMesh& tet_mesh,
                      const Parameter& params, MedialMesh& mmesh,
                      std::vector<MedialSphere>& all_medial_spheres,
                      bool is_debug = false);
  int num_extf_itr = 0;
  int run_extf_fix_itr(const Parameter& param, TetMesh& tet_mesh,
                       const SurfaceMesh& sf_mesh,
                       const std::vector<ConvexCellHost>& cells_to_show,
                       std::vector<MedialSphere>& all_medial_spheres,
                       MedialMesh& mmesh, bool is_debug = false);
  int num_intf_itr = 0;
  int run_intf_fix_itr(const SurfaceMesh& sf_mesh, const TetMesh& tet_mesh,
                       const MedialMesh& mmesh,
                       std::vector<MedialSphere>& all_medial_spheres,
                       bool is_debug = false);

  // Convex cells
  void show_result_convex_cells(std::vector<ConvexCellHost>& cells_to_show,
                                bool is_slice_plane = false);

  // show pcell site tet samples
  int given_pcell_sample_id = -1;

  // Tetmesh
  void show_tet_mesh(const TetMesh& tet_mesh, bool is_shown_meta = false);

  // Medial Mesh
  int given_mat_face_id = -1;
  void show_medial_mesh(const MedialMesh& mmesh,
                        std::string mmname = "MedialMesh");
  double given_thinning_thres = 0.3;

  // PARTICLE/MAT_PARTICLE/MAT_GLOBAL
  int num_particle_itr = 20;
  int num_all_itr = 0;

  // automation
  void auto_opt_rpd_only_particle();
  void auto_fix_topo_and_clean();
  void auto_fix_extf();
  void auto_fix_intf();
  void auto_fix_intf_pre();

  // MatStruct
  void proj_all_sphere_clean_extf();
  void proj_sphere_SQEM_clean_extf();
  double shrink_factor = 5.f;

 private:
  void auto_eval_add_medge_len(const int itr_print);
  void auto_eval_add_medge(const int itr_print);
};

#endif  // __H_MAIN_GUI_H__
