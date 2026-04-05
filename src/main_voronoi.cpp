#include <cuda_runtime.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "fix_intf_wrapper.h"
#include "input_types.h"
#include "io.h"
// #include "io_utils.hpp"
#include "eval/mesh_eval.h"
#include "init/init_sampling.h"
#include "io_wrapper.h"
#include "main_gui.h"
#include "medial_mesh.h"
#include "opt/opt_rpd.h"
#include "params.h"
#include "sharp_feature_detection.h"
#include "shrinking.h"
#include "triangulation.h"
#include "voronoi.h"

using timer = std::chrono::system_clock;

volatile std::sig_atomic_t interrupted = 0;

void handle_sigint(int) {
  interrupted = 1;
  fprintf(stderr, "\n[INFO] Caught SIGINT (Ctrl+C), cleaning up...\n");
  cudaDeviceReset();
  std::exit(1);
}

void setup_signal_handlers() {
  std::signal(SIGINT, handle_sigint);
  std::signal(SIGTERM, handle_sigint);
}

int main(int argc, char** argv) {
  setup_signal_handlers();
  std::srand(RAN_SEED);  // set random seed
  std::string tet_mesh_file;
  std::string output_dir = "../out";
  int poisson_diag = 40;
  Parameter params;

  CLI::App app{"matstruct"};
  app.add_option("-i,--input", tet_mesh_file,
                 "Input tet mesh in .msh format. (string, required)")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-p,--poisson", poisson_diag,
                 "Poisson radius = 1.f / poisson_diag * bbox_diag_l, bigger "
                 "more samples. (double, optional, default=40.f)");
  app.add_flag(
      "-o,--organic", params.is_run_organic,
      "Toggle for running Organic shapes. We do not detect feature edges "
      "SE/CE for organic shape. (boolean, optional, default=false)");
  app.add_option("--se,--convex", params.thres_convex,
                 "Convex angle threshold in degrees, bigger is less sensitive. "
                 "(double, optional, default=55.)");
  app.add_option("--ce,--concave", params.thres_concave,
                 "Concave angle threshold in degrees, bigger is more "
                 "sensitive. (double, optional, default=60.)");
  app.add_option("-d,--output-dir", output_dir,
                 "Output directory (default: ../out)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  set_output_dir(output_dir);

  params.is_run_cad = !params.is_run_organic;
  if (params.is_run_organic) {
    params.thres_concave = 70.;
    params.is_sample_rpd = false;
  }
  printf("output_dir: %s\n", g_output_dir.c_str());
  printf("setting poisson_diag to %d\n", poisson_diag);
  printf("is_run_cad: %d, is_run_organic: %d\n", params.is_run_cad,
         params.is_run_organic);
  printf("thres_convex %f, thres_concave %f\n", params.thres_convex,
         params.thres_concave);

  int* initptr = nullptr;
  cudaError_t err = cudaMalloc(
      &initptr, sizeof(int));  // unused memory, needed for initialize the
                               // GPU before time measurements
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate (error code << " << cudaGetErrorString(err)
              << ")! [file: " << __FILE__ << ", line: " << __LINE__ << "]"
              << std::endl;
    return 1;
  }

  // Load tet from file
  TetMesh tet_mesh(tet_mesh_file);
  // please set normalize to true for model joint.tet/.xyz
  if (!load_tet(tet_mesh_file, tet_mesh.tet_vertices, tet_mesh.tet_indices,
                true /*normalize*/, params)) {
    std::cerr << tet_mesh_file << ": could not load file" << std::endl;
    return 1;
  }
  load_v2tets(tet_mesh.tet_vertices, tet_mesh.tet_indices, tet_mesh.v2tets);
  // Read surface file .geogram if exists
  // we store the mapping from tet_vertices to sf_mesh in vertex attributes
  //
  // this will make sure all geogram algoriths can be used
  GEO::initialize();
  GEO::CmdLine::import_arg_group("algo");
  std::string surface_path =
      get_other_file_path(tet_mesh_file, 0 /* _scaled_sf.geogram*/);
  std::string surface_path_obj =
      get_other_file_path(tet_mesh_file, 1 /* _scaled_sf.obj*/);
  std::cout << "surface path: " << surface_path << std::endl;
  SurfaceMesh sf_mesh;
  bool is_sf_file_exist = false;
  if (!is_file_exist(surface_path)) {
    std::cout << "surface path: " << surface_path
              << " not exist, load from tet_mesh_file: " << tet_mesh_file
              << std::endl;
    std::vector<std::array<float, 3>> sf_vertices;
    std::vector<std::array<int, 3>> sf_faces;
    get_surface_from_tet(tet_mesh.tet_vertices, tet_mesh.tet_indices,
                         sf_vertices, sf_faces, sf_mesh.sf2tet_vs_mapping);
    load_sf_mesh_from_internal(sf_vertices, sf_faces, sf_mesh.sf2tet_vs_mapping,
                               sf_mesh);
    // ATTENTION: this will REORDER GEO::Mesh!!!!
    pre_and_init_aabb(sf_mesh, sf_mesh.aabb_wrapper, true /*is_reorder*/);
  } else {
    is_sf_file_exist = true;
    if (!load_surface_mesh_geogram(surface_path, sf_mesh)) {
      std::cerr << surface_path << ": could not load surface file" << std::endl;
      return 1;
    }
    pre_and_init_aabb(sf_mesh, sf_mesh.aabb_wrapper, false /*is_reorder*/);
    printf("loaded surface from geogram file %s\n", surface_path.c_str());
  }

  std::cout << "loaded surface_mesh #v: " << sf_mesh.vertices.nb()
            << ", #f: " << sf_mesh.facets.nb() << std::endl;

  //////////////////////////////////////////
  // load the mapping from tet_vertices to sf_mesh
  // NOTE: must use after pre_and_init_aabb(), which may REORDER GEO::Mesh
  std::map<int, int> _;
  std::map<int, std::set<int>> __;
  load_sf_tet_mapping(sf_mesh, _, __, tet_mesh.tet_vs2sf_fids);
  load_tet_adj_info(tet_mesh.v2tets, tet_mesh.tet_indices,
                    tet_mesh.tet_vs2sf_fids, sf_mesh.facets.nb(),
                    tet_mesh.v_adjs, tet_mesh.e_adjs, tet_mesh.f_adjs,
                    tet_mesh.f_ids);
  sf_mesh.reload_sf2tet_vs_mapping();
  sf_mesh.cache_sf_fid_neighs();

  if (params.is_run_cad) {  // CAD
    // Detect sharp features
    // (store in tet mesh indices, matching tet_vertices)
    detect_mark_sharp_features(params, sf_mesh, tet_mesh, is_sf_file_exist);
    pre_and_init_feature_aabb(tet_mesh, sf_mesh.aabb_wrapper);
    sf_mesh.cache_sf_fid_to_fe_ids(tet_mesh.feature_edges);
    sf_mesh.cache_fe_sf_fs_map();
    sf_mesh.collect_sf_regions_no_cross(false /*is_debug*/);
  }
  sf_mesh.cache_sf_fid_neighs_no_cross();
  sf_mesh.cache_sf_fid_krings_no_cross_se_only(KNN_CLUSTER);

  //////////////////////////////////////////
  // save with SE/CE marked in .geogram format
  if (!is_sf_file_exist) {
    save_sf_mesh_geogram(surface_path, sf_mesh);
    // save surface mesh as obj file
    save_sf_mesh(surface_path_obj, sf_mesh);
  }
  // // save sf_mesh with extf marked in .ma format
  // save_sf_mesh_with_extf(get_other_file_path(tet_mesh_file, 4
  // /*_scaled_extf.ma*/), sf_mesh);

  //////////////////////////////////////////
  // Load surface samples using poisson disk sampling
  std::string surface_pts_path = get_other_file_path(
      tet_mesh_file, 5 /*_scaled_poisson_xx.xyz*/, poisson_diag);
  if (is_file_exist(surface_pts_path)) {
    load_poisson_samples(surface_pts_path, true /*is_load_fid*/,
                         sf_mesh.samples_fids);
  } else {
    double min_dist = (1. / poisson_diag) * params.bbox_diag_l;
    get_surface_poisson_disk_samples(surface_path_obj, min_dist, sf_mesh,
                                     sf_mesh.samples_fids, false /*is_debug*/);
    save_poission_samples(surface_pts_path, true /*is_save_fid*/,
                          sf_mesh.samples_fids);
  }

  //////////////////////////////////////////
  // Init medial spheres from surface samples
  std::vector<MedialSphere> all_medial_spheres;
  init_and_shrink_given_pins(sf_mesh, tet_mesh, all_medial_spheres,
                             sf_mesh.samples_fids, -1 /*itr_limit*/,
                             false /*is_debug*/);
  if (params.is_run_cad)
    insert_spheres_for_concave_lines_new(
        sf_mesh, tet_mesh.cc_corners, tet_mesh.feature_edges, tet_mesh.ce_lines,
        all_medial_spheres,
        params.cc_len_eps_rel * params.bbox_diag_l /*cc_len_eps*/,
        false /*is_debug*/);

  RegularTriangulationNN rt;
  MedialMesh mmesh;
  SphereTopK sphere_topk;

  MainGuiWindow main_gui;
  main_gui.name_no_ext =
      get_only_file_name(tet_mesh.tet_path_with_ext, false /*withExtension*/);
  main_gui.set_params(params);
  main_gui.set_tet_mesh(tet_mesh);
  main_gui.set_sf_mesh(sf_mesh);
  main_gui.set_all_medial_spheres(all_medial_spheres);
  main_gui.set_rt(rt);
  main_gui.set_medial_mesh(mmesh);
  main_gui.set_sphere_topk(sphere_topk);

  printf("calling OPT_RPD....\n");
  const bool is_rpd = true;
  OPT_RPD opt_rpd(is_rpd, false /*is_debug*/);
  opt_rpd.init(&tet_mesh, &sf_mesh, &params, &all_medial_spheres, &mmesh);
  opt_rpd.set_file_name_no_ext(main_gui.name_no_ext);
  main_gui.set_opt_rpd(&opt_rpd);
  main_gui.set_rpd3d(opt_rpd.get_rpd3d());

  if (params.is_run_cad) {
    main_gui.compute_rpd_partial_init();  // for calling auto_fix_extf()
    main_gui.auto_fix_extf();
    main_gui.auto_fix_intf_pre();
    main_gui.auto_fix_extf();
  }
  main_gui.auto_fix_topo_and_clean();
  main_gui.auto_opt_rpd_only_particle();
#ifdef MATSTRUCT_WITH_VISUALIZATION
  main_gui.show(true /*is_compute_rpd*/);
#endif

  // Release pointers before stack unwinds to prevent destructors from
  // calling delete on stack-allocated objects passed via set_*() / init().
  opt_rpd.get_rpd3d()->release_pointers();
  opt_rpd.release_pointers();
  mmesh.clear();
  main_gui.release_pointers();

  cudaDeviceReset();
  if (initptr != nullptr) cudaFree(initptr);
  return 0;
}
