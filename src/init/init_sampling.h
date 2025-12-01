#include "common_geogram.h"
#include "input_types.h"

void get_surface_poisson_disk_samples(const std::string& mesh_path,
                                      const double min_dist,
                                      const SurfaceMesh& sf_mesh,
                                      std::vector<v2int>& samples_fids,
                                      bool is_debug = false);

void load_poisson_samples(const std::string& in_path, const bool is_load_fid,
                          std::vector<v2int>& samples_fids);

void save_poission_samples(const std::string& out_path, const bool is_save_fid,
                           const std::vector<v2int>& samples_fids);