#include "io.h"
#include "io_cuda.h"
#include "io_utils.hpp"

// // for exporting more info
// void export_ma_and_struct_all(const std::string& maname, MedialMesh& mat);

// shrink sheets of medial mesh
void compute_shrink_mstructures(const MedialMesh& mmesh,
                                std::vector<Vector3>& shrinkVertices,
                                std::vector<aint3>& shrinkFaces,
                                std::vector<int>& faceStructIds,
                                const double shrinkFactor = 5.f);

void compute_and_save_shrink_mstructures(const std::string& maname,
                                         const MedialMesh& mmesh,
                                         std::vector<Vector3>& shrinkVertices,
                                         std::vector<aint3>& shrinkFaces,
                                         std::vector<int>& faceStructIds,
                                         const double shrinkFactor = 5.f);

bool export_convex_cells_obj(
    const Parameter params, const std::vector<MedialSphere>& all_medial_spheres,
    const std::vector<ConvexCellHost>& convex_cells_returned,
    std::string folder_name, std::string rpd_name, const int max_sf_fid,
    const bool is_boundary_only);

void export_input_tet_obj(const std::string& tet_name, const TetMesh& tet_mesh);

void export_ma_obj(const std::string& maname, const MedialMesh& mat);
void export_ma_obj_itr(std::string& folder_name, const std::string& maname,
                       const MedialMesh& mat);
void export_ma_obj_only(const std::string& maname, const MedialMesh& mat);
void export_ma_and_struct_clean(const std::string& maname, MedialMesh& mat);
void load_ma_and_struct_clean(const std::string& ma_path,
                              std::vector<MedialSphere>& all_medial_spheres,
                              MedialMesh& mat);
void export_spheres_normals(const MedialMesh& mmesh, const std::string& maname);

bool export_centers_xyz(const std::string& folder_name,
                        const std::string& file_prefix,
                        const std::string& filename,
                        const std::vector<Vector3>& vertices);

void export_MSER(const std::string& filename, const double mser);
void export_TQ(const std::string& filename, const double tq);

// for shibo
bool save_mstruct_houdini(const MedialMesh& mmesh, std::string mstruct_name,
                          const bool is_store_sheets_split = false,
                          bool is_debug = false);

// from MATTopo
void load_ma_dup_cnt(const std::string& ma_path,
                     std::vector<MedialSphere>& all_medial_spheres,
                     MedialMesh& mat);