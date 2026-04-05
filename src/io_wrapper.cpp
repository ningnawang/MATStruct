#include "io_wrapper.h"

std::string g_output_dir = "../out";

void set_output_dir(const std::string& dir) {
  g_output_dir = dir;
  while (!g_output_dir.empty() && g_output_dir.back() == '/')
    g_output_dir.pop_back();
}

// helper function
void get_mat_clean_and_map(const MedialMesh& mat, std::map<int, int>& map_matv,
                           std::map<int, int>& map_mate,
                           std::map<int, int>& map_matf,
                           std::vector<Vector4>& vertices,
                           std::vector<aint2>& edges,
                           std::vector<int>& edges_types,
                           std::vector<aint3>& faces) {
  map_matv.clear();
  map_mate.clear();
  map_matf.clear();
  vertices.clear();
  edges.clear();
  edges_types.clear();
  faces.clear();

  auto get_vertex_mapped_id = [&](const int vid) {
    if (map_matv.find(vid) == map_matv.end()) {
      // add a new vertex
      map_matv[vid] = map_matv.size();
    }
    return map_matv.at(vid);
  };

  // faces
  for (int f = 0; f < mat.faces.size(); f++) {
    const auto& face = mat.faces[f];
    if (face.is_deleted) continue;
    aint3 one_f;
    for (uint j = 0; j < 3; j++) {
      int vid = get_vertex_mapped_id(face.vertices_[j]);
      one_f[j] = vid;
    }
    std::sort(one_f.begin(), one_f.end());
    faces.push_back(one_f);
    map_matf[f] = faces.size() - 1;
  }
  printf("faces: %d \n", faces.size());

  // edges
  for (int e = 0; e < mat.edges.size(); e++) {
    const auto& edge = mat.edges[e];
    if (edge.is_deleted) continue;
    // if (edge.faces_.empty()) continue;
    int vid1 = get_vertex_mapped_id(edge.vertices_[0]);
    int vid2 = get_vertex_mapped_id(edge.vertices_[1]);
    aint2 one_e = {vid1, vid2};
    std::sort(one_e.begin(), one_e.end());
    edges.push_back(one_e);
    if (edge.is_intf) {
      edges_types.push_back(MedialEdgeType::INTF);
    } else if (edge.is_extf) {
      edges_types.push_back(MedialEdgeType::EXTF);
    } else {
      edges_types.push_back(MedialEdgeType::EOTHER);
    }
    map_mate[e] = edges.size() - 1;
  }
  printf("edges: %d \n", edges.size());

  // vertices
  // save from map_matv, to avoid (0,0,0,0) in .ma file
  vertices.resize(map_matv.size());
  for (const auto& v_pair : map_matv) {
    int old_vid = v_pair.first;
    int new_vid = v_pair.second;
    const auto& mat_v = mat.vertices->at(old_vid);
    vertices[new_vid] = Vector4(mat_v.center[0], mat_v.center[1],
                                mat_v.center[2], mat_v.radius);
  }
  printf("vertcies: %d \n", vertices.size());
}

/*
 * helper function
 * Format:
 * numStructs
 * structID structType numElements
 * f1/e1/v1 f2/e2/v2 f3/e3/v3 f4/e4/v3 ... fn/en/
 */
void export_struct_only(const std::string& ma_name_full,
                        const std::map<int, int>& map_matv,
                        const std::map<int, int>& map_mate,
                        const std::map<int, int>& map_matf, MedialMesh& mat) {
  std::ofstream fout;
  fout.open(ma_name_full, std::ofstream::out | std::ofstream::app);  //   append
  fout << mat.mstructure.size() << std::endl;
  for (const auto& one_struct : mat.mstructure) {
    fout << one_struct.id << " " << int(one_struct.type) << " ";
    switch (one_struct.type) {
      case MedialType::SHEET:
        fout << int(one_struct.m_face_ids.size()) << std::endl;
        for (const auto& mfid : one_struct.m_face_ids) {
          if (map_matf.empty())
            fout << mfid << " ";
          else {
            assert(map_matf.find(mfid) != map_matf.end());
            fout << map_matf.at(mfid) << " ";
          }
        }
        fout << std::endl;
        break;
      case MedialType::SEAM:
      case MedialType::BOUNDARY:
        fout << int(one_struct.m_edge_ids.size()) << std::endl;
        for (const auto& meid : one_struct.m_edge_ids) {
          if (map_mate.empty())
            fout << meid << " ";
          else {
            assert(map_mate.find(meid) != map_mate.end());
            fout << map_mate.at(meid) << " ";
          }
        }
        fout << std::endl;
        break;
      case MedialType::JUNCTION:
        fout << int(one_struct.m_sphere_ids.size()) << std::endl;
        for (const auto& mvid : one_struct.m_sphere_ids) {
          if (map_matv.empty())
            fout << mvid << " ";
          else {
            assert(map_matv.find(mvid) != map_matv.end());
            fout << map_matv.at(mvid) << " ";
          }
        }
        fout << std::endl;
        break;
      case MedialType::MUNKOWN:
        assert("Unknown medial type");
      default:
        assert("Unknown medial type");
    }
  }  // for each struct
  fout.close();
}

//  Function to export the mesh to an OBJ file
// 1. obj file is 1-based index!!!
// 2. T is either Vector3 or Vector4
// 3. if face_ids is given, then assign color based on the id
template <typename T>
bool export_mesh_obj(const std::string obj_file_path,
                     const std::vector<T>& vertices,
                     const std::vector<aint2>& edges,
                     const std::vector<aint3>& faces,
                     const std::vector<int>& face_flags = {}) {
  assert(obj_file_path.empty() == false);
  std::ofstream outFile(obj_file_path);
  if (!outFile.is_open()) {
    std::cerr << "Error: Unable to open file for writing: "
              << obj_file_path.c_str() << std::endl;
    return false;
  }

  // Reference the MTL file
  if (!face_flags.empty()) {
    assert(face_flags.size() == faces.size());
    outFile << "mtllib MatStructColorBar.mtl\n";
    outFile << "usemtl MatStructColorBar\n";  // Match material name in MTL
  }

  // Write vertices to the file
  for (const auto& vertex : vertices) {
    outFile << "v " << vertex[0] << " " << vertex[1] << " " << vertex[2]
            << "\n";
  }

  // Write edges to the file
  // NOTE: obj file is 1-based index!!!
  for (const auto& edge : edges) {
    outFile << "l " << edge[0] + 1 << " " << edge[1] + 1 << "\n";
  }

  // Write faces to the file
  // NOTE: obj file is 1-based index!!!
  if (!face_flags.empty()) {
    // The texture coordinates vt should range from [0,1]
    // in both U (horizontal) and V (vertical). So we normalize face_flags.
    double max_flag = *std::max_element(face_flags.begin(), face_flags.end());
    double min_flag = *std::min_element(face_flags.begin(), face_flags.end());
    std::unordered_map<int, double> flag_to_uv;
    for (size_t i = 0; i < faces.size(); i++) {
      int flag_id = face_flags[i];
      // Compute normalized UV coordinate (along U axis)
      if (flag_to_uv.find(flag_id) == flag_to_uv.end()) {
        // Normalize in range [0,1]
        // flag_to_uv[flag_id] = (flag_id - min_flag) / (max_flag - min_flag);
        flag_to_uv[flag_id] = RANDOM_01();
      }
    }
    // Write texture coordinates (UVs)
    for (const auto& [flag, uv] : flag_to_uv) {
      outFile << "vt " << uv << " 0\n";  // Fixed V=0.5 (middle of the texture)
    }
    // Write faces with UV mapping
    for (size_t i = 0; i < faces.size(); i++) {
      int flag_id = face_flags[i];

      int vt_index =
          std::distance(flag_to_uv.begin(), flag_to_uv.find(flag_id)) + 1;
      outFile << "f " << faces[i][0] + 1 << "/" << vt_index << " "
              << faces[i][1] + 1 << "/" << vt_index << " " << faces[i][2] + 1
              << "/" << vt_index << "\n";
    }
  } else {
    // Write faces with NO UV mapping
    for (const auto& face : faces) {
      outFile << "f " << face[0] + 1 << " " << face[1] + 1 << " " << face[2] + 1
              << "\n";
    }
  }

  outFile.close();
  std::cout << "Mesh successfully exported to " << obj_file_path.c_str()
            << std::endl;
  return true;
}

//  Function to export the mesh to an OBJ file
// 1. obj file is 1-based index!!!
// 2. T is either Vector3 or Vector4
// 3. if face_ids is given, then assign color based on the id
template <typename T>
void export_mesh_obj(const std::string folder_name, const std::string& filename,
                     const std::vector<T>& vertices,
                     const std::vector<aint2>& edges,
                     const std::vector<aint3>& faces,
                     const std::vector<int>& face_flags = {}) {
  create_dir(folder_name);
  std::string ma_name_full =
      folder_name + filename + "_" + get_timestamp() + ".obj";
  export_mesh_obj(ma_name_full, vertices, edges, faces, face_flags);
}

// NOTE: obj file is 1-based index!!!
// Function to export edges with cleaned vertices to an OBJ file
bool export_mat_feature_edge_obj(const std::string& folder_name,
                                 const std::string& maname,
                                 const std::vector<Vector4>& vertices,
                                 const std::vector<aint2>& edges,
                                 const std::vector<int>& edge_types,
                                 int target_type) {
  if (edges.size() != edge_types.size()) {
    std::cerr << "Error: edges and edge_types vectors must have the same size."
              << std::endl;
    return false;
  }

  // Filter edges by type
  std::unordered_set<int> usedVertices;
  std::vector<aint2> filteredEdges;

  for (size_t i = 0; i < edges.size(); ++i) {
    if (edge_types[i] == target_type) {
      filteredEdges.push_back(edges[i]);
      usedVertices.insert(edges[i][0]);
      usedVertices.insert(edges[i][1]);
    }
  }

  // Map old vertex indices to new indices
  std::vector<Vector4> cleanedVertices;
  std::unordered_map<int, int> indexMapping;
  int newIndex = 1;

  // NOTE: obj file is 1-based index!!!
  for (size_t i = 0; i < vertices.size(); ++i) {
    if (usedVertices.count(i)) {  // 0-based index
      cleanedVertices.push_back(vertices[i]);
      indexMapping[i] = newIndex++;  // Map 0-based to 1-based index
    }
  }

  // Update edges to use new indices
  std::vector<aint2> updatedEdges;
  for (const auto& edge : filteredEdges) {
    updatedEdges.push_back({indexMapping[edge[0]], indexMapping[edge[1]]});
  }

  // Create directory and file path
  create_dir(folder_name);
  std::string ma_wire = folder_name + "/mat_wire" +
                        std::to_string(target_type) + "_" + maname + "_" +
                        get_timestamp() + ".obj";
  std::ofstream outFile(ma_wire);
  if (!outFile.is_open()) {
    std::cerr << "Error: Unable to open file for writing: " << ma_wire
              << std::endl;
    return false;
  }

  // Write vertices
  for (const auto& vertex : cleanedVertices) {
    outFile << "v " << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
  }

  // Write edges
  for (const auto& edge : updatedEdges) {
    outFile << "l " << edge[0] << " " << edge[1] << "\n";  // Already 1-based
  }

  outFile.close();
  std::cout << "MAT edge of type " << target_type
            << " successfully exported to " << ma_wire << "." << std::endl;
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Functions
////////////////////////////////////////////////////////////////////////////////////////////////////////

// /*
//  * Format:
//  * numVertices numEdges numFaces
//  * v x y z r flag_type flag_delete
//  * e v1 v2 flag_type flag_delete
//  * f v1 v2 v3 importance flag_delete
//  * numStructs
//  * structID structType numElements
//  * f1/e1/v1 f2/e2/v2 f3/e3/v3 f4/e4/v3 ... fn/en/
//  */
// // conains some deleted vertices, edges, faces
// // not cleaned ma
// void export_ma_and_struct_all(const std::string& maname, MedialMesh& mat) {
//   std::string folder_name = g_output_dir + "/" + maname + "/mat";
//   create_dir(folder_name);
//   std::string ma_name_full =
//       folder_name + "/mat_" + maname + "_" + get_timestamp() +
//       ".ma_struct_all";
//   std::ofstream fout(ma_name_full);
//   fout << mat.vertices->size() << " " << mat.edges.size() << " "
//        << mat.faces.size() << std::endl;

//   // mat vertices
//   for (auto& mat_v : *mat.vertices) {
//     fout << "v " << std::setiosflags(std::ios::fixed) <<
//     std::setprecision(15)
//          << mat_v.center[0] << " " << mat_v.center[1] << " " <<
//          mat_v.center[2]
//          << " " << mat_v.radius << " " << int(mat_v.type);
//     if (mat_v.is_deleted) {
//       fout << " 1";
//     } else {
//       fout << " 0";
//     }
//     fout << std::endl;
//   }

//   // mat edges
//   for (int i = 0; i < mat.edges.size(); i++) {
//     const auto& mat_e = mat.edges[i];
//     aint2 edge = {{mat_e.vertices_[0], mat_e.vertices_[1]}};
//     std::sort(edge.begin(), edge.end());
//     fout << "e " << edge[0] << " " << edge[1];
//     // is this edge a feature edge
//     if (mat_e.is_intf) {
//       fout << " " << 1;  // internal
//     } else if (mat_e.is_extf) {
//       fout << " " << 2;  // external
//     } else {
//       fout << " " << 0;
//     }
//     fout << " " << mat_e.is_deleted;
//     fout << std::endl;
//   }

//   // mat faces
//   for (int i = 0; i < mat.faces.size(); i++) {
//     const auto& mat_f = mat.faces[i];
//     fout << "f";
//     for (uint v = 0; v < 3; v++) fout << " " << mat_f.vertices_[v];
//     fout << " " << std::setiosflags(std::ios::fixed) << std::setprecision(15)
//          << mat_f.importance;
//     fout << " " << mat_f.is_deleted;
//     fout << std::endl;
//   }

//   fout.close();

//   // store mat mstructure
//   std::map<int, int> map_matv;
//   std::map<int, int> map_mate;
//   std::map<int, int> map_matf;
//   export_struct_only(ma_name_full, map_matv, map_mate, map_matf, mat);

//   printf("saved mat_struct_all at: %s \n", ma_name_full.c_str());
// }

void compute_shrink_mstructures(const MedialMesh& mmesh,
                                std::vector<Vector3>& shrinkVertices,
                                std::vector<aint3>& shrinkFaces,
                                std::vector<int>& faceStructIds,
                                const double shrinkFactor) {
  if (mmesh.vertices == nullptr) return;
  if (mmesh.mstructure.empty()) return;
  shrinkVertices.clear();
  shrinkFaces.clear();
  faceStructIds.clear();

  auto get_sphere_center = [&](const int vid) {
    return mmesh.vertices->at(vid).center;
  };

  // cannot run in parallel
  auto process_one_mstruct = [&](const MedialStruct& one_mstruct) {
    if (one_mstruct.type != MedialType::SHEET) return;
    std::map<int, int> vertex_old2new;
    // all save the old vids/eids
    std::unordered_map<int, int> edgeCount;
    std::unordered_set<int> boundaryVertices;
    std::unordered_map<int, Vector3> inwardDirections;

    // Step 1: add all vertices to the shrinkVertices
    for (const auto& fid : one_mstruct.m_face_ids) {
      const auto& mface = mmesh.faces[fid];
      for (int vid : mface.vertices_) {
        if (vertex_old2new.find(vid) != vertex_old2new.end()) continue;
        vertex_old2new[vid] = shrinkVertices.size();
        shrinkVertices.push_back(get_sphere_center(vid));
      }
    }

    // Step 2: Identify boundary edges in a sheet
    for (const auto& fid : one_mstruct.m_face_ids) {
      const auto& mface = mmesh.faces[fid];
      for (int eid : mface.edges_) edgeCount[eid]++;
    }

    // Step 3: Identify boundary vertices and compute inward directions
    for (const auto& fid : one_mstruct.m_face_ids) {
      const auto& mface = mmesh.faces[fid];
      // save faces in shrinkFaces
      shrinkFaces.push_back({vertex_old2new[mface.vertices_[0]],
                             vertex_old2new[mface.vertices_[1]],
                             vertex_old2new[mface.vertices_[2]]});
      faceStructIds.push_back(one_mstruct.id);
      for (int eid : mface.edges_) {
        const auto& medge = mmesh.edges[eid];
        // Not a boundary edge, then skip
        assert(edgeCount.find(eid) != edgeCount.end());
        if (edgeCount[eid] != 1) continue;
        int vA = mmesh.edges.at(eid).vertices_[0];
        int vB = mmesh.edges.at(eid).vertices_[1];
        boundaryVertices.insert(vA);
        boundaryVertices.insert(vB);
        Vector3 edgeVec =
            GEO::normalize(get_sphere_center(vB) - get_sphere_center(vA));
        // Get the adjacent face normal
        const auto& mface = mmesh.faces[fid];
        Vector3 v0 = get_sphere_center(mface.vertices_[0]);
        Vector3 v1 = get_sphere_center(mface.vertices_[1]);
        Vector3 v2 = get_sphere_center(mface.vertices_[2]);
        Vector3 faceNormal = get_normal(v0, v1, v2);
        // Compute inward direction: perpendicular to edgeVec and faceNormal
        Vector3 inwardDir = GEO::normalize(GEO::cross(faceNormal, edgeVec));

        // Find the non-boundary vertex in the face
        int nonBoundaryVertex =
            (mface.vertices_[0] != vA && mface.vertices_[0] != vB)
                ? mface.vertices_[0]
            : (mface.vertices_[1] != vA && mface.vertices_[1] != vB)
                ? mface.vertices_[1]
                : mface.vertices_[2];
        Vector3 nonBoundaryPos = get_sphere_center(nonBoundaryVertex);

        // Ensure inwardDir points towards the non-boundary vertex
        Vector3 checkDir =
            GEO::normalize(nonBoundaryPos - get_sphere_center(vA));
        if (GEO::dot(checkDir, inwardDir) < 0) {
          inwardDir = -inwardDir;
        }

        inwardDirections[vA] += inwardDir;
        inwardDirections[vB] += inwardDir;
      }
    }

    // printf("mstruct %d has %d boundary vertices\n", one_mstruct.id,
    //        boundaryVertices.size());

    // Step 4: update boundary vertices inward
    for (int v_old_id : boundaryVertices) {
      assert(vertex_old2new.find(v_old_id) != vertex_old2new.end());
      Vector3 moveDir = GEO::normalize(inwardDirections[v_old_id]);
      shrinkVertices[vertex_old2new[v_old_id]] += shrinkFactor * moveDir;
    }
  };

  for (const auto& one_mstruct : mmesh.mstructure) {
    process_one_mstruct(one_mstruct);
  }
}

void compute_and_save_shrink_mstructures(const std::string& maname,
                                         const MedialMesh& mmesh,
                                         std::vector<Vector3>& shrinkVertices,
                                         std::vector<aint3>& shrinkFaces,
                                         std::vector<int>& faceStructIds,
                                         const double shrinkFactor) {
  compute_shrink_mstructures(mmesh, shrinkVertices, shrinkFaces, faceStructIds,
                             shrinkFactor);
  std::string folder_name = g_output_dir + "/" + maname + "/mat/";
  create_dir(folder_name);
  std::string ma_name_full =
      folder_name + "mat_shrink_" + maname + "_" + get_timestamp() + ".obj";
  export_mesh_obj<Vector3>(folder_name, "mat_shrink_" + maname, shrinkVertices,
                           {}, shrinkFaces, faceStructIds);
  printf("saved shrink mat at: %s \n", ma_name_full.c_str());
}

bool export_convex_cells_obj(
    const Parameter params, const std::vector<MedialSphere>& all_medial_spheres,
    const std::vector<ConvexCellHost>& convex_cells_returned,
    std::string folder_name, std::string rpd_name, const int max_sf_fid,
    const bool is_boundary_only) {
  if (folder_name.empty()) {
    folder_name = g_output_dir + "/" + rpd_name + "/rpd/";
  }
  create_dir(folder_name);

  std::vector<cfloat3> voro_points;
  std::vector<std::vector<unsigned>> all_voro_faces;
  std::vector<int> voro_faces_sites, _;
  for (const auto& cc_trans : convex_cells_returned) {
    get_one_convex_cell_faces_const(cc_trans, voro_points, all_voro_faces,
                                    voro_faces_sites, _, true /*is_triangle*/,
                                    max_sf_fid, is_boundary_only);
  }
  assert(voro_faces_sites.size() == all_voro_faces.size());

  // save obj
  std::vector<Vector3> voro_points_vec;
  for (const auto& point : voro_points) {
    voro_points_vec.push_back(Vector3(point.x, point.y, point.z));
  }
  std::vector<aint3> voro_triangles;
  for (const auto& face : all_voro_faces) {
    assert(face.size() == 3);
    aint3 one_face = {face[0], face[1], face[2]};
    voro_triangles.push_back(one_face);
  }
  export_mesh_obj<Vector3>(folder_name, rpd_name, voro_points_vec, {},
                           voro_triangles, voro_faces_sites);
}

// each tet has its own vertices copy
void export_input_tet_obj(const std::string& tet_name,
                          const TetMesh& tet_mesh) {
  std::string folder_name = g_output_dir + "/" + tet_name + "/input/";
  create_dir(folder_name);

  static const int tet_face_indices[4][3] = {
      {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};

  std::vector<Vector3> tet_vertices;
  std::vector<aint3> tet_faces;
  std::vector<int> tet_faces_flags;
  std::map<int, int> tet_indices_new;
  for (int i = 0; i < tet_mesh.tet_indices.size() / 4; i++) {
    // save all 4 vertices
    for (int j = 0; j < 4; j++) {
      int vid = tet_mesh.tet_indices[i * 4 + j];
      tet_vertices.push_back(Vector3(tet_mesh.tet_vertices[vid * 3],
                                     tet_mesh.tet_vertices[vid * 3 + 1],
                                     tet_mesh.tet_vertices[vid * 3 + 2]));
    }

    // save all faces
    for (int f = 0; f < 4; ++f) {
      aint3 one_face = {tet_vertices.size() - 4 + tet_face_indices[f][0],
                        tet_vertices.size() - 4 + tet_face_indices[f][1],
                        tet_vertices.size() - 4 + tet_face_indices[f][2]};
      tet_faces.push_back(one_face);
      tet_faces_flags.push_back(i);
    }
  }

  export_mesh_obj<Vector3>(folder_name, "tet_" + tet_name, tet_vertices, {},
                           tet_faces, tet_faces_flags);
}

void export_ma_junctions(const std::string& folder_name,
                         const std::string& maname, const MedialMesh& mat) {
  create_dir(folder_name);
  std::vector<Vector3> junctions;
  for (int vid = 0; vid < mat.vertices->size(); vid++) {
    const auto& msphere = mat.vertices->at(vid);
    if (msphere.mstruct_jun_id > -1) junctions.push_back(msphere.center);
  }
  export_centers_xyz(folder_name, "mat_junctions", maname, junctions);
}

void export_ma_obj(const std::string& maname, const MedialMesh& mat) {
  std::string folder_name = g_output_dir + "/" + maname + "/mat/";
  create_dir(folder_name);

  std::map<int, int> map_matv, map_mate, map_matf;
  std::vector<Vector4> vertices;
  std::vector<aint2> edges;
  std::vector<int> edges_types;
  std::vector<aint3> faces;
  get_mat_clean_and_map(mat, map_matv, map_mate, map_matf, vertices, edges,
                        edges_types, faces);

  ///////////////////////////
  // export clean mat as obj
  export_mesh_obj<Vector4>(folder_name, "mat_" + maname, vertices, edges,
                           faces);

  ///////////////////////////
  // export junctions as xyz file
  export_ma_junctions(folder_name, maname, mat);
}

void export_ma_obj_itr(std::string& folder_name, const std::string& maname,
                       const MedialMesh& mat) {
  if (folder_name.empty()) {
    folder_name = g_output_dir + "/" + maname + "/mmesh_itr/";
  }
  create_dir(folder_name);

  std::map<int, int> map_matv, map_mate, map_matf;
  std::vector<Vector4> vertices;
  std::vector<aint2> edges;
  std::vector<int> _;
  std::vector<aint3> faces;
  get_mat_clean_and_map(mat, map_matv, map_mate, map_matf, vertices, edges, _,
                        faces);

  ///////////////////////////
  // export clean mat as obj
  export_mesh_obj<Vector4>(folder_name, maname, vertices, edges, faces);
}

void export_ma_obj_only(const std::string& maname, const MedialMesh& mat) {
  std::string folder_name = g_output_dir + "/" + maname + "/mat/";
  create_dir(folder_name);

  std::map<int, int> map_matv, map_mate, map_matf;
  std::vector<Vector4> vertices;
  std::vector<aint2> edges;
  std::vector<int> edges_types;
  std::vector<aint3> faces;
  get_mat_clean_and_map(mat, map_matv, map_mate, map_matf, vertices, edges,
                        edges_types, faces);
  ///////////////////////////
  // export clean mat as obj
  export_mesh_obj<Vector4>(folder_name, "mat_" + maname, vertices, edges,
                           faces);
  // export clean mat
  printf("calling export_ma_given..\n");
  export_ma_given(maname, vertices, edges, faces, false /*is_use_given_name*/);
}

/*
 * Format:
 * numVertices numEdges numFaces
 * v x y z r
 * e v1 v2
 * f v1 v2 v3
 * numStructs
 * structID structType numElements
 * f1/e1/v1 f2/e2/v2 f3/e3/v3 f4/e4/v3 ... fn/en/
 */
void export_ma_and_struct_clean(const std::string& maname, MedialMesh& mat) {
  std::string folder_name = g_output_dir + "/" + maname + "/mat/";
  create_dir(folder_name);

  std::map<int, int> map_matv, map_mate, map_matf;
  std::vector<Vector4> vertices;
  std::vector<aint2> edges;
  std::vector<int> edges_types;
  std::vector<aint3> faces;
  get_mat_clean_and_map(mat, map_matv, map_mate, map_matf, vertices, edges,
                        edges_types, faces);
  ///////////////////////////
  // export clean mat as obj
  export_mesh_obj<Vector4>(folder_name, "mat_" + maname, vertices, edges,
                           faces);
  // export clean mat
  printf("calling export_ma_given..\n");
  export_ma_given(maname, vertices, edges, faces, false /*is_use_given_name*/);

  ///////////////////////////
  // export intf/extf edges only
  export_mat_feature_edge_obj(folder_name, maname, vertices, edges, edges_types,
                              MedialEdgeType::INTF);
  export_mat_feature_edge_obj(folder_name, maname, vertices, edges, edges_types,
                              MedialEdgeType::EXTF);
  ///////////////////////////
  // export junctions as xyz file
  export_ma_junctions(folder_name, maname, mat);

  // ///////////////////////////
  // // save shrink mstructures
  // std::vector<Vector3> shrinkVertices;
  // std::vector<aint3> shrinkFaces;
  // std::vector<int> faceStructIds;
  // compute_and_save_shrink_mstructures(maname, mat, shrinkVertices,
  // shrinkFaces, faceStructIds);
}

/*
 * Format:
 * numVertices numEdges numFaces
 * v x y z r
 * e v1 v2
 * f v1 v2 v3
 * numStructs
 * structID structType(0-sheet, 1-seam, 2-boundary, 3-junction) numElements
 * f1/e1/v1 f2/e2/v2 f3/e3/v3 f4/e4/v3 ... fn/en/
 * ...
 *
 */
void load_ma_and_struct_clean(const std::string& ma_path,
                              std::vector<MedialSphere>& all_medial_spheres,
                              MedialMesh& mat) {
  std::ifstream ma_file(ma_path);
  int num_vs, num_es, num_fs;
  char ch;
  ma_file >> num_vs >> num_es >> num_fs;
  printf("loading .ma with num_vs: %d, num_es: %d, num_fs: %d from %s\n",
         num_vs, num_es, num_fs, ma_path.c_str());

  // load spheres
  mat.clear();
  all_medial_spheres.clear();
  for (int i = 0; i < num_vs; i++) {
    MedialSphere msphere(all_medial_spheres.size(), Vector3(0, 0, 0),
                         Vector3(0, 0, 0), SphereType::T_2);
    ma_file >> ch >> msphere.center[0] >> msphere.center[1] >>
        msphere.center[2];
    ma_file >> msphere.radius;
    all_medial_spheres.push_back(msphere);
  }
  mat.vertices = &all_medial_spheres;
  // load edge
  int e1, e2;
  for (int e = 0; e < num_es; e++) {
    ma_file >> ch >> e1 >> e2;
    // printf("ma e %d has (%d,%d)\n", e, e1, e2);
    int eid = mat.create_edge(e1, e2);
    auto& medge = mat.edges.at(eid);
  }
  // load facees
  int f1, f2, f3;
  for (int f = 0; f < num_fs; f++) {
    ma_file >> ch >> f1 >> f2 >> f3;
    // printf("ma f %d has (%d,%d,%d)\n", f, f1, f2, f3);
    aint3 fvs = {{f1, f2, f3}};
    int mfid = mat.create_face(fvs);
  }

  // load mat structure
  int num_structs, structID, structType, numElements;
  int ele;
  ma_file >> num_structs;
  for (int str = 0; str < num_structs; str++) {
    ma_file >> structID >> structType >> numElements;
    MedialStruct mstruct(structID, MedialType(structType));
    for (int i = 0; i < numElements; i++) {
      ma_file >> ele;
      switch (mstruct.type) {
        case MedialType::SHEET:
          mstruct.m_face_ids.insert(ele);
          mat.faces.at(ele).mstruct_id = structID;
          break;
        case MedialType::SEAM:
        case MedialType::BOUNDARY:
          mstruct.m_edge_ids.insert(ele);
          mat.edges.at(ele).mstruct_id = structID;
          break;
        case MedialType::JUNCTION:
          mstruct.m_sphere_ids.insert(ele);
          mat.vertices->at(ele).mstruct_jun_id = structID;
          break;
        default:
          assert("Unknown medial type");
      }
    }
    mat.mstructure.push_back(mstruct);
  }
  ma_file.close();
  printf("loaded .ma_struct with %d structs from %s\n", num_structs,
         ma_path.c_str());
}

// 'map_vid_old_to_new_copies': MedialSphere::id -> vector of local_vids.
// One copy for each sheet, if vertex only on sheet, then only one copy.
void save_mstruct_houdini_split_sheet_only_helper(
    const MedialMesh& mmesh, std::string mstruct_name,
    std::map<int, std::set<int>>& map_vid_old_to_new_copies /*global*/) {
  if (mmesh.vertices == nullptr) return;
  if (mmesh.mstructure.empty()) return;
  std::string folder_name = g_output_dir + "/" + mstruct_name + "/mstruct/";
  create_dir(folder_name);
  std::string mstruct_sheet_path =
      folder_name + "mstruct_sheet_" + mstruct_name + "_" + get_timestamp();
  printf("saving mstruct_sheet_path: %s, mstructure size %zu\n",
         mstruct_sheet_path.c_str(), mmesh.mstructure.size());

  // each vertex on sheet has it's own copied
  std::vector<float3> msphere_centers_new_copies;
  std::vector<std::vector<unsigned>> all_sheets_faces;
  std::vector<int> all_struct_ids;
  std::vector<int> all_struct_types;
  map_vid_old_to_new_copies.clear();

  // each sheet should has only 1 vertex copy
  std::map<int, int> map_vid_old_to_new_within_one_sheet; /*local*/
  std::vector<unsigned> one_seam_sheet;
  for (const auto& mstruct : mmesh.mstructure) {
    one_seam_sheet.clear();
    map_vid_old_to_new_within_one_sheet.clear();
    if (mstruct.type != MedialType::SHEET) continue;
    for (const auto& mfid : mstruct.m_face_ids) {
      one_seam_sheet.clear();
      const auto& mface = mmesh.faces.at(mfid);
      // loop vertices
      for (int v = 0; v < mface.vertices_.size(); v++) {
        int old_vid = mface.vertices_[v];
        const auto& msphere = mmesh.vertices->at(old_vid);
        if (map_vid_old_to_new_within_one_sheet.find(old_vid) !=
            map_vid_old_to_new_within_one_sheet.end()) {
          // already has a vertex copy in this sheet
          one_seam_sheet.push_back(
              (unsigned)map_vid_old_to_new_within_one_sheet[old_vid]);
        } else {
          // create a new vertex copy
          msphere_centers_new_copies.push_back({(float)msphere.center[0],
                                                (float)msphere.center[1],
                                                (float)msphere.center[2]});
          int new_vid = msphere_centers_new_copies.size() - 1;
          map_vid_old_to_new_within_one_sheet[old_vid] = new_vid;
          map_vid_old_to_new_copies[old_vid].insert(new_vid);
          one_seam_sheet.push_back((unsigned)new_vid);
        }
      }
      all_sheets_faces.push_back(one_seam_sheet);
      all_struct_ids.push_back(mstruct.id);
      all_struct_types.push_back(mstruct.type);
    }
  }

  printf(
      "msphere_centers_new_copies size: %zu, all_sheets_faces size: %zu, "
      "all_struct_ids size %zu\n",
      msphere_centers_new_copies.size(), all_sheets_faces.size(),
      all_struct_ids.size());

  assert(all_struct_ids.size() == all_sheets_faces.size());

  // save houdini file
  IO::Geometry geometry;
  IO::GeometryWriter geometry_writer(".");
  geometry.AddParticleAttribute("P", msphere_centers_new_copies);
  geometry.AddPolygon(all_sheets_faces);
  geometry.AddPrimitiveAttribute("mstruct_id", all_struct_ids);
  geometry.AddPrimitiveAttribute("mstruct_type", all_struct_types);
  geometry_writer.OutputGeometry(mstruct_sheet_path, geometry);
  printf("saved houdini file: %s \n", mstruct_sheet_path);
}

bool load_mstruct_jun_seam_sheet_helper(
    const MedialMesh& mmesh,
    std::vector<std::vector<unsigned>>& junc_seam_sheet,
    std::vector<int>& all_struct_ids, std::vector<int>& all_struct_types) {
  // save seam, boundary, sheet, junction
  std::vector<unsigned> one_seam_sheet;
  for (const auto& mstruct : mmesh.mstructure) {
    one_seam_sheet.clear();
    if (mstruct.type == MedialType::JUNCTION) {
      one_seam_sheet.clear();
      assert(mstruct.m_sphere_ids.size() == 1);
      for (const auto& mvid : mstruct.m_sphere_ids) {
        one_seam_sheet.push_back((unsigned)mvid);
      }
      junc_seam_sheet.push_back(one_seam_sheet);
      all_struct_ids.push_back(mstruct.id);
      all_struct_types.push_back(mstruct.type);
    } else if (mstruct.type == MedialType::SEAM ||
               mstruct.type == MedialType::BOUNDARY) {
      for (const auto& meid : mstruct.m_edge_ids) {
        one_seam_sheet.clear();
        const auto& medge = mmesh.edges.at(meid);
        one_seam_sheet.push_back((unsigned)medge.vertices_[0]);
        one_seam_sheet.push_back((unsigned)medge.vertices_[1]);
        junc_seam_sheet.push_back(one_seam_sheet);
        all_struct_ids.push_back(mstruct.id);
        all_struct_types.push_back(mstruct.type);
      }
    } else if (mstruct.type == MedialType::SHEET) {
      for (const auto& mfid : mstruct.m_face_ids) {
        one_seam_sheet.clear();
        const auto& mface = mmesh.faces.at(mfid);
        one_seam_sheet.push_back((unsigned)mface.vertices_[0]);
        one_seam_sheet.push_back((unsigned)mface.vertices_[1]);
        one_seam_sheet.push_back((unsigned)mface.vertices_[2]);
        junc_seam_sheet.push_back(one_seam_sheet);
        all_struct_ids.push_back(mstruct.id);
        all_struct_types.push_back(mstruct.type);
      }
    }
  }
  printf("junc_seam_sheet size: %zu, all_struct_ids size %zu\n",
         junc_seam_sheet.size(), all_struct_ids.size());
  assert(all_struct_ids.size() == junc_seam_sheet.size());
}

// Store two houdini files:
//  1. call save_mstruct_houdini_split_sheet_only_helper() to save sheet only
//  2. save all types of mstructs, for each vertex, save its copies in step1.
bool save_mstruct_houdini(const MedialMesh& mmesh, std::string mstruct_name,
                          const bool is_store_sheets_split, bool is_debug) {
  std::string folder_name = g_output_dir + "/" + mstruct_name + "/mstruct/";
  create_dir(folder_name);
  std::string mstruct_all_path =
      folder_name + "mstruct_all_" + mstruct_name + "_" + get_timestamp();
  printf("saving mstruct_all_path: %s, mstructure size %zu\n",
         mstruct_all_path.c_str(), mmesh.mstructure.size());

  //////////////////////////////////////////
  // each sheet store a vertex copy
  // there might be deleted/isolated vertices
  std::map<int, std::set<int>> map_vid_old_to_new_copies;
  if (is_store_sheets_split) {
    save_mstruct_houdini_split_sheet_only_helper(mmesh, mstruct_name,
                                                 map_vid_old_to_new_copies);
    // get max num of new copies
    int max_num_new_copies = 0;
    for (const auto& vid_new_copies : map_vid_old_to_new_copies) {
      max_num_new_copies =
          std::max(max_num_new_copies, (int)vid_new_copies.second.size());
    }
    if (is_debug) printf("max_num_new_copies: %d\n", max_num_new_copies);
    assert(max_num_new_copies <= 4);  // dunno what to do if > 4
  }

  //////////////////////////////////////////
  // store mspheres,
  // and their copies ids if is_store_sheets_split = true
  std::vector<float3> msphere_centers;
  std::vector<int4> msphere_new_copies;
  for (const auto& msphere : *mmesh.vertices) {
    msphere_centers.push_back({(float)msphere.center[0],
                               (float)msphere.center[1],
                               (float)msphere.center[2]});
    int4 attr_copies = {-1, -1, -1, -1};
    // there might be deleted/isolated vertices
    if (is_store_sheets_split && map_vid_old_to_new_copies.find(msphere.id) !=
                                     map_vid_old_to_new_copies.end()) {
      const auto& tmp = map_vid_old_to_new_copies.at(msphere.id);
      // std::set to std::vector
      std::vector<int> vid_new_copies(tmp.begin(), tmp.end());
      std::sort(vid_new_copies.begin(), vid_new_copies.end());
      assert(vid_new_copies.size() <= 4);
      for (int i = 0; i < vid_new_copies.size(); i++) {
        if (i == 0) attr_copies.x = vid_new_copies[i];
        if (i == 1) attr_copies.y = vid_new_copies[i];
        if (i == 2) attr_copies.z = vid_new_copies[i];
        if (i == 3) attr_copies.w = vid_new_copies[i];
      }
    }
    msphere_new_copies.push_back(attr_copies);
  }
  assert(msphere_centers.size() == msphere_new_copies.size());
  if (is_debug)
    printf("msphere_centers size: %zu, msphere_new_copies size: %zu\n",
           msphere_centers.size(), msphere_new_copies.size());

  //////////////////////////////////////////
  std::vector<std::vector<unsigned>> junc_seam_sheet;
  std::vector<int> all_struct_ids;
  std::vector<int> all_struct_types;
  load_mstruct_jun_seam_sheet_helper(mmesh, junc_seam_sheet, all_struct_ids,
                                     all_struct_types);
  assert(all_struct_ids.size() == junc_seam_sheet.size());

  // save houdini file
  IO::Geometry geometry;
  IO::GeometryWriter geometry_writer(".");
  geometry.AddParticleAttribute("P", msphere_centers);
  geometry.AddParticleAttribute("sheets_vids", msphere_new_copies);
  geometry.AddPolygon(junc_seam_sheet);
  geometry.AddPrimitiveAttribute("mstruct_id", all_struct_ids);
  geometry.AddPrimitiveAttribute("mstruct_type", all_struct_types);
  geometry_writer.OutputGeometry(mstruct_all_path, geometry);
  printf("saved houdini file: %s \n", mstruct_all_path.c_str());
}

void export_spheres_normals(const MedialMesh& mmesh,
                            const std::string& maname) {
  if (mmesh.vertices == nullptr) return;
  const auto& all_medial_spheres = *mmesh.vertices;

  std::string folder_name = g_output_dir + "/" + maname + "/normals/";
  create_dir(folder_name);
  std::string normals_path =
      folder_name + "normals_" + maname + "_" + get_timestamp() + ".normal";
  printf("saving normals_path: %s, all_medial_spheres size %zu\n",
         normals_path.c_str(), all_medial_spheres.size());

  std::ofstream fout;
  fout.open(normals_path, std::ofstream::out | std::ofstream::app);  //   append
  fout << all_medial_spheres.size() << std::endl;
  for (int i = 0; i < all_medial_spheres.size(); i++) {
    const auto& msphere = all_medial_spheres[i];
    if (!msphere.tan_planes.empty()) {
      fout << i << " " << msphere.tan_planes.size() << std::endl;
      for (const auto& tan_plane : msphere.tan_planes) {
        fout << ss_params::encode_fid(tan_plane.fid, false) << " "
             << tan_plane.tan_point[0] << " " << tan_plane.tan_point[1] << " "
             << tan_plane.tan_point[2] << std::endl;
      }
    }
    if (!msphere.tan_cc_lines.empty()) {
      fout << i << " " << msphere.tan_cc_lines.size() << std::endl;
      for (const auto& tan_cc_line : msphere.tan_cc_lines) {
        fout << ss_params::encode_fid(tan_cc_line.id_fe, true) << " "
             << tan_cc_line.tan_point[0] << " " << tan_cc_line.tan_point[1]
             << " " << tan_cc_line.tan_point[2] << std::endl;
      }
    }
  }
  fout.close();
}

bool export_centers_xyz(const std::string& folder_name,
                        const std::string& file_prefix,
                        const std::string& filename,
                        const std::vector<Vector3>& vertices) {
  // Create directory and file path
  create_dir(folder_name);
  std::string ma_junctions = folder_name + "/" + file_prefix + "_" + filename +
                             "_" + get_timestamp() + ".xyz";
  std::ofstream outFile(ma_junctions);
  if (!outFile.is_open()) {
    std::cerr << "Error: Unable to open file for writing: " << ma_junctions
              << std::endl;
    return false;
  }

  for (size_t i = 0; i < vertices.size(); ++i) {
    const auto& vertex = vertices[i];
    outFile << vertex[0] << " " << vertex[1] << " " << vertex[2] << "\n";
  }
  outFile.close();
  return true;
}

void export_MSER(const std::string& filename, const double mser) {
  std::string out = g_output_dir + "/MSER.txt";
  std::ofstream outFile;
  outFile.open(out, std::ios::app);  // append
  outFile << filename << ": " << mser << std::endl;
  outFile.close();
}

void export_TQ(const std::string& filename, const double tq) {
  std::string out = g_output_dir + "/TQ.txt";
  std::ofstream outFile;
  outFile.open(out, std::ios::app);  // append
  outFile << filename << ": " << tq << std::endl;
  outFile.close();
}

/*
 * from MATTopo
 * numVertices numEdges numFaces numTets
 * v x y z r flag_type flag_delete dup_cnt
 * e v1 v2 flag_type flag_delete dup_cnt
 * f v1 v2 v3 importance flag_delete dup_cnt
 * t v1 v2 v3 v4 flag_delete dup_cnt
 */
void load_ma_dup_cnt(const std::string& ma_path,
                     std::vector<MedialSphere>& all_medial_spheres,
                     MedialMesh& mat) {
  std::ifstream ma_file(ma_path);
  int num_vs, num_es, num_fs, num_ts;
  int type, is_deleted, dup_cnt;
  double importance;
  char ch;
  ma_file >> num_vs >> num_es >> num_fs >> num_ts;
  printf("loading .ma with num_vs: %d, num_es: %d, num_fs: %d, num_ts: %d\n",
         num_vs, num_es, num_fs, num_ts);

  // load spheres
  mat.clear();
  all_medial_spheres.clear();
  for (int i = 0; i < num_vs; i++) {
    MedialSphere msphere(all_medial_spheres.size(), Vector3(0, 0, 0),
                         Vector3(0, 0, 0), SphereType::T_2);
    ma_file >> ch >> msphere.center[0] >> msphere.center[1] >>
        msphere.center[2];
    ma_file >> msphere.radius;
    ma_file >> type;
    if (type != SphereType::T_UNK)
      msphere.type = SphereType(type);  // else as T2 sphere
    ma_file >> is_deleted;
    ma_file >> dup_cnt;
    msphere.is_deleted = is_deleted;
    msphere.dup_cnt = dup_cnt;
    all_medial_spheres.push_back(msphere);
    // mat
    if (!msphere.is_deleted) mat.numSpheres_active += dup_cnt;
  }
  mat.vertices = &all_medial_spheres;

  // load edge
  int e1, e2;
  for (int e = 0; e < num_es; e++) {
    ma_file >> ch >> e1 >> e2 >> type >> is_deleted >> dup_cnt;
    if (is_deleted) continue;
    // printf("ma e %d has (%d,%d)\n", e, e1, e2);
    int eid = mat.create_edge(e1, e2, dup_cnt);
    auto& medge = mat.edges.at(eid);
    if (type == 1) {
      medge.is_intf = true;
    } else if (type == 2) {
      medge.is_extf = true;
    }
    if (dup_cnt > 1) {
      printf("[MAT_DUP] edge %d (%d,%d) has dup_cnt %d\n", e, e1, e2, dup_cnt);
    }
  }
  // load facees
  int f1, f2, f3;
  for (int f = 0; f < num_fs; f++) {
    ma_file >> ch >> f1 >> f2 >> f3 >> importance >> is_deleted >> dup_cnt;
    printf("ma f %d has (%d,%d,%d), importance: %f\n", f, f1, f2, f3,
           importance);
    aint3 fvs = {{f1, f2, f3}};
    int mfid = mat.create_face(fvs, dup_cnt);
    mat.faces.at(mfid).importance = importance;
    if (dup_cnt > 1) {
      printf("[MAT_DUP] face %d (%d,%d,%d) has dup_cnt %d\n", f, f1, f2, f3,
             dup_cnt);
    }
  }
  // load tets
  int t1, t2, t3, t4;
  for (int t = 0; t < num_ts; t++) {
    ma_file >> ch >> t1 >> t2 >> t3 >> t4 >> is_deleted >> dup_cnt;
    aint4 tvs = {{t1, t2, t3, t4}};
    mat.create_tet(tvs, dup_cnt);

    // error?
    if (dup_cnt > 1) {
      printf("[MAT_DUP] ERROR: tet %d (%d,%d,%d,%d) has dup_cnt %d\n", t, t1,
             t2, t3, t4, dup_cnt);
    }
  }
  ma_file.close();
}