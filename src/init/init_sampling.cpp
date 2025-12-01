#include "init_sampling.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/poisson_disk_sampler.h"
#include "geometrycentral/surface/surface_point.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

void get_surface_poisson_disk_samples(const std::string& mesh_path,
                                      const double min_dist,
                                      const SurfaceMesh& sf_mesh,
                                      std::vector<v2int>& samples_fids,
                                      bool is_debug) {
  if (is_debug)
    printf("[PoissonDisk] calling get_surface_poisson_disk_samples...\n");
  samples_fids.clear();
  using namespace geometrycentral;
  using namespace geometrycentral::surface;

  // Load a mesh
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = readManifoldSurfaceMesh(mesh_path);
  if (is_debug)
    printf("[PoissonDisk] loaded mesh with %zu vertices and %zu faces\n",
           mesh->nVertices(), mesh->nFaces());

  // Detect SE vertices to avoid
  // Convert SE vertices from sf_mesh to geometrycentral mesh
  std::set<int> sf_se_vs_ids;
  GEO::Attribute<int> attr_corners(sf_mesh.vertices.attributes(), "corner");
  for (int v = 0; v < sf_mesh.vertices.nb(); v++) {
    if (attr_corners[v] == 1) {
      sf_se_vs_ids.insert(v);
    }
  }
  GEO::Attribute<int> attr_se(sf_mesh.edges.attributes(), "se");
  for (int e = 0; e < sf_mesh.edges.nb(); e++) {
    if (attr_se[e] == 1) {
      sf_se_vs_ids.insert((int)sf_mesh.edges.vertex(e, 0));
      sf_se_vs_ids.insert((int)sf_mesh.edges.vertex(e, 1));
    }
  }
  std::vector<SurfacePoint> mesh_se_points;
  for (Vertex v : mesh->vertices()) {
    if (sf_se_vs_ids.find(v.getIndex()) != sf_se_vs_ids.end()) {
      mesh_se_points.push_back(SurfacePoint(v));
    }
  }
  // if (is_debug)
  printf("[PoissonDisk] found %zu/%zu SE vertices to avoid\n",
         sf_se_vs_ids.size(), mesh_se_points.size());

  // Create a sampler
  PoissonDiskSampler poissonSampler(*mesh, *geometry);
  std::vector<SurfacePoint> samples_tmp;

  // Sample with some different parameters.
  assert(min_dist > 0);
  if (is_debug) printf("[PoissonDisk] sampling with min_dist %f\n", min_dist);
  PoissonDiskOptions sampleOptions;
  sampleOptions.minDist = min_dist;
  if (!mesh_se_points.empty()) {
    sampleOptions.pointsToAvoid = mesh_se_points;
    sampleOptions.minDistAvoidance = min_dist;
  }
  samples_tmp = poissonSampler.sample(sampleOptions);

  if (is_debug)
    printf("[PoissonDisk] sampled %ld points\n", samples_tmp.size());

  // fetch normals
  for (const SurfacePoint& sp : samples_tmp) {
    assert(sp.type == SurfacePointType::Face);
    auto& f = sp.face;

    int i = 0;
    geometrycentral::Vector3 v_new = geometrycentral::Vector3::zero();
    for (Vertex fv : f.adjacentVertices()) {
      auto& fv_pos = geometry->vertexPositions[fv];
      v_new += fv_pos * sp.faceCoords[i++];
    }

    GEO::vec3 p(v_new.x, v_new.y, v_new.z);
    // skip if too close to se
    double sq_dist_to_se = sf_mesh.aabb_wrapper.get_sq_dist_to_se(p);
    if (sq_dist_to_se < min_dist * min_dist) continue;
    samples_fids.push_back({p, f.getIndex()});
    // int p_fid = sf_mesh.aabb_wrapper.project_to_sf_get_nearest_face(p);
    // assert(p_fid != -1);
    // assert(p_fid == f.getIndex());
    // samples_fids.push_back({p, p_fid});
    if (is_debug) {
      printf("sample new p: %f %f %f\n", p[0], p[1], p[2]);
      // printf("f.getIndex(): %d, p_fid: %d\n", f.getIndex(), p_fid);
    }
  }
  printf("[PoissonDisk] sampled %ld points\n", samples_fids.size());
}

void load_poisson_samples(const std::string& in_path, const bool is_load_fid,
                          std::vector<v2int>& samples_fids) {
  std::ifstream in(in_path);
  if (in.fail()) return;
  samples_fids.clear();
  v2int pair;
  if (is_load_fid) {
    while (in >> pair.first[0] >> pair.first[1] >> pair.first[2] >>
           pair.second) {
      samples_fids.push_back(pair);
    }
  } else {
    while (in >> pair.first[0] >> pair.first[1] >> pair.first[2]) {
      pair.second = -1;
      samples_fids.push_back(pair);
    }
  }
  in.close();
  printf("[PoissonDisk] loaded %zu samples from %s\n", samples_fids.size(),
         in_path.c_str());
}

// is_save_fid = true: save (x,y,z)
// is_save_fid = false: save (x,y,z,fid)
void save_poission_samples(const std::string& out_path, const bool is_save_fid,
                           const std::vector<v2int>& samples_fids) {
  printf("[PoissonDisk] saving %zu samples to %s\n", samples_fids.size(),
         out_path.c_str());
  std::ofstream out(out_path);
  for (const auto& pair : samples_fids) {
    out << pair.first[0] << " " << pair.first[1] << " " << pair.first[2];
    if (is_save_fid)
      out << " " << pair.second << std::endl;
    else
      out << std::endl;
  }
  out.close();
}
