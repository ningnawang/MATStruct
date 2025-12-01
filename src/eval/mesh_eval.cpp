#include "mesh_eval.h"

double eval_triangle_quality(const MedialMesh& mmesh) {
  int i = 0;
  double Q_all = 0;
  for (const auto& mface : mmesh.faces) {
    const auto& p1 = mmesh.vertices->at(mface.vertices_[0]).center;
    const auto& p2 = mmesh.vertices->at(mface.vertices_[1]).center;
    const auto& p3 = mmesh.vertices->at(mface.vertices_[2]).center;
    double a = (p1 - p2).length();
    double b = (p2 - p3).length();
    double c = (p3 - p1).length();
    double edges[] = {a, b, c};
    double ht = *std::max_element(edges, edges + 3);
    double s = (a + b + c) / 2;
    double area = std::sqrt(s * (s - a) * (s - b) * (s - c));
    double Q = 0;
    try {
      Q = 6 / std::sqrt(3) * (area / (ht * s));
    } catch (const std::exception& e) {
      printf("[TriangleQuality] Error: Division by zero for mface %d\n", i);
      Q = 0;
    }
    Q_all += Q;
    i++;
  }
  double Q = Q_all / i;
  printf("[TriangleQuality] Q: %f, Q_all: %f, i: %d\n", Q, Q_all, i);
  return Q;
}

double eval_seam_junction_quality_after_tracing(
    const MedialMesh& mmesh, std::vector<Vector3>& bad_sphere_centers,
    std::vector<Vector3>& all_intf_sphere_centers) {
  // -1: init, 0: good, 1: bad
  std::vector<int> seam_junction_quality(mmesh.vertices->size(), -1);
  bad_sphere_centers.clear();
  all_intf_sphere_centers.clear();

  auto eval_junction_parallel = [&](const MedialStruct& mstruct) {
    if (mstruct.type != MedialType::JUNCTION) return;
    for (const auto& mvid : mstruct.m_sphere_ids) {
      const auto& msphere = mmesh.vertices->at(mvid);
      if (msphere.is_deleted) continue;
      // already evaluated
      if (seam_junction_quality[mvid] != -1) continue;
      seam_junction_quality[mvid] = 1;  // bad by default
      if (msphere.is_on_junction()) seam_junction_quality[mvid] = 0;
    }
  };

  auto eval_seam_parallel = [&](const MedialStruct& mstruct) {
    if (mstruct.type != MedialType::SEAM) return;
    for (const auto& meid : mstruct.m_edge_ids) {
      const auto& medge = mmesh.edges.at(meid);
      for (int vid : medge.vertices_) {
        const auto& msphere = mmesh.vertices->at(vid);
        if (msphere.is_deleted) continue;
        // already evaluated
        if (seam_junction_quality[vid] != -1) continue;
        seam_junction_quality[vid] = 1;  // bad by default
        // if (msphere.is_on_intf() && !msphere.is_on_junction())
        // no need to be seam only, junctions are fine,
        // as long as it's a sudden change
        if (msphere.is_on_intf()) seam_junction_quality[vid] = 0;
      }
    }
  };

  // eval junctions
  // NOTE: must check junctions frist
  // otherwise some junctions will be marked as bad in eval_seam_parallel()
  GEO::parallel_for(0, mmesh.mstructure.size(), [&](int mid) {
    eval_junction_parallel(mmesh.mstructure[mid]);
  });
  // eval seams
  GEO::parallel_for(0, mmesh.mstructure.size(), [&](int mid) {
    eval_seam_parallel(mmesh.mstructure[mid]);
  });
  // print results
  int num_bad = 0, num_seam_junction_total = 0;
  for (int vid = 0; vid < seam_junction_quality.size(); vid++) {
    if (seam_junction_quality[vid] == -1) continue;
    num_seam_junction_total++;
    all_intf_sphere_centers.push_back(mmesh.vertices->at(vid).center);
    if (seam_junction_quality[vid] == 1) {
      num_bad++;
      bad_sphere_centers.push_back(mmesh.vertices->at(vid).center);
      // printf("[seam_junction_quality] bad seam/junction sphere %d\n", vid);
    }
  }
  double bad_ratio = num_seam_junction_total == 0
                         ? 0.f
                         : 1.f * num_bad / num_seam_junction_total;
  printf("[seam_junction_quality] num_bad: %d/%d, total %d, bad ratio: %f\n",
         num_bad, num_seam_junction_total, seam_junction_quality.size(),
         bad_ratio);
  return bad_ratio;
}

#include "topKdist2mesh.h"
double eval_mat_intf_mspheres(
    const SurfaceMesh& sf_mesh, const MedialMesh& mmesh,
    std::vector<std::vector<int>>& sphere_top_tri_indices,
    std::vector<std::vector<float>>& sphere_top_tri_dists,
    std::vector<std::vector<afloat3>>& sphere_top_tri_tangents) {
  printf("calling eval_mat_intf_mspheres ...\n");

  // use findTopTriangleDistancesWithTangents() to find the top K=6 surface
  // triangles that are closest to each medial sphere center
  std::vector<afloat3> query_points;
  std::vector<afloat3> vertices;
  std::vector<aint3> triangles;
  for (int vid = 0; vid < sf_mesh.vertices.nb(); vid++) {
    const auto& p = sf_mesh.vertices.point(vid);
    vertices.push_back({p[0], p[1], p[2]});
  }
  for (int fid = 0; fid < sf_mesh.facets.nb(); fid++) {
    triangles.push_back({sf_mesh.facets.vertex(fid, 0),
                         sf_mesh.facets.vertex(fid, 1),
                         sf_mesh.facets.vertex(fid, 2)});
  }
  // load all mspheres centers
  // to make sure index in sphere_top_tri_indices is matching
  for (const auto& msphere : *mmesh.vertices) {
    query_points.push_back(
        {msphere.center[0], msphere.center[1], msphere.center[2]});
  }

  int K = 10;
  sphere_top_tri_indices.clear();
  sphere_top_tri_dists.clear();
  findTopTriangleDistancesWithTangents(
      query_points, vertices, triangles, K, sphere_top_tri_indices,
      sphere_top_tri_dists, sphere_top_tri_tangents);
}