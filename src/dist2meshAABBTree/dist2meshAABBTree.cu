#include "common.h"
#include "dist2meshAABBTree.h"

// --------------------------------------------------------
// --- AABB Tree Construction (Geogram-style) ---
int max_node_index(int node_index, int b, int e) {
  assert(e > b);
  if (b + 1 == e) {
    return node_index;
  }
  int m = b + (e - b) / 2;
  int childl = 2 * node_index;
  int childr = 2 * node_index + 1;
  return std::max(max_node_index(childl, b, m), max_node_index(childr, m, e));
}

void compute_triangle_aabb(const std::vector<afloat3>& points,
                           const AABBTriangle& tri, AABB& box) {
  float3 v0 =
      make_float3(points[tri.v0][0], points[tri.v0][1], points[tri.v0][2]);
  float3 v1 =
      make_float3(points[tri.v1][0], points[tri.v1][1], points[tri.v1][2]);
  float3 v2 =
      make_float3(points[tri.v2][0], points[tri.v2][1], points[tri.v2][2]);

  box.min = fminf(fminf(v0, v1), v2);
  box.max = fmaxf(fmaxf(v0, v1), v2);
}

void init_bboxes_recursive(std::vector<AABBTreeNode>& tree,
                           const std::vector<afloat3>& points,
                           const std::vector<AABBTriangle>& triangles,
                           int node_index, int begin, int end) {
  if (end == begin + 1) {
    compute_triangle_aabb(points, triangles[begin], tree[node_index].box);
    return;
  }

  int mid = begin + (end - begin) / 2;
  int childl = 2 * node_index;
  int childr = 2 * node_index + 1;

  if (childr >= (int)tree.size()) tree.resize(childr + 1);

  init_bboxes_recursive(tree, points, triangles, childl, begin, mid);
  init_bboxes_recursive(tree, points, triangles, childr, mid, end);

  const AABB& bl = tree[childl].box;
  const AABB& br = tree[childr].box;
  tree[node_index].box.min = fminf(bl.min, br.min);
  tree[node_index].box.max = fmaxf(bl.max, br.max);
}

// --------------------------------------------------------
// --- CUDA Kernel: Ray Casting with BVH Traversal
__device__ bool intersect_ray_aabb(const float3& orig, const float3& dir,
                                   const AABB& box) {
  float tmin = -FLT_MAX, tmax = FLT_MAX;
  for (int i = 0; i < 3; ++i) {
    float invD = 1.0f / (i == 0 ? dir.x : (i == 1 ? dir.y : dir.z));
    float t0 = ((i == 0 ? box.min.x : (i == 1 ? box.min.y : box.min.z)) -
                (i == 0 ? orig.x : (i == 1 ? orig.y : orig.z))) *
               invD;
    float t1 = ((i == 0 ? box.max.x : (i == 1 ? box.max.y : box.max.z)) -
                (i == 0 ? orig.x : (i == 1 ? orig.y : orig.z))) *
               invD;
    if (invD < 0.0f) {
      float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    tmin = fmaxf(tmin, t0);
    tmax = fminf(tmax, t1);
    if (tmax < tmin) return false;
  }
  return true;
}

__device__ bool intersect_ray_triangle(const float3& orig, const float3& dir,
                                       const float3& v0, const float3& v1,
                                       const float3& v2, float& t) {
  const float EPSILON = 1e-6f;
  float3 e1 = v1 - v0;
  float3 e2 = v2 - v0;
  float3 h = cross(dir, e2);
  float a = dot(e1, h);
  if (fabsf(a) < EPSILON) return false;
  float f = 1.0f / a;
  float3 s = orig - v0;
  float u = f * dot(s, h);
  if (u < 0.0f || u > 1.0f) return false;
  float3 q = cross(s, e1);
  float v = f * dot(dir, q);
  if (v < 0.0f || u + v > 1.0f) return false;
  t = f * dot(e2, q);
  return t > EPSILON;
}

__device__ int ray_intersect_count(const float3& orig, const float3& dir,
                                   const AABBTreeNode* tree,
                                   const float3* vertices,
                                   const AABBTriangle* triangles, int root_idx,
                                   int b_root, int e_root) {
  int count = 0;
  int stack_size = 0;
  StackItem stack[MAX_STACK_SIZE];
  stack[stack_size++] = {root_idx, b_root, e_root};

  while (stack_size > 0) {
    StackItem item = stack[--stack_size];
    int node_idx = item.node_idx;
    int b = item.b;
    int e = item.e;
    const AABBTreeNode& node = tree[node_idx];

    if (!intersect_ray_aabb(orig, dir, node.box)) continue;

    if (e == b + 1) {
      const AABBTriangle& tri = triangles[b];
      float t;
      if (intersect_ray_triangle(orig, dir, vertices[tri.v0], vertices[tri.v1],
                                 vertices[tri.v2], t)) {
        ++count;
      }
      continue;
    }

    int mid = b + (e - b) / 2;
    int childl = 2 * node_idx;
    int childr = 2 * node_idx + 1;
    if (stack_size + 2 < MAX_STACK_SIZE) {
      stack[stack_size++] = {childl, b, mid};
      stack[stack_size++] = {childr, mid, e};
    }
  }

  return count;
}

__global__ void point_inside_mesh_kernel(const float3* queries, int n_queries,
                                         const float3* vertices,
                                         const AABBTriangle* triangles,
                                         int n_triangles,
                                         const AABBTreeNode* tree, int n_nodes,
                                         uint8_t* is_inside) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_queries) return;

  float3 ray_dir = make_float3(1.0f, 0.0f, 0.0f);  // e.g., X axis
  int count = ray_intersect_count(queries[idx], ray_dir, tree, vertices,
                                  triangles, 1, 0, n_triangles);
  is_inside[idx] = (count % 2 == 1);
}

// --------------------------------------------------------
// --- CUDA Kernel: Find Closest Point with BVH Traversal
__host__ __device__ float3 closest_point_on_triangle(const float3& p,
                                                     const float3& a,
                                                     const float3& b,
                                                     const float3& c, float& u,
                                                     float& v) {
  float3 ab = b - a;
  float3 ac = c - a;
  float3 ap = p - a;

  float d1 = dot(ab, ap);
  float d2 = dot(ac, ap);
  float d3 = dot(ab, ab);
  float d4 = dot(ab, ac);
  float d5 = dot(ac, ac);

  float denom = d3 * d5 - d4 * d4;
  u = (d1 * d5 - d2 * d4) / denom;
  v = (d2 * d3 - d1 * d4) / denom;

  u = fmaxf(0.0f, fminf(1.0f, u));      // lambda1
  v = fmaxf(0.0f, fminf(1.0f - u, v));  // lambda2

  return a + u * ab + v * ac;
}

__device__ float point_box_signed_squared_distance(const float3& p,
                                                   const AABB& box) {
  bool inside = true;
  float result = 0.0f;

  for (int c = 0; c < 3; ++c) {
    float pc = c == 0 ? p.x : (c == 1 ? p.y : p.z);
    float minc = c == 0 ? box.min.x : (c == 1 ? box.min.y : box.min.z);
    float maxc = c == 0 ? box.max.x : (c == 1 ? box.max.y : box.max.z);

    if (pc < minc) {
      inside = false;
      result += (pc - minc) * (pc - minc);
    } else if (pc > maxc) {
      inside = false;
      result += (pc - maxc) * (pc - maxc);
    }
  }

  if (inside) {
    // Compute minimal distance to one of the box sides (Geogram-style)
    float inner_dist2 = (p.x - box.min.x) * (p.x - box.min.x);
    inner_dist2 = fminf(inner_dist2, (p.x - box.max.x) * (p.x - box.max.x));
    inner_dist2 = fminf(inner_dist2, (p.y - box.min.y) * (p.y - box.min.y));
    inner_dist2 = fminf(inner_dist2, (p.y - box.max.y) * (p.y - box.max.y));
    inner_dist2 = fminf(inner_dist2, (p.z - box.min.z) * (p.z - box.min.z));
    inner_dist2 = fminf(inner_dist2, (p.z - box.max.z) * (p.z - box.max.z));
    result = -inner_dist2;
  }

  return result;
}

__device__ void traverse_AABBTree(const float3& p, const AABBTreeNode* tree,
                                  const float3* vertices,
                                  const AABBTriangle* triangles, int root_idx,
                                  int b_root, int e_root, ProjectedPoint& best,
                                  bool is_debug) {
  StackItem stack[MAX_STACK_SIZE];
  int stack_size = 0;
  stack[stack_size++] = {root_idx, b_root, e_root};

  while (stack_size > 0) {
    StackItem item = stack[--stack_size];
    int node_idx = item.node_idx;
    int b = item.b;
    int e = item.e;
    const AABBTreeNode& node = tree[node_idx];

    if (is_debug) {
      printf("Visiting Node %d [b=%d, e=%d]\n", node_idx, b, e);
      printf("  AABB min: (%.4f, %.4f, %.4f), max: (%.4f, %.4f, %.4f)\n",
             node.box.min.x, node.box.min.y, node.box.min.z, node.box.max.x,
             node.box.max.y, node.box.max.z);
    }

    if (e == b + 1) {
      const AABBTriangle& tri = triangles[b];
      float u, v;
      float3 proj = closest_point_on_triangle(
          p, vertices[tri.v0], vertices[tri.v1], vertices[tri.v2], u, v);
      float sq_dist = lengthSquared(p - proj);
      if (is_debug) {
        printf("  Leaf node for tri %d. sq_dist: %.6f\n", b, sq_dist);
      }
      if (sq_dist < best.sq_dist) {
        best.sq_dist = sq_dist;
        best.projected = proj;
        best.face_id = b;
        best.lambda_u = u;
        best.lambda_v = v;
        if (is_debug) {
          printf(
              "  -> Updated best result: proj(%.4f, %.4f, %.4f), u=%.4f, "
              "v=%.4f\n",
              proj.x, proj.y, proj.z, u, v);
        }
      }
      continue;
    }

    int mid = b + (e - b) / 2;
    int childl = 2 * node_idx;
    int childr = 2 * node_idx + 1;

    float dl = point_box_signed_squared_distance(p, tree[childl].box);
    float dr = point_box_signed_squared_distance(p, tree[childr].box);

    if (is_debug) {
      printf("  Children: L=%d [dl=%.6f], R=%d [dr=%.6f], best.sq_dist=%.6f\n",
             childl, dl, childr, dr, best.sq_dist);
    }

    // Push further child last (so it's visited after closer child)
    if (dl < dr) {
      if (dr < best.sq_dist && stack_size < MAX_STACK_SIZE)
        stack[stack_size++] = {childr, mid, e};
      if (dl < best.sq_dist && stack_size < MAX_STACK_SIZE)
        stack[stack_size++] = {childl, b, mid};
    } else {
      if (dl < best.sq_dist && stack_size < MAX_STACK_SIZE)
        stack[stack_size++] = {childl, b, mid};
      if (dr < best.sq_dist && stack_size < MAX_STACK_SIZE)
        stack[stack_size++] = {childr, mid, e};
    }
  }
}

__global__ void project_points_kernel(const float3* queries, int n_queries,
                                      const float3* vertices,
                                      const AABBTriangle* triangles,
                                      const int n_triangles,
                                      const AABBTreeNode* tree, int n_nodes,
                                      ProjectedPoint* results) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 0 || idx >= n_queries) return;

  bool is_debug = false;
  // if (idx == 0) is_debug = true;

  ProjectedPoint result;
  result.sq_dist = FLT_MAX;
  result.face_id = -1;

  traverse_AABBTree(queries[idx], tree, vertices, triangles, 1, 0, n_triangles,
                    result, is_debug);
  results[idx] = result;

  if (is_debug)
    printf("Query %d result: face_id=%d, sq_dist=%.6f\n", idx, result.face_id,
           result.sq_dist);
}

// --------------------------------------------------------
// public APIs
void build_AABBTree_CPU(const std::vector<afloat3>& points,
                        const std::vector<AABBTriangle>& triangles,
                        std::vector<AABBTreeNode>& tree) {
  const int num_triangles = triangles.size();
  if (num_triangles == 0) return;

  // Compute exact number of nodes needed
  int tree_size = max_node_index(1, 0, num_triangles) + 1;
  tree.clear();
  tree.resize(tree_size);

  init_bboxes_recursive(tree, points, triangles, 1, 0, num_triangles);
}

void proj_points_on_triangles_with_lambdas_cuda(
    const std::vector<afloat3>& query_points,
    const std::vector<afloat3>& vertices,
    const std::vector<AABBTriangle>& triangles,
    const std::vector<AABBTreeNode>& tree, std::vector<afloat3>& projPoints,
    std::vector<int>& projFaceIds, std::vector<float>& projLambda1,
    std::vector<float>& projLambda2) {
  assert(!query_points.empty() && !vertices.empty() && !triangles.empty() &&
         !tree.empty());

  std::vector<float3> verts_f3(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i)
    verts_f3[i] = make_float3(vertices[i][0], vertices[i][1], vertices[i][2]);

  std::vector<float3> queries_f3(query_points.size());
  for (size_t i = 0; i < query_points.size(); ++i)
    queries_f3[i] =
        make_float3(query_points[i][0], query_points[i][1], query_points[i][2]);

  float3 *d_vertices, *d_queries;
  AABBTriangle* d_tris;
  AABBTreeNode* d_tree;
  ProjectedPoint* d_results;

  cudaMalloc(&d_vertices, verts_f3.size() * sizeof(float3));
  cudaMalloc(&d_queries, queries_f3.size() * sizeof(float3));
  cudaMalloc(&d_tris, triangles.size() * sizeof(AABBTriangle));
  cudaMalloc(&d_tree, tree.size() * sizeof(AABBTreeNode));
  cudaMalloc(&d_results, queries_f3.size() * sizeof(ProjectedPoint));

  cudaMemcpy(d_vertices, verts_f3.data(), verts_f3.size() * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_queries, queries_f3.data(), queries_f3.size() * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tris, triangles.data(), triangles.size() * sizeof(AABBTriangle),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tree, tree.data(), tree.size() * sizeof(AABBTreeNode),
             cudaMemcpyHostToDevice);

  int threads = 128;
  int blocks = (queries_f3.size() + threads - 1) / threads;
  project_points_kernel<<<blocks, threads>>>(
      d_queries, queries_f3.size(), d_vertices, d_tris, triangles.size(),
      d_tree, tree.size(), d_results);
  cudaDeviceSynchronize();

  std::vector<ProjectedPoint> h_results(queries_f3.size());
  cudaMemcpy(h_results.data(), d_results,
             h_results.size() * sizeof(ProjectedPoint), cudaMemcpyDeviceToHost);

  for (const auto& r : h_results) {
    projPoints.push_back({r.projected.x, r.projected.y, r.projected.z});
    assert(r.face_id >= 0 && r.face_id < triangles.size());
    projFaceIds.push_back(r.face_id);
    projLambda1.push_back(r.lambda_u);
    projLambda2.push_back(r.lambda_v);
  }

  cudaFree(d_vertices);
  cudaFree(d_queries);
  cudaFree(d_tris);
  cudaFree(d_tree);
  cudaFree(d_results);
}

void point_inside_mesh_cuda(const std::vector<afloat3>& query_points,
                            const std::vector<afloat3>& vertices,
                            const std::vector<AABBTriangle>& triangles,
                            const std::vector<AABBTreeNode>& tree,
                            std::vector<bool>& is_inside_flags) {
  assert(!query_points.empty() && !vertices.empty() && !triangles.empty() &&
         !tree.empty());

  // Convert to float3
  std::vector<float3> queries_f3(query_points.size());
  for (size_t i = 0; i < query_points.size(); ++i)
    queries_f3[i] =
        make_float3(query_points[i][0], query_points[i][1], query_points[i][2]);

  std::vector<float3> verts_f3(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i)
    verts_f3[i] = make_float3(vertices[i][0], vertices[i][1], vertices[i][2]);

  // Device buffers
  float3* d_queries = nullptr;
  float3* d_vertices = nullptr;
  AABBTriangle* d_tris = nullptr;
  AABBTreeNode* d_tree = nullptr;
  uint8_t* d_flags = nullptr;

  const size_t n_queries = queries_f3.size();
  const size_t n_tris = triangles.size();
  const size_t n_nodes = tree.size();

  cudaMalloc(&d_queries, n_queries * sizeof(float3));
  cudaMalloc(&d_vertices, verts_f3.size() * sizeof(float3));
  cudaMalloc(&d_tris, n_tris * sizeof(AABBTriangle));
  cudaMalloc(&d_tree, n_nodes * sizeof(AABBTreeNode));
  cudaMalloc(&d_flags, n_queries * sizeof(uint8_t));

  // Copy to device
  cudaMemcpy(d_queries, queries_f3.data(), n_queries * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertices, verts_f3.data(), verts_f3.size() * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tris, triangles.data(), n_tris * sizeof(AABBTriangle),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tree, tree.data(), n_nodes * sizeof(AABBTreeNode),
             cudaMemcpyHostToDevice);

  // Launch kernel
  int threads = 128;
  int blocks = (n_queries + threads - 1) / threads;
  point_inside_mesh_kernel<<<blocks, threads>>>(d_queries, n_queries,
                                                d_vertices, d_tris, n_tris,
                                                d_tree, n_nodes, d_flags);

  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_flags(n_queries);
  cudaMemcpy(h_flags.data(), d_flags, n_queries * sizeof(bool),
             cudaMemcpyDeviceToHost);

  // Output
  is_inside_flags.resize(n_queries);
  for (size_t i = 0; i < n_queries; ++i) is_inside_flags[i] = h_flags[i];

  // Cleanup
  cudaFree(d_queries);
  cudaFree(d_vertices);
  cudaFree(d_tris);
  cudaFree(d_tree);
  cudaFree(d_flags);
}

void point_inside_and_project_cuda(const std::vector<afloat3>& query_points,
                                   const std::vector<afloat3>& vertices,
                                   const std::vector<AABBTriangle>& triangles,
                                   const std::vector<AABBTreeNode>& tree,
                                   std::vector<afloat3>& projPoints,
                                   std::vector<int>& projFaceIds,
                                   std::vector<float>& projLambda1,
                                   std::vector<float>& projLambda2,
                                   std::vector<bool>& is_inside) {
  assert(!query_points.empty() && !vertices.empty() && !triangles.empty() &&
         !tree.empty());

  int n_queries = query_points.size();
  int n_vertices = vertices.size();
  int n_triangles = triangles.size();
  int n_nodes = tree.size();

  // Convert to float3
  std::vector<float3> queries_f3(n_queries);
  for (int i = 0; i < n_queries; ++i)
    queries_f3[i] =
        make_float3(query_points[i][0], query_points[i][1], query_points[i][2]);

  std::vector<float3> vertices_f3(n_vertices);
  for (int i = 0; i < n_vertices; ++i)
    vertices_f3[i] =
        make_float3(vertices[i][0], vertices[i][1], vertices[i][2]);

  // Allocate device memory
  float3* d_queries;
  float3* d_vertices;
  AABBTriangle* d_triangles;
  AABBTreeNode* d_tree;
  ProjectedPoint* d_results;
  uint8_t* d_is_inside;

  cudaMalloc(&d_queries, sizeof(float3) * n_queries);
  cudaMalloc(&d_vertices, sizeof(float3) * n_vertices);
  cudaMalloc(&d_triangles, sizeof(AABBTriangle) * n_triangles);
  cudaMalloc(&d_tree, sizeof(AABBTreeNode) * n_nodes);
  cudaMalloc(&d_results, sizeof(ProjectedPoint) * n_queries);
  cudaMalloc(&d_is_inside, sizeof(uint8_t) * n_queries);

  // Copy data to device
  cudaMemcpy(d_queries, queries_f3.data(), sizeof(float3) * n_queries,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertices, vertices_f3.data(), sizeof(float3) * n_vertices,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_triangles, triangles.data(), sizeof(AABBTriangle) * n_triangles,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tree, tree.data(), sizeof(AABBTreeNode) * n_nodes,
             cudaMemcpyHostToDevice);

  // Launch kernels
  int blockSize = 256;
  int numBlocks = (n_queries + blockSize - 1) / blockSize;

  point_inside_mesh_kernel<<<numBlocks, blockSize>>>(
      d_queries, n_queries, d_vertices, d_triangles, n_triangles, d_tree,
      n_nodes, d_is_inside);

  project_points_kernel<<<numBlocks, blockSize>>>(
      d_queries, n_queries, d_vertices, d_triangles, n_triangles, d_tree,
      n_nodes, d_results);

  // Copy results back
  std::vector<ProjectedPoint> h_results(n_queries);
  std::vector<uint8_t> h_is_inside(n_queries);

  cudaMemcpy(h_results.data(), d_results, sizeof(ProjectedPoint) * n_queries,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_is_inside.data(), d_is_inside, sizeof(uint8_t) * n_queries,
             cudaMemcpyDeviceToHost);

  // Convert to std::vector<bool>
  is_inside.resize(n_queries);
  for (int i = 0; i < n_queries; ++i) is_inside[i] = h_is_inside[i] != 0;

  // Populate output vectors only for inside points
  projPoints.clear();
  projFaceIds.clear();
  projLambda1.clear();
  projLambda2.clear();

  for (int i = 0; i < n_queries; ++i) {
    const auto& r = h_results[i];
    projPoints.push_back({r.projected.x, r.projected.y, r.projected.z});
    projFaceIds.push_back(r.face_id);
    projLambda1.push_back(r.lambda_u);
    projLambda2.push_back(r.lambda_v);
  }

  // Free device memory
  cudaFree(d_queries);
  cudaFree(d_vertices);
  cudaFree(d_triangles);
  cudaFree(d_tree);
  cudaFree(d_results);
  cudaFree(d_is_inside);
}
