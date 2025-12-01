#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "common.h"
#include "cuda_helper_math.h"

constexpr int MAX_STACK_SIZE = 64;
struct StackItem {
  int node_idx;
  int b;
  int e;
};

struct AABB {
  float3 min, max;
};

struct AABBTreeNode {
  AABB box;
};

struct AABBTriangle {
  int v0, v1, v2;
};

struct ProjectedPoint {
  float3 projected;
  float sq_dist;
  int face_id;
  // lambda0 = 1 - lambda_u - lambda_v
  float lambda_u;  // lambda1
  float lambda_v;  // lambda2
};

void build_AABBTree_CPU(const std::vector<afloat3>& points,
                        const std::vector<AABBTriangle>& triangles,
                        std::vector<AABBTreeNode>& tree);

void proj_points_on_triangles_with_lambdas_cuda(
    const std::vector<afloat3>& query_points,
    const std::vector<afloat3>& vertices,
    const std::vector<AABBTriangle>& triangles,
    const std::vector<AABBTreeNode>& tree, std::vector<afloat3>& projPoints,
    std::vector<int>& projFaceIds, std::vector<float>& projLambda1,
    std::vector<float>& projLambda2);

void point_inside_mesh_cuda(const std::vector<afloat3>& query_points,
                            const std::vector<afloat3>& vertices,
                            const std::vector<AABBTriangle>& triangles,
                            const std::vector<AABBTreeNode>& tree,
                            std::vector<bool>& is_inside_flags);

void point_inside_and_project_cuda(const std::vector<afloat3>& query_points,
                                   const std::vector<afloat3>& vertices,
                                   const std::vector<AABBTriangle>& triangles,
                                   const std::vector<AABBTreeNode>& tree,
                                   std::vector<afloat3>& projPoints,
                                   std::vector<int>& projFaceIds,
                                   std::vector<float>& projLambda1,
                                   std::vector<float>& projLambda2,
                                   std::vector<bool>& is_inside);