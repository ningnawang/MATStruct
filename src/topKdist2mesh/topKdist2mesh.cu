#include <cuda_runtime.h>
#include <float.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include "common_cuda.h"
#include "topKdist2mesh.h"

// Compute the squared distance and tangent point to a triangle
__device__ float squaredDistanceToTriangleWithTangent(const float3& p,
                                                      const float3& v1,
                                                      const float3& v2,
                                                      const float3& v3,
                                                      float3& tangentPoint) {
  float3 edge0 = v2 - v1;
  float3 edge1 = v3 - v1;
  float3 v0 = v1 - p;

  // Compute dot products
  float a = dot(edge0, edge0);
  float b = dot(edge0, edge1);
  float c = dot(edge1, edge1);
  float d = dot(edge0, v0);
  float e = dot(edge1, v0);
  float det = a * c - b * b;

  // Compute barycentric coordinates
  float s = b * e - c * d;
  float t = b * d - a * e;

  // Check if the point is inside the triangle
  if (s + t <= det && s >= 0 && t >= 0) {
    float invDet = 1.0f / det;
    s *= invDet;
    t *= invDet;
    tangentPoint = v1 + edge0 * s + edge1 * t;
    return lengthSquared(tangentPoint - p);
  }

  // Outside triangle, project onto edges
  auto edgeDist = [&](const float3& vA, const float3& vB, float3& projection) {
    float3 edge = vB - vA;
    float t = max(0.0f, min(1.0f, dot(p - vA, edge) / lengthSquared(edge)));
    projection = vA + edge * t;
    return lengthSquared(projection - p);
  };

  float3 proj1, proj2, proj3;
  float d1 = edgeDist(v1, v2, proj1);
  float d2 = edgeDist(v2, v3, proj2);
  float d3 = edgeDist(v3, v1, proj3);

  if (d1 < d2 && d1 < d3) {
    tangentPoint = proj1;
    return d1;
  } else if (d2 < d3) {
    tangentPoint = proj2;
    return d2;
  } else {
    tangentPoint = proj3;
    return d3;
  }
}

// CUDA kernel to compute distances, face IDs, and tangent points
__global__ void computeTopDistancesWithTangents(
    const float3* points, int numPoints, const float3* vertices,
    const int3* triangles, int numTriangles, const int topK, int* topFaceIDs,
    float* topDistances, float3* topTangents) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPoints) return;
  float3 queryPoint = points[idx];

  float* localTopSqDistances = new float[topK];
  int* localTopFaceIDs = new int[topK];
  float3* localTopTangents = new float3[topK];
  for (int i = 0; i < topK; ++i) {
    localTopSqDistances[i] = FLT_MAX;
    localTopFaceIDs[i] = -1;
    localTopTangents[i] = make_float3(0, 0, 0);
  }

  for (int i = 0; i < numTriangles; ++i) {
    const auto& tri = triangles[i];
    const float3& v1 = vertices[tri.x];
    const float3& v2 = vertices[tri.y];
    const float3& v3 = vertices[tri.z];

    float3 tangentPoint;
    float sq_dist = squaredDistanceToTriangleWithTangent(queryPoint, v1, v2, v3,
                                                         tangentPoint);

    // Insert into the local top distances, face IDs, and tangents
    for (int j = 0; j < topK; ++j) {
      if (sq_dist < localTopSqDistances[j]) {
        for (int k = topK - 1; k > j; --k) {
          localTopSqDistances[k] = localTopSqDistances[k - 1];
          localTopFaceIDs[k] = localTopFaceIDs[k - 1];
          localTopTangents[k] = localTopTangents[k - 1];
        }
        localTopSqDistances[j] = sq_dist;
        localTopFaceIDs[j] = i;
        localTopTangents[j] = tangentPoint;
        break;
      }
    }
  }

  // Write results back to global memory
  for (int j = 0; j < topK; ++j) {
    topDistances[idx * topK + j] = sqrtf(localTopSqDistances[j]);
    topFaceIDs[idx * topK + j] = localTopFaceIDs[j];
    topTangents[idx * topK + j] = localTopTangents[j];
  }

  delete[] localTopSqDistances;
  delete[] localTopFaceIDs;
  delete[] localTopTangents;
}

// Host function to invoke the kernel
void findTopTriangleDistancesWithTangents(
    const std::vector<afloat3>& points, const std::vector<afloat3>& vertices,
    const std::vector<aint3>& triangles, int k,
    std::vector<std::vector<int>>& topFaceIds,
    std::vector<std::vector<float>>& topDistances,
    std::vector<std::vector<afloat3>>& topTangents) {
  int numPoints = points.size();
  int numVertices = vertices.size();
  int numTriangles = triangles.size();

  // Convert input to CUDA-compatible arrays
  std::vector<float3> pointsCuda(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    pointsCuda[i] = make_float3(points[i][0], points[i][1], points[i][2]);
  }

  std::vector<float3> verticesCuda(numVertices);
  for (size_t i = 0; i < numVertices; ++i) {
    verticesCuda[i] =
        make_float3(vertices[i][0], vertices[i][1], vertices[i][2]);
  }

  std::vector<int3> trianglesCuda(numTriangles);
  for (int i = 0; i < numTriangles; ++i) {
    trianglesCuda[i] =
        make_int3(triangles[i][0], triangles[i][1], triangles[i][2]);
  }

  // Allocate device memory
  float3* d_points;
  float3* d_vertices;
  int3* d_triangles;
  int* d_topFaceIds;
  float* d_topDistances;
  float3* d_topTangents;

  cudaMalloc(&d_points, numPoints * sizeof(float3));
  cudaMalloc(&d_vertices, numVertices * sizeof(float3));
  cudaMalloc(&d_triangles, numTriangles * sizeof(int3));
  cudaMalloc(&d_topFaceIds, numPoints * k * sizeof(int));
  cudaMalloc(&d_topDistances, numPoints * k * sizeof(float));
  cudaMalloc(&d_topTangents, numPoints * k * sizeof(float3));

  cudaMemcpy(d_points, pointsCuda.data(), numPoints * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertices, verticesCuda.data(), numVertices * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_triangles, trianglesCuda.data(), numTriangles * sizeof(int3),
             cudaMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
  computeTopDistancesWithTangents<<<blocksPerGrid, threadsPerBlock>>>(
      d_points, numPoints, d_vertices, d_triangles, numTriangles, k,
      d_topFaceIds, d_topDistances, d_topTangents);

  // Copy results back to host
  std::vector<int> topFaceIdsFlat(numPoints * k);
  std::vector<float> topDistancesFlat(numPoints * k);
  std::vector<float3> topTangentsFlat(numPoints * k);
  cudaMemcpy(topFaceIdsFlat.data(), d_topFaceIds, numPoints * k * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(topDistancesFlat.data(), d_topDistances,
             numPoints * k * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(topTangentsFlat.data(), d_topTangents,
             numPoints * k * sizeof(float3), cudaMemcpyDeviceToHost);

  // Reshape flat arrays into 2D vectors
  topFaceIds.resize(numPoints, std::vector<int>(k));
  topDistances.resize(numPoints, std::vector<float>(k));
  topTangents.resize(numPoints, std::vector<afloat3>(k));
  for (int i = 0; i < numPoints; ++i) {
    for (int j = 0; j < k; ++j) {
      topFaceIds[i][j] = topFaceIdsFlat[i * k + j];
      topDistances[i][j] = topDistancesFlat[i * k + j];
      topTangents[i][j] = {topTangentsFlat[i * k + j].x,
                           topTangentsFlat[i * k + j].y,
                           topTangentsFlat[i * k + j].z};
    }
  }

  // Free device memory
  cudaFree(d_points);
  cudaFree(d_vertices);
  cudaFree(d_triangles);
  cudaFree(d_topFaceIds);
  cudaFree(d_topDistances);
  cudaFree(d_topTangents);
}
