
#ifndef H_DIST2MESH_H
#define H_DIST2MESH_H

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <array>
#include <fstream>
#include <iostream>
#include <vector>

#include "common.h"
#include "cuda_helper_math.h"
#include "cuda_utils.h"

void findTopTriangleDistancesWithTangents(
    const std::vector<afloat3>& points, const std::vector<afloat3>& vertices,
    const std::vector<aint3>& triangles, int k,
    std::vector<std::vector<int>>& topFaceIds,
    std::vector<std::vector<float>>& topDistances,
    std::vector<std::vector<afloat3>>& topTangents);
#endif  // __DIST2MESH_H__