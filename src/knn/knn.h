#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "common_geogram.h"

// And this is the "dataset to kd-tree" Adaptor class:
template <typename T>
struct PointAdaptor {
  struct Point {
    T x, y, z;
  };
  std::vector<Point> pts;

  inline void init_points(const std::vector<Vector3>& samples) {
    pts.resize(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
      pts[i].x = samples[i][0];
      pts[i].y = samples[i][1];
      pts[i].z = samples[i][2];
    }
  }

  inline void clear() { pts.clear(); }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return pts[idx].x;
    else if (dim == 1)
      return pts[idx].y;
    else
      return pts[idx].z;
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned
  //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
  //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }

  auto const* elem_ptr(const unsigned idx) const { return &pts[idx].x; }
};

//////////////////////////////////////////////////////////////////////
void knn_search(const std::vector<Vector3>& points, const int num_neighbors,
                std::vector<std::vector<uint32_t>>& rets_index,
                std::vector<std::vector<double>>& rets_dist_sqr, bool is_debug);

void knn_search_radius_sq(const std::vector<Vector3>& points,
                          const std::vector<bool>& is_point_on_extf,
                          const double search_radius_sq,
                          const double search_radius_se_sq,
                          std::vector<std::vector<uint32_t>>& rets_index,
                          std::vector<std::vector<double>>& rets_dist_sqr,
                          bool is_debug);

void knn_search_radius_sq_or_k(const std::vector<Vector3>& points,
                               const std::vector<bool>& is_point_on_extf,
                               const double search_radius_sq,
                               const double search_radius_se_sq,
                               const int num_neighbors,
                               std::vector<std::vector<uint32_t>>& rets_index,
                               std::vector<std::vector<double>>& rets_dist_sqr,
                               bool is_debug);