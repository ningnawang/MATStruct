#include "knn.h"

#include <cstddef>
#include <type_traits>
#include <vector>

#include "nanoflann.hpp"

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------
// knnSearch():  Perform a search for the N closest points
// ----------------------------------------------------------------
void knn_search(const std::vector<Vector3>& points, const int num_neighbors,
                std::vector<std::vector<uint32_t>>& rets_index,
                std::vector<std::vector<double>>& rets_dist_sqr,
                bool is_debug) {
  using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, PointAdaptor<double>>,
      PointAdaptor<double>, 3 /*dim*/>;
  PointAdaptor<double> adapt;
  adapt.init_points(points);

  int num_points = points.size();
  rets_index.clear();
  rets_index.resize(num_points);
  rets_dist_sqr.clear();
  rets_dist_sqr.resize(num_points);

  // loop over all points
  auto run_thread = [&](int i) {
    std::array<double, 3> query_pt = {points[i][0], points[i][1], points[i][2]};
    rets_index[i].resize(num_neighbors);
    rets_dist_sqr[i].resize(num_neighbors);

    my_kd_tree_t index(3 /*dim*/, adapt, {10 /* max leaf */});
    index.knnSearch(&query_pt[0], num_neighbors, &rets_index[i][0],
                    &rets_dist_sqr[i][0]);

    // if (i == 585) {
    //   printf("query point %d: (%f,%f,%f) found %zu neighbors \n", i,
    //          query_pt[0], query_pt[1], query_pt[2],
    //          rets_index[i].size());
    //   print_vec<uint32_t>(rets_index[i]);
    //   print_vec<double>(rets_dist_sqr[i]);
    // }

    // In case of less points in the tree than requested:
    rets_index[i].resize(num_neighbors);
    rets_dist_sqr[i].resize(num_neighbors);
  };

  // for (int i = 0; i < num_points; i++) {
  //   run_thread(i);
  // }
  GEO::parallel_for(0, num_points, [&](int i) { run_thread(i); });
}

void knn_search_radius_sq(const std::vector<Vector3>& points,
                          const std::vector<bool>& is_point_on_extf,
                          const double search_radius_sq,
                          const double search_radius_se_sq,
                          std::vector<std::vector<uint32_t>>& rets_index,
                          std::vector<std::vector<double>>& rets_dist_sqr,
                          bool is_debug) {
  using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, PointAdaptor<double>>,
      PointAdaptor<double>, 3 /*dim*/>;
  PointAdaptor<double> adapt;
  adapt.init_points(points);

  int num_points = points.size();
  rets_index.clear();
  rets_index.resize(num_points);
  rets_dist_sqr.clear();
  rets_dist_sqr.resize(num_points);

  // the first element is a point index
  // the second the corresponding distance
  std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
  double search_radius_sq_tmp = -1;
  // loop over all points
  for (int i = 0; i < num_points; i++) {
    std::array<double, 3> query_pt = {points[i][0], points[i][1], points[i][2]};
    if (is_point_on_extf.at(i))
      search_radius_sq_tmp = search_radius_se_sq;
    else
      search_radius_sq_tmp = search_radius_sq;

    my_kd_tree_t index(3 /*dim*/, adapt, {10 /* max leaf */});
    ret_matches.clear();
    const size_t nMatches =
        index.radiusSearch(&query_pt[0], search_radius_sq_tmp, ret_matches);

    for (size_t j = 0; j < nMatches; j++) {
      rets_index[i].push_back(ret_matches[j].first);
      rets_dist_sqr[i].push_back(ret_matches[j].second);
    }

    if (is_debug) {
      printf(
          "query point %d: (%f,%f,%f) search_radius_sq_tmp: %f, found %zu "
          "neighbors\n",
          i, query_pt[0], query_pt[1], query_pt[2], search_radius_sq_tmp,
          rets_index[i].size());
      print_vec<uint32_t>(rets_index[i]);
      print_vec<double>(rets_dist_sqr[i]);
    }
  }
}

// try radius search first, if failed, use knn
void knn_search_radius_sq_or_k(const std::vector<Vector3>& points,
                               const std::vector<bool>& is_point_on_extf,
                               const double search_radius_sq,
                               const double search_radius_se_sq,
                               const int num_neighbors,
                               std::vector<std::vector<uint32_t>>& rets_index,
                               std::vector<std::vector<double>>& rets_dist_sqr,
                               bool is_debug) {
  using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, PointAdaptor<double>>,
      PointAdaptor<double>, 3 /*dim*/>;
  PointAdaptor<double> adapt;
  adapt.init_points(points);

  const int num_points = points.size();
  // int num_radius_poinsts = 0;
  rets_index.clear();
  rets_index.resize(num_points);
  rets_dist_sqr.clear();
  rets_dist_sqr.resize(num_points);

  // loop over all points
  auto run_thread = [&](int i) {
    std::array<double, 3> query_pt = {points[i][0], points[i][1], points[i][2]};
    double search_radius_sq_tmp = -1;
    if (is_point_on_extf.at(i))
      search_radius_sq_tmp = search_radius_se_sq;
    else
      search_radius_sq_tmp = search_radius_sq;

    bool is_thread_debug = false;
    // if (i % 100 == 0) is_thread_debug = true;

    my_kd_tree_t index(3 /*dim*/, adapt, {10 /* max leaf */});
    // the first element is a point index
    // the second the corresponding distance
    std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
    const size_t nMatches =
        index.radiusSearch(&query_pt[0], search_radius_sq_tmp, ret_matches);
    bool is_use_radius = nMatches > 2 ? true : false;

    if (is_use_radius) {  // contains itself
      for (size_t j = 0; j < nMatches; j++) {
        rets_index[i].push_back(ret_matches[j].first);
        rets_dist_sqr[i].push_back(ret_matches[j].second);
      }
      // num_radius_poinsts++;
    } else {
      rets_index[i].resize(num_neighbors);
      rets_dist_sqr[i].resize(num_neighbors);
      my_kd_tree_t index(3 /*dim*/, adapt, {10 /* max leaf */});
      index.knnSearch(&query_pt[0], num_neighbors, &rets_index[i][0],
                      &rets_dist_sqr[i][0]);
      // // In case of less points in the tree than requested:
      // rets_index[i].resize(num_neighbors);
      // rets_dist_sqr[i].resize(num_neighbors);
    }

    if (is_thread_debug) {
      if (is_use_radius)
        printf(
            "query point %d: (%f,%f,%f) search_radius_sq_tmp: %f, found "
            "%zu neighbors\n",
            i, query_pt[0], query_pt[1], query_pt[2], search_radius_sq_tmp,
            rets_index[i].size());
      else
        printf(
            "query point %d: (%f,%f,%f) num_neighbors: %d, found %zu "
            "neighbors\n",
            i, query_pt[0], query_pt[1], query_pt[2], num_neighbors,
            rets_index[i].size());
      print_vec<uint32_t>(rets_index[i]);
      print_vec<double>(rets_dist_sqr[i]);
    }
  };

  // for (int i = 0; i < num_points; i++) {
  //   run_thread(i);
  // }
  GEO::parallel_for(0, num_points, [&](int i) { run_thread(i); });

  // if (is_debug) {
  //   printf("query point %d/%d uses radius search, others use knn \n",
  //          num_radius_poinsts, num_points);
  // }
}