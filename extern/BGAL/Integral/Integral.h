#pragma once
#include <Eigen/Dense>

#include "Tetrahedron_arbq_rule.h"
#include "common_geogram.h"
namespace BGAL {

typedef std::array<double, 3> adouble3;

class Integral {
 public:
  template <class F>
  static Eigen::VectorXd integral_triangle3D(F f, const Eigen::Vector3d& p1,
                                             const Eigen::Vector3d& p2,
                                             const Eigen::Vector3d& p3);
  template <class F>
  static void get_triangle_integral_samples(
      const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
      const Eigen::Vector3d& p3, std::vector<Eigen::Vector3d>& samples,
      std::vector<double>& weights);

  template <class F>
  static Eigen::VectorXd integral_tetrahedron(F f, const Eigen::Vector3d& p1,
                                              const Eigen::Vector3d& p2,
                                              const Eigen::Vector3d& p3,
                                              const Eigen::Vector3d& p4);
  template <class F>
  static void get_tetra_integral_samples(const adouble3& p1, const adouble3& p2,
                                         const adouble3& p3, const adouble3& p4,
                                         std::vector<adouble3>& samples,
                                         std::vector<double>& weights);
};

template <class F>
Eigen::VectorXd Integral::integral_triangle3D(F f, const Eigen::Vector3d& p1,
                                              const Eigen::Vector3d& p2,
                                              const Eigen::Vector3d& p3) {
  Eigen::VectorXd r1 = 1.0 / 30 * f(p2 * 0.5 + p3 * 0.5);
  Eigen::VectorXd r2 = 1.0 / 30 * f(p1 * 0.5 + p2 * 0.5);
  Eigen::VectorXd r3 = 1.0 / 30 * f(p1 * 0.5 + p3 * 0.5);
  Eigen::VectorXd r4 = 9.0 / 30 * f(p1 / 6.0 + p2 / 6.0 + p3 * 2.0 / 3.0);
  Eigen::VectorXd r5 = 9.0 / 30 * f(p3 / 6.0 + p1 / 6.0 + p2 * 2.0 / 3.0);
  Eigen::VectorXd r6 = 9.0 / 30 * f(p2 / 6.0 + p3 / 6.0 + p1 * 2.0 / 3.0);
  Eigen::Vector3d v1 = p2 - p1;
  Eigen::Vector3d v2 = p3 - p1;
  double area = v1.cross(v2).norm() * 0.5;
  return area * (r1 + r2 + r3 + r4 + r5 + r6);
}

template <class F>
void Integral::get_triangle_integral_samples(
    const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3, std::vector<Eigen::Vector3d>& samples,
    std::vector<double>& weights) {
  samples.clear();
  weights.clear();

  samples.push_back(p2 * 0.5 + p3 * 0.5);
  samples.push_back(p1 * 0.5 + p2 * 0.5);
  samples.push_back(p1 * 0.5 + p3 * 0.5);
  samples.push_back(p1 / 6.0 + p2 / 6.0 + p3 * 2.0 / 3.0);
  samples.push_back(p3 / 6.0 + p1 / 6.0 + p2 * 2.0 / 3.0);
  samples.push_back(p2 / 6.0 + p3 / 6.0 + p1 * 2.0 / 3.0);

  Eigen::Vector3d v1 = p2 - p1;
  Eigen::Vector3d v2 = p3 - p1;
  double area = v1.cross(v2).norm() * 0.5;

  weights.push_back(area * 1.0 / 30);
  weights.push_back(area * 1.0 / 30);
  weights.push_back(area * 1.0 / 30);
  weights.push_back(area * 9.0 / 30);
  weights.push_back(area * 9.0 / 30);
  weights.push_back(area * 9.0 / 30);
  assert(samples.size() == weights.size());
}

template <class F>
Eigen::VectorXd Integral::integral_tetrahedron(F f, const Eigen::Vector3d& p1,
                                               const Eigen::Vector3d& p2,
                                               const Eigen::Vector3d& p3,
                                               const Eigen::Vector3d& p4) {
  // printf("------[integral_tetrahedron]: \n");
  // printf("p1 (%f,%f,%f)\n", p1(0), p1(1), p1(2));
  // printf("p2 (%f,%f,%f)\n", p2(0), p2(1), p2(2));
  // printf("p3 (%f,%f,%f)\n", p3(0), p3(1), p3(2));
  // printf("p4 (%f,%f,%f)\n", p4(0), p4(1), p4(2));

  double node_xyz2[12];
  node_xyz2[0] = p1(0);
  node_xyz2[1] = p1(1);
  node_xyz2[2] = p1(2);
  node_xyz2[3] = p2(0);
  node_xyz2[4] = p2(1);
  node_xyz2[5] = p2(2);
  node_xyz2[6] = p3(0);
  node_xyz2[7] = p3(1);
  node_xyz2[8] = p3(2);
  node_xyz2[9] = p4(0);
  node_xyz2[10] = p4(1);
  node_xyz2[11] = p4(2);

  double* w;
  double* xyz;
  double* xyz2;
  int order_num = keast_order_num(4);
  xyz = new double[3 * order_num];
  xyz2 = new double[3 * order_num];
  w = new double[order_num];

  keast_rule(4, order_num, xyz, w);
  tetrahedron_reference_to_physical(node_xyz2, order_num, xyz, xyz2);
  std::vector<Eigen::Vector3d> samples;
  for (int i = 0; i < order_num; ++i) {
    samples.push_back(
        Eigen::Vector3d(xyz2[0 + i * 3], xyz2[1 + i * 3], xyz2[2 + i * 3]));
  }
  // printf("sample 0: (%f,%f,%f)\n", samples[0](0), samples[0](1),
  // samples[0](2));

  double volume = tetrahedron_volume(node_xyz2);
  Eigen::VectorXd r = volume * w[0] * f(samples[0]);
  // Eigen::VectorXd r = f(samples[0]);
  for (int i = 1; i < order_num; ++i) {
    // printf("sample %d: (%f,%f,%f)\n", i, samples[i](0), samples[i](1),
    //  samples[i](2));
    r = r + volume * w[i] * f(samples[i]);
    // r = r + f(samples[i]);
  }

  delete[] w;
  delete[] xyz;
  delete[] xyz2;

  return r;
}

template <class F>
void Integral::get_tetra_integral_samples(const adouble3& p1,
                                          const adouble3& p2,
                                          const adouble3& p3,
                                          const adouble3& p4,
                                          std::vector<adouble3>& samples,
                                          std::vector<double>& weights) {
  samples.clear();
  weights.clear();

  double node_xyz2[12];
  node_xyz2[0] = p1[0];
  node_xyz2[1] = p1[1];
  node_xyz2[2] = p1[2];
  node_xyz2[3] = p2[0];
  node_xyz2[4] = p2[1];
  node_xyz2[5] = p2[2];
  node_xyz2[6] = p3[0];
  node_xyz2[7] = p3[1];
  node_xyz2[8] = p3[2];
  node_xyz2[9] = p4[0];
  node_xyz2[10] = p4[1];
  node_xyz2[11] = p4[2];

  int order_num = keast_order_num(4);
  double* w = new double[order_num];
  double* xyz = new double[3 * order_num];
  double* xyz2 = new double[3 * order_num];

  keast_rule(4, order_num, xyz, w);
  tetrahedron_reference_to_physical(node_xyz2, order_num, xyz, xyz2);

  double volume = tetrahedron_volume(node_xyz2);
  for (int i = 0; i < order_num; ++i) {
    samples.push_back({{xyz2[0 + i * 3], xyz2[1 + i * 3], xyz2[2 + i * 3]}});
    weights.push_back(volume * w[i]);
  }

  delete[] w;
  delete[] xyz;
  delete[] xyz2;
}

}  // namespace BGAL