#pragma once

#include <Eigen/Dense>

#include "rpd3d_wrapper.h"

#define FEATURE_PARTICLE_TANGENT_WITH_SPHERE_TYPE IN_USE

enum ParticleProjectionType {
  FAIL = -1,   // T1 sphere
  FIXED = 0,   // T4 sphere, do not move, contains corner/T4 spheres
  PLANE = 1,   // T2 spheres, return normal of the solution plane
  LINE = 2,    // T3 spheres, return tangent of the solution line
  SE_LINE = 3  // T1_2 spheres, return SE tangent direction
};

struct ParticleProjection {
  ParticleProjectionType proj_type = ParticleProjectionType::FAIL;
  // plane(normal) or line(tangent) direction
  Vector3 dir = Vector3(0.f, 0.f, 0.f);
  // one pseudo sphere center on plane or line
  Vector3 pseudo_center = Vector3(0.f, 0.f, 0.f);
  double pseudo_radius = -1;
  // for computing the pseudo radius, depends on the pseudo sphere
  Vector4 A_row0 = Vector4(0.f, 0.f, 0.f, 0.f);
  double b_0 = 0.f;
};

void get_particle_projection_gradient_or_tangent(
    const MedialSphere& msphere, const std::vector<FeatureEdge>& feature_edges,
    const RPD3D_Wrapper& rpd3d, ParticleProjection& particle_proj,
    bool is_debug);

void project_particle_gradient(const ParticleProjection& particle_proj,
                               Vector3& grad, bool is_debug);
void project_particle_sphere(const ParticleProjection& particle_proj,
                             Vector4& mi, bool is_debug);

// for debug
void get_particle_tangent_or_normal_all_spheres(
    const std::vector<FeatureEdge>& feature_edges, const RPD3D_Wrapper& rpd3d,
    std::vector<ParticleProjection>& all_particle_projs,
    std::vector<Vector4>& all_pseudo_sphere_projs, bool is_debug);