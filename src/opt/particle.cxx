#include "particle.h"

#include <Eigen/Core>
#include <Eigen/SVD>

// for sphere on SE, then use the SE direction as the tangent
void get_particle_tangent_SE(const FeatureEdge& feature_edge,
                             ParticleProjection& particle_proj, bool is_debug) {
  particle_proj.dir =
      GEO::normalize(feature_edge.t2vs_pos[0] - feature_edge.t2vs_pos[1]);
  particle_proj.proj_type = ParticleProjectionType::SE_LINE;

  // NO need to update this
  // particle_proj.pseudo_center =
  //     1. / 2 * (feature_edge.t2vs_pos[0] + feature_edge.t2vs_pos[1]);
  // particle_proj.pseudo_radius = SCALAR_FEATURE_RADIUS;
}

Eigen::VectorXd get_linear_equation_b(const std::vector<Vector3>& normals,
                                      const std::vector<Vector3>& points) {
  assert(normals.size() == points.size());
  int num_points = normals.size();  // = j

  // get matrix N, jx4
  Eigen::MatrixXd N(num_points, 4);
  for (int n = 0; n < num_points; n++) {
    N.row(n) = Eigen::Vector4d(normals[n][0], normals[n][1], normals[n][2], 1);
  }

  // get Vector P, jx1
  Eigen::VectorXd P(num_points);
  for (int n = 0; n < num_points; n++) {
    P[n] = GEO::dot(points[n], normals[n]);
  }

  // get b = N^T * P, 4x1
  Eigen::VectorXd b(4);
  b = N.transpose() * P;
  return b;
}

// return:
// -1:  fail, return nothing
// 1:   T2 sphere, return normal,
// 2:   T3 sphere, return tangent direction
void get_particle_tangent_or_normal(const RPD3D_Wrapper& rpd3d,
                                    const MedialSphere& msphere,
                                    ParticleProjection& particle_proj,
                                    bool is_debug) {
  // if (msphere.id == 23) is_debug = true;
  // init particle_proj
  particle_proj.proj_type = ParticleProjectionType::FAIL;
  particle_proj.pseudo_center = msphere.center;
  particle_proj.pseudo_radius = msphere.radius;

  if (is_debug)
    printf("[ParticleTangent] calling get particle tangent or normal ...\n");
  const int sphere_id = msphere.id;
  auto normals = rpd3d.get_sphere_pcell_samples_attribute<Vector3>(
      sphere_id, PCELL_SAMPLE::ATTR::PLANE);
  if (normals.empty()) {
    if (is_debug)
      printf("[ParticleTangent] sphere %d normals empty, return\n", sphere_id);
    particle_proj.proj_type = ParticleProjectionType::FAIL;
    return;
  };
  auto sf_projs = rpd3d.get_sphere_pcell_samples_attribute<Vector3>(
      sphere_id, PCELL_SAMPLE::ATTR::PROJ);  // surface projections

  // compute SVD and rank
  Eigen::MatrixXd N(normals.size(), 4);
  for (int n = 0; n < normals.size(); n++) {
    N.row(n) = Eigen::Vector4d(normals[n][0], normals[n][1], normals[n][2], 1);
  }
  Eigen::Matrix4d A = N.transpose() * N;  /// 4x4
  // SVD, 4x4 matrix
  Eigen::JacobiSVD<Eigen::Matrix4d> svd = Eigen::JacobiSVD<Eigen::Matrix4d>(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  // determine which singular values should be considered nonzero.
  svd.setThreshold(SCALAR_ZERO_2);
  int num_rank = svd.rank();
  if (num_rank == 1) {
    if (is_debug)
      printf("[ParticleTangent] sphere %d SVD rank is 1, dunno what to do\n");
    particle_proj.proj_type = ParticleProjectionType::FAIL;
    return;
  }

#if FEATURE_PARTICLE_TANGENT_WITH_SPHERE_TYPE == IN_USE
  if ((msphere.type == SphereType::T_2 || msphere.type == SphereType::T_2_c) &&
      num_rank > 2) {
    if (is_debug)
      printf("[ParticleTangent] sphere %d type %d update rank %d->%d\n",
             sphere_id, msphere.type, num_rank, 2);
    num_rank = 2;
  }
  // this happens when T3 sphere stuck at T4 and never move during particle
  if ((msphere.type == SphereType::T_3_MORE ||
       msphere.type == SphereType::T_N_c) &&
      (num_rank > 3 || num_rank == 2)) {
    // yes, num_rank == 2 may happens
    // for model 0017061_partstudio_15_model_ste_00_2048
    // should be T3 but rank=2
    if (is_debug)
      printf("[ParticleTangent] sphere %d type %d update rank %d->%d\n",
             sphere_id, msphere.type, num_rank, 3);
    num_rank = 3;
  }
#endif

  assert(num_rank == 2 || num_rank == 3 || num_rank == 4);
  // get U, S, V matrices
  Eigen::MatrixXd U = svd.matrixU();         // 4x4
  Eigen::VectorXd S = svd.singularValues();  // singular values in a vector
  Eigen::MatrixXd V = svd.matrixV();         // 4x4
  Eigen::Matrix4d V_trans = V.transpose();
  if (is_debug) {
    printf("[ParticleTangent] num_rank %d \n", num_rank);
    std::cout << "V_trans:" << std::endl << V_trans << std::endl;
    std::cout << "S:" << std::endl << S << std::endl;
  }
  // build S_inv (inverse of S)
  Eigen::MatrixXd S_inv(4, 4);
  S_inv.setZero();
  for (int i = 0; i < S.size(); ++i) {
    // if (i < num_rank && S(i) > SCALAR_ZERO_6) {  // avoid division by zero
    if (S(i) > SCALAR_ZERO_6) {  // avoid division by zero
      S_inv(i, i) = 1.0 / S(i);
    }
  }
  if (is_debug) {
    std::cout << "S_inv:" << std::endl << S_inv << std::endl;
  }
  // compute pseudo-inverse: A_pinv = V * S_inv * U^T
  Eigen::MatrixXd A_pinv = V * S_inv * U.transpose();
  Eigen::VectorXd b = get_linear_equation_b(normals, sf_projs);
  if (is_debug) {
    std::cout << "b:" << std::endl << b << std::endl;
    std::cout << "A_pinv:" << std::endl << A_pinv << std::endl;
  }
  // compute peudo sphere
  Eigen::VectorXd mi_tmp = A_pinv * b;
  assert(mi_tmp.size() == 4);
  particle_proj.pseudo_center = Vector3(mi_tmp[0], mi_tmp[1], mi_tmp[2]);
  particle_proj.pseudo_radius = mi_tmp[3];
  particle_proj.A_row0 = Vector4(A(0, 0), A(0, 1), A(0, 2), A(0, 3));
  particle_proj.b_0 = b[0];
  //
  if (num_rank == 4) {
    if (is_debug)
      printf(
          "[ParticleTangent] sphere %d SVD rank is 4, Zero tangent, no move\n",
          sphere_id);
    particle_proj.proj_type = ParticleProjectionType::FIXED;
    return;
  }
  if (num_rank == 2) {
    // compute T2 normal
    //
    // since rank(N^TN)=2, there are two zero singular values,
    // last two columns in V spans the null space of A.
    int c = 2;  // one before last row index
    int d = 3;  // last row index
    if (is_debug)
      printf("[ParticleTangent] processing normal, c: %d, d: %d\n", c, d);
    Vector3 n1(V_trans(c, 0), V_trans(c, 1), V_trans(c, 2));
    Vector3 n2(V_trans(d, 0), V_trans(d, 1), V_trans(d, 2));
    particle_proj.dir = GEO::cross(n1, n2);
    if (is_debug) {
      printf("[ParticleTangent] sphere %d normal no normalize (%f,%f,%f)\n",
             sphere_id, particle_proj.dir[0], particle_proj.dir[1],
             particle_proj.dir[2]);
    }
    particle_proj.dir = GEO::normalize(particle_proj.dir);
    particle_proj.proj_type = ParticleProjectionType::PLANE;
    return;
  }
  if (num_rank == 3) {
    // T3: tangent of the edge
    //
    // since rank(N^TN)=3, there is one zero singular value, and the
    // corresponding column in V spans the null space of A. This column is the
    // tangent direction of the solution line. So the last row in V_trans is the
    // tangent direction.
    int d = 3;  // last row index
    if (is_debug) printf("[ParticleTangent] processing tangent, d: %d\n", d);
    particle_proj.dir = Vector3(V_trans(d, 0), V_trans(d, 1), V_trans(d, 2));
    if (GEO::length(particle_proj.dir) < SCALAR_ZERO_6) {
      if (is_debug)
        printf(
            "[ParticleTangent] rank 3, sphere %d norm 0, dunno what to do\n");
      particle_proj.proj_type = ParticleProjectionType::FAIL;
      return;
    }
    particle_proj.dir = GEO::normalize(particle_proj.dir);
    particle_proj.proj_type = ParticleProjectionType::LINE;
    return;
  }
  assert(false);
}

void get_particle_tangent_or_normal_wrapper(
    const MedialSphere& msphere, const std::vector<FeatureEdge>& feature_edges,
    const RPD3D_Wrapper& rpd3d, ParticleProjection& particle_proj,
    bool is_debug) {
  // init pseudo_center
  particle_proj.pseudo_center = msphere.center;
  particle_proj.pseudo_radius = msphere.radius;
  if (msphere.is_on_corner()) {
    if (is_debug)
      printf("[ParticleTangent] corner sphere %d, do not move\n", msphere.id);
    particle_proj.dir = Vector3(0, 0, 0);
    particle_proj.proj_type = ParticleProjectionType::FIXED;
  } else if (msphere.is_on_se()) {
    const int se_id = msphere.se_edge_id;
    if (se_id < 0 || se_id >= feature_edges.size()) {
      msphere.print_info();
      printf("[ParticleTangent] sphere %d, se_id %d out of range (0,%d)\n",
             msphere.id, se_id, feature_edges.size());
    }
    assert(se_id > -1 && se_id < feature_edges.size());
    get_particle_tangent_SE(feature_edges.at(se_id), particle_proj, false
                            /*is_debug*/);
  } else {
    get_particle_tangent_or_normal(rpd3d, msphere, particle_proj, is_debug);
  }
  return;
}

////////////////////////////////////////////////////////////////////////////////////////////
// public functions
void get_particle_projection_gradient_or_tangent(
    const MedialSphere& msphere, const std::vector<FeatureEdge>& feature_edges,
    const RPD3D_Wrapper& rpd3d, ParticleProjection& particle_proj,
    bool is_debug) {
  get_particle_tangent_or_normal_wrapper(msphere, feature_edges, rpd3d,
                                         particle_proj, is_debug);
  if (particle_proj.proj_type == ParticleProjectionType::FAIL ||
      particle_proj.proj_type == ParticleProjectionType::FIXED) {
    particle_proj.dir = Vector3(0, 0, 0);
  }
  double len = GEO::length(particle_proj.dir);
  double diff = std::abs(1.f - len);
  if (particle_proj.proj_type == ParticleProjectionType::FAIL ||
      particle_proj.proj_type == ParticleProjectionType::FIXED) {
    assert(len == 0.f);
  } else if (diff > SCALAR_ZERO_6) {
    printf(
        "[ParticleTangent] msphere %d proj_type: %d, dir (%f,%f,%f), wrong len "
        "%f, dif %f\n",
        msphere.id, particle_proj.proj_type, particle_proj.dir[0],
        particle_proj.dir[1], particle_proj.dir[2], len, diff);
    assert(false);
  }
  return;
}

void project_particle_gradient(const ParticleProjection& particle_proj,
                               Vector3& grad, bool is_debug) {
  if (particle_proj.proj_type == ParticleProjectionType::FAIL ||
      particle_proj.proj_type == ParticleProjectionType::FIXED) {
    grad = Vector3(0, 0, 0);
    return;
  }
  // particle_proj.dir must be normalized
  // including (0,0,0) if ParticleProjectionType::FAIL
  double len = GEO::length(particle_proj.dir);
  double diff = std::abs(1.f - len);
  if (diff > SCALAR_ZERO_6) {
    printf(
        "[ParticleTangent] proj_type: %d, dir (%f,%f,%f), wrong len %f, diff "
        "%f\n",
        particle_proj.proj_type, particle_proj.dir[0], particle_proj.dir[1],
        particle_proj.dir[2], len, diff);
    assert(false);
  }

  if (particle_proj.proj_type == ParticleProjectionType::PLANE) {
    grad = grad - GEO::dot(grad, particle_proj.dir) * particle_proj.dir;
  } else if (particle_proj.proj_type == ParticleProjectionType::LINE ||
             particle_proj.proj_type == ParticleProjectionType::SE_LINE) {
    grad = GEO::dot(grad, particle_proj.dir) * particle_proj.dir;
  } else {
    printf("[ParticleTangent] unkown projection type %d\n",
           particle_proj.proj_type);
    assert(false);
  }
}

void project_particle_sphere(const ParticleProjection& particle_proj,
                             Vector4& mi, bool is_debug) {
  if (particle_proj.proj_type == ParticleProjectionType::FAIL) {
    // do not change mi
    return;
  }

  // project mi to solution PLANE or LINE
  Vector3 theta_0(mi[0], mi[1], mi[2]);
  Vector3 theta = particle_proj.pseudo_center;
  Vector3 theta_proj(0, 0, 0);
  double ri = -1.f;
  if (particle_proj.proj_type == ParticleProjectionType::FIXED) {
    theta_proj = particle_proj.pseudo_center;
  } else if (particle_proj.proj_type == ParticleProjectionType::PLANE) {
    theta_proj = theta_0 + GEO::dot(theta - theta_0, particle_proj.dir) *
                               particle_proj.dir;

  } else if (particle_proj.proj_type == ParticleProjectionType::LINE ||
             particle_proj.proj_type == ParticleProjectionType::SE_LINE) {
    theta_proj = theta + GEO::dot(theta_0 - theta, particle_proj.dir) *
                             particle_proj.dir;
  }

  // update radius
  if (particle_proj.proj_type == ParticleProjectionType::SE_LINE) {
    ri = SCALAR_FEATURE_RADIUS;
  } else if (particle_proj.proj_type == ParticleProjectionType::FIXED) {
    ri = particle_proj.pseudo_radius;
  } else {
    // ri is changed based on the theta_proj
    Vector3 A_row0_to2 =
        Vector3(particle_proj.A_row0[0], particle_proj.A_row0[1],
                particle_proj.A_row0[2]);
    ri = (particle_proj.b_0 - GEO::dot(A_row0_to2, theta_proj)) /
         particle_proj.A_row0[3];
  }
  if (ri == -1.f || isnan(ri)) {
    // do not update mi, return
    return;
  }
  assert(ri != -1.f && !isnan(ri));
  mi = Vector4(theta_proj[0], theta_proj[1], theta_proj[2], ri);
}

// for debug
void get_particle_tangent_or_normal_all_spheres(
    const std::vector<FeatureEdge>& feature_edges, const RPD3D_Wrapper& rpd3d,
    std::vector<ParticleProjection>& all_particle_projs,
    std::vector<Vector4>& all_pseudo_sphere_projs, bool is_debug) {
  printf("calling get particle tangent or normaln all spheres ... \n");
  int num_spheres = rpd3d.all_medial_spheres->size();
  all_particle_projs.clear();
  all_particle_projs.resize(num_spheres);
  all_pseudo_sphere_projs.clear();
  all_pseudo_sphere_projs.resize(num_spheres);
  GEO::parallel_for(0, num_spheres, [&](int sphere_id) {
    // if (sphere_id == 1461)
    //   is_debug = true;
    // else
    //   is_debug = false;
    const auto& msphere = rpd3d.all_medial_spheres->at(sphere_id);
    get_particle_tangent_or_normal_wrapper(
        msphere, feature_edges, rpd3d, all_particle_projs[sphere_id], is_debug);

    all_pseudo_sphere_projs[sphere_id] =
        Vector4(msphere.center[0], msphere.center[1], msphere.center[2],
                msphere.radius);
    project_particle_sphere(all_particle_projs[sphere_id],
                            all_pseudo_sphere_projs[sphere_id], is_debug);
  });
}