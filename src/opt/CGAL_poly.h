#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/point_generators_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 CGAL_Point;
typedef CGAL::Polyhedron_3<K> CGAL_Mesh;

namespace PMP = CGAL::Polygon_mesh_processing;