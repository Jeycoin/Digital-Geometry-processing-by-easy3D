#include "OptimalDelaunayTriangulation.h"

#include"math.h"

namespace easy3d
{
	OptimalDelaunayTriangulation::OptimalDelaunayTriangulation(SurfaceMesh * mesh)
	{
		mesh_ = mesh;
		points_ = mesh_->get_vertex_property<vec3>("v:point");
		
	}

	OptimalDelaunayTriangulation::~OptimalDelaunayTriangulation() = default;

	float OptimalDelaunayTriangulation::Calculate_face_area(SurfaceMesh::Face f)
	{
		return mesh_->compute_face_Area(f);
	}

	vec3 OptimalDelaunayTriangulation::Calculate_face_circumcenter(SurfaceMesh::Face f)
	{
		SurfaceMesh::Halfedge h = mesh_->halfedge(f);

		vec3 vertices[3];
		int i = 0;
		for (auto vertex : mesh_->vertices(f))
		{
			if (i == 3)break;
			vertices[i] = mesh_->position(vertex);
			i++;
		}

		double x1, y1, x2, y2, x3, y3;
		x1 = vertices[0].x; y1 = vertices[0].y;
		x2 = vertices[1].x; y2 = vertices[1].y;
		x3 = vertices[2].x; y3 = vertices[2].y;
	
		double a1, b1, c1, a2, b2, c2;
		a1 = 2 * (x2 - x1);	  a2 = 2 * (x3 - x2);	c1 = x2 * x2 + y2 * y2 - x1 * x1 - y1 * y1;
		b1 = 2 * (y2 - y1);	  b2 = 2 * (y3 - y2);	c2 = x3 * x3 + y3 * y3 - x2 * x2 - y2 * y2;

		vec3 circumcenter(0.0, 0.0, 0.0);
		circumcenter[0] = (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1);
		circumcenter[1] = (a1 * c2 - a2 * c1) / (a1 * b2 - a2 * b1);
		circumcenter[2] = 0;

		return circumcenter;

	}

	void OptimalDelaunayTriangulation::getDelaunayTriangulation(int iteration)
	{
	
	


		int k = 0;
		for (int i = 0; i < iteration; i++)
		{
			//flip
			for (auto edge : mesh_->edges())
			{
				if (mesh_->is_border(edge) || !mesh_->is_flip_ok(edge)) continue;

				SurfaceMesh::Halfedge h1 = mesh_->halfedge(edge, 0);
				SurfaceMesh::Halfedge h2 = mesh_->next(h1);
				SurfaceMesh::Halfedge h3 = mesh_->next(mesh_->halfedge(edge, 1));

				vec3 v1 = mesh_->position(mesh_->source(h1));
				vec3 v2 = mesh_->position(mesh_->target(h1));
				vec3 v3 = mesh_->position(mesh_->target(h2));
				vec3 v4 = mesh_->position(mesh_->target(h3));

				double alpha(0.0), alpha1(0.0), alpha2(0.0);
				alpha1 = acos((pow((v1 - v3).norm(), 2) + pow((v2 - v3).norm(), 2)
					- pow((v1 - v2).norm(), 2)) / (2 * (v1 - v3).norm() * (v2 - v3).norm()));
				alpha2 = acos((pow((v1 - v4).norm(), 2) + pow((v2 - v4).norm(), 2)
					- pow((v1 - v2).norm(), 2)) / (2 * (v1 - v4).norm() * (v2 - v4).norm()));
				alpha = alpha1 + alpha2;
				if (alpha > M_PI)	mesh_->flip(edge);
			}

			//update vertex position
			for (auto vertex : mesh_->vertices())
			{
				if (mesh_->is_border(vertex))continue;

				vec3 tmp(0.0, 0.0, 0.0);
				double area(0.0), sum_area(0.0);
				for (auto face : mesh_->faces(vertex))
				{
					area = mesh_->compute_face_Area(face);
					sum_area += area;
					vec3 center = Calculate_face_circumcenter(face);
					tmp = tmp + area * center;
				}
				mesh_->position(vertex) = tmp / sum_area;
			}
		}
	}

	
}
