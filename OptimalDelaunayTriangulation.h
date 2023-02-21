#pragma once
#include <easy3d/core/surface_mesh.h>
#include<easy3d/core/poly_mesh.h>
namespace easy3d
{


	class OptimalDelaunayTriangulation
	{
	public:
		OptimalDelaunayTriangulation(SurfaceMesh* mesh);

		~OptimalDelaunayTriangulation();

		float Calculate_face_area(SurfaceMesh::Face f);

		vec3 Calculate_face_circumcenter(SurfaceMesh::Face f);

		void getDelaunayTriangulation(int iteration);

		
	private:

		SurfaceMesh* mesh_;
		PolyMesh * pmesh_;
		SurfaceMesh::VertexProperty <vec3> points_;
		std::vector<vec3>pointvector;
	};
}

