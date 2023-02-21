/********************************************************************
 * Copyright (C) 2020-2021 by Liangliang Nan <liangliang.nan@gmail.com>
 * Copyright (C) 2011-2020 the Polygon Mesh Processing Library developers.
 *
 * The code in this file is adapted from the PMP (Polygon Mesh Processing
 * Library) with modifications.
 *      https://github.com/pmp-library/pmp-library
 * The original code was distributed under a MIT-style license, see
 *      https://github.com/pmp-library/pmp-library/blob/master/LICENSE.txt
 ********************************************************************/


#include <easy3d/algo/surface_mesh_smoothing.h>
#include <easy3d/algo/surface_mesh_geometry.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#define EPS 1e-4


namespace easy3d {

    // \cond
    using SparseMatrix = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;
    // \endcond

    //-----------------------------------------------------------------------------

    SurfaceMeshSmoothing::SurfaceMeshSmoothing(SurfaceMesh *mesh) : mesh_(mesh) {
        how_many_edge_weights_ = 0;
        how_many_vertex_weights_ = 0;
    }

    //-----------------------------------------------------------------------------

    SurfaceMeshSmoothing::~SurfaceMeshSmoothing() {
        auto vweight = mesh_->get_vertex_property<float>("v:area");
        if (vweight)
            mesh_->remove_vertex_property(vweight);

        auto eweight = mesh_->get_edge_property<float>("e:cotan");
        if (eweight)
            mesh_->remove_edge_property(eweight);
    }

    //-----------------------------------------------------------------------------

    void SurfaceMeshSmoothing::compute_edge_weights(bool use_uniform_laplace) {
        auto eweight = mesh_->edge_property<float>("e:cotan");

        if (use_uniform_laplace) {
            for (auto e : mesh_->edges())
                eweight[e] = 1.0;
        } else {
            for (auto e : mesh_->edges())
                eweight[e] = std::max(0.0, geom::cotan_weight(mesh_, e));
        }

        how_many_edge_weights_ = mesh_->n_edges();
    }

    //-----------------------------------------------------------------------------

    void SurfaceMeshSmoothing::compute_vertex_weights(bool use_uniform_laplace) {
        auto vweight = mesh_->vertex_property<float>("v:area");

        if (use_uniform_laplace) {
            for (auto v : mesh_->vertices())
                vweight[v] = 1.0 / mesh_->valence(v);
        } else {
            for (auto v : mesh_->vertices())
                vweight[v] = 0.5 / geom::voronoi_area(mesh_, v);
        }

        how_many_vertex_weights_ = mesh_->n_vertices();
    }

    //-----------------------------------------------------------------------------

    void SurfaceMeshSmoothing::explicit_smoothing(unsigned int iters,
                                                  bool use_uniform_laplace) {
        if (!mesh_->n_vertices())
            return;

        //1. compute Laplace weight per edge: cotan or uniform
        
        auto eweight = mesh_->get_edge_property<float>("e:cotan");
        if (!eweight || how_many_edge_weights_ != mesh_->n_edges())
            compute_edge_weights(use_uniform_laplace);
        eweight = mesh_->get_edge_property<float>("e:cotan");

        auto points = mesh_->get_vertex_property<vec3>("v:point");
        auto laplace = mesh_->add_vertex_property<vec3>("v:laplace");

        // smoothing iterations
        SurfaceMesh::Vertex vv;
        SurfaceMesh::Edge e;
        for (unsigned int i = 0; i < iters; ++i) {
            // step 1: compute Laplace for each vertex
            for (auto v : mesh_->vertices()) {
                vec3 l(0, 0, 0);

                if (!mesh_->is_border(v)) {
                    float w(0);

                    for (auto h : mesh_->halfedges(v)) {
                        vv = mesh_->target(h);
                        e = mesh_->edge(h);
                        l += eweight[e] * (points[vv] - points[v]);
                        w += eweight[e];
                    }

                    l /= w;
                }

                laplace[v] = l;
            }

            // step 2: move each vertex by its (damped) Laplacian
            for (auto v : mesh_->vertices()) {
                points[v] += 0.5f * laplace[v];
            }
        }

        // clean-up custom properties
        mesh_->remove_vertex_property(laplace);
    }

    //-----------------------------------------------------------------------------

    void SurfaceMeshSmoothing::implicit_smoothing(float timestep,
                                                  bool use_uniform_laplace,
                                                  bool rescale) {
        if (!mesh_->n_vertices())
            return;

        // compute edge weights if they don't exist or if the mesh changed
        auto eweight = mesh_->get_edge_property<float>("e:cotan");
        if (!eweight || how_many_edge_weights_ != mesh_->n_edges())
            compute_edge_weights(use_uniform_laplace);
        eweight = mesh_->get_edge_property<float>("e:cotan");

        // compute vertex weights
        compute_vertex_weights(use_uniform_laplace);

        // store center and area
        vec3 center_before = geom::centroid(mesh_);
        float area_before = geom::surface_area(mesh_);

        // properties
        auto points = mesh_->get_vertex_property<vec3>("v:point");
        auto vweight = mesh_->get_vertex_property<float>("v:area");
        auto idx = mesh_->add_vertex_property<int>("v:idx", -1);

        // collect free (non-boundary) vertices in array free_vertices[]
        // assign indices such that idx[ free_vertices[i] ] == i
        unsigned i = 0;
        std::vector<SurfaceMesh::Vertex> free_vertices;
        free_vertices.reserve(mesh_->n_vertices());
        for (auto v : mesh_->vertices()) {
            if (!mesh_->is_border(v)) {
                idx[v] = i++;
                free_vertices.push_back(v);
            }
        }
        const unsigned int n = free_vertices.size();

        // A*X = B
        SparseMatrix A(n, n);
        Eigen::MatrixXd B(n, 3);

        // nonzero elements of A as triplets: (row, column, value)
        std::vector<Triplet> triplets;

        // setup matrix A and rhs B
        dvec3 b;
        double ww;
        SurfaceMesh::Vertex v, vv;
        SurfaceMesh::Edge e;
        for (unsigned int i = 0; i < n; ++i) {
            v = free_vertices[i];

            // rhs row
            b = static_cast<dvec3>(points[v]) / vweight[v];

            // lhs row
            ww = 0.0;
            for (auto h : mesh_->halfedges(v)) {
                vv = mesh_->target(h);
                e = mesh_->edge(h);
                ww += eweight[e];

                // fixed boundary vertex -> right hand side
                if (mesh_->is_border(vv)) {
                    b -= -timestep * eweight[e] * static_cast<dvec3>(points[vv]);
                }
                    // free interior vertex -> matrix
                else {
                    triplets.emplace_back(i, idx[vv], -timestep * eweight[e]);
                }

                B.row(i) = (Eigen::Vector3d) b;
            }

            // center vertex -> matrix
            triplets.emplace_back(i, i, 1.0 / vweight[v] + timestep * ww);
        }

        // build sparse matrix from triplets
        A.setFromTriplets(triplets.begin(), triplets.end());

        // solve A*X = B
        Eigen::SimplicialLDLT<SparseMatrix> solver(A);
        Eigen::MatrixXd X = solver.solve(B);
        if (solver.info() != Eigen::Success) {
            std::cerr << "SurfaceMeshSmoothing: Could not solve linear system\n";
        } else {
            // copy solution
            for (unsigned int i = 0; i < n; ++i) {
                const auto &tmp = X.row(i);
                points[free_vertices[i]] = vec3(tmp(0), tmp(1), tmp(2));
            }
        }

        if (rescale) {
            // restore original surface area
            float area_after = geom::surface_area(mesh_);
            float scale = sqrt(area_before / area_after);
            for (auto v : mesh_->vertices())
                mesh_->position(v) *= scale;

            // restore original center
            vec3 center_after = geom::centroid(mesh_);
            vec3 trans = center_before - center_after;
            for (auto v : mesh_->vertices())
                mesh_->position(v) += trans;
        }

        // clean-up
        mesh_->remove_vertex_property(idx);
    }

    void SurfaceMeshSmoothing::BilateralNormalFiltering_smoothing(float Normal_right,
        int normal_iteration,
        int vertex_iteration){
        if (!mesh_->n_vertices())
        {
            return;
        }
        std::vector<vec3>FNormal(mesh_->n_faces());
        std::vector<vec3>newNormal(mesh_->n_faces());
        std::vector<double>FaceArea(mesh_->n_faces());
        std::vector<vec3>FaceCenter(mesh_->n_faces());
        std::vector<vec3>NewPos(mesh_->n_vertices());

        //float SigmaCenter = 0.0;
        //for (auto face : mesh_->faces())
        //{
        //    int f_id = face.idx();
        //    newNormal[f_id] = mesh_->compute_face_normal(face);
        //    FNormal[f_id] = mesh_->compute_face_normal(face);
        //    FaceArea[f_id] = mesh_->compute_face_Area(face);
        //    FaceCenter[f_id] = mesh_->compute_face_gravityPoint(face);
        //    //
        //    for (auto halfedge : mesh_->halfedges(face))
        //    {
        //        auto nei_f = mesh_->face(mesh_->opposite(halfedge));
        //        SigmaCenter += (FaceCenter[f_id] - FaceCenter[nei_f.idx()]).norm();
        //    }

        //}
        //SigmaCenter = SigmaCenter / (mesh_->n_faces() * 3);

        //for (int i = 0;i < normal_iteration;i++)
        //{
        //    for (auto face : mesh_->faces())
        //    {
        //        double Kp = 0;
        //        vec3 NewN(0, 0, 0);
        //        int fid = face.idx();
        //        double delta_center = 0;
        //        double delta_normal = 0;
        //        double Aj = 0;
        //        double Wc = 0;
        //        double Ws = 0;
        //        for (auto halfedge : mesh_->halfedges(face))
        //        {
        //            auto nei_f = mesh_->face(mesh_->opposite(halfedge));
        //            delta_center = (FaceCenter[face.idx()] - FaceCenter[nei_f.idx()]).norm();
        //            delta_normal = (newNormal[face.idx()] - newNormal[nei_f.idx()]).norm();
        //            Aj = FaceArea[nei_f.idx()];
        //            Wc = exp(-delta_center * delta_center / (2 * SigmaCenter * SigmaCenter));
        //            Ws = exp(-delta_normal * delta_normal / (2 * Normal_right * Normal_right));
        //            NewN += Aj * Wc * Ws * newNormal[nei_f.idx()];
        //            Kp+=Aj*Wc*Ws;
        //        }
        //        newNormal[face.idx()] = NewN / Kp;
        //        newNormal[face.idx()].normalize();
        //    }
        //}

        //for (int i = 0;i < vertex_iteration;i++)
        //{
        //    for (auto vertex : mesh_->vertices())
        //    {
        //        vec3 x = mesh_->position(vertex);
        //        NewPos[vertex.idx()] = x;
        //        vec3 delta_xi(0, 0, 0);
        //        int Nei_count = 0;
        //        
        //        for (auto face : mesh_->faces(vertex))
        //        {
        //            Nei_count++;
        //            vec3 Center = mesh_->compute_face_gravityPoint(face);
        //            vec3 nj = newNormal[face.idx()];
        //            delta_xi = delta_xi + nj * (nj[0] * (Center[0] - x[0]) + nj[1] * (Center[1] - x[1]) + nj[2] * (Center[2] - x[2]));
        //        }
        //        x = x + delta_xi / Nei_count;
        //        mesh_->position(vertex) = x;
        //        NewPos[vertex.idx()] = x;
        //    }
        //}


        //1. Calculate Center right;
        //2. compute each face normal
        float Center_right = 0;
        for (SurfaceMesh::Face face : mesh_->faces())
        {
            vec3 faceP  = mesh_->compute_face_gravityPoint(face);
            vec3 face_normal = mesh_->compute_face_normal(face);
            //mesh_->appendBilateral_faceNormal(face, face_normal);改为
            FNormal[face.idx()] = face_normal;
            newNormal[face.idx()] = face_normal;
            for (SurfaceMesh::Halfedge halfeg : SurfaceMesh::HalfedgeAroundFaceCirculator(mesh_, face))
            {
                //if (!mesh_->is_border(halfeg))
                {
                    SurfaceMesh::Halfedge opposite_halfEg = mesh_->opposite(halfeg);
                    SurfaceMesh::Face neigborFace = mesh_->face(opposite_halfEg);
					
                    vec3 temp_facep = mesh_->compute_face_gravityPoint(neigborFace);
                    Center_right += length(temp_facep - faceP);//temp_facep.distance2(faceP);
                }
            }
        }
        Center_right = Center_right / (3 * mesh_->n_faces());

        //step1 : update face normal
        
        for (int i = 0;i < normal_iteration;i++)//
        {
         //--------------出错了：：   float Kp = 0;
            
            for (SurfaceMesh::Face face : mesh_->faces())
            {
                float Kp = 0;
                vec3 Normal;
                vec3 face_normal = mesh_->compute_face_normal(face);
                vec3 face_gpoint = mesh_->compute_face_gravityPoint(face);

                for (SurfaceMesh::Halfedge halfeg : SurfaceMesh::HalfedgeAroundFaceCirculator(mesh_, face))
                {
                    
                    //if (!mesh_->is_border(halfeg))
                    {
                        SurfaceMesh::Halfedge opposite_halfEg = mesh_->opposite(halfeg);
                        SurfaceMesh::Face neigborFace = mesh_->face(opposite_halfEg);
                        vec3 nei_normal = mesh_->compute_face_normal(neigborFace);
                        vec3 nei_gpoint = mesh_->compute_face_gravityPoint(neigborFace);
                        float nei_Area = mesh_->compute_face_Area(neigborFace);
                        //Calculate surface Ws
                        float normal_length = (nei_normal - face_normal).norm();//nei_normal.distance2(face_normal);
                        float Ws = exp(-(normal_length * normal_length) / (2 * Normal_right * Normal_right));
                        //Calculate surface Wc
                        float center_length = (nei_gpoint - face_gpoint).norm();//nei_gpoint.distance2(face_gpoint);
                        float Wc = exp(-(center_length * center_length) / (2 * Center_right * Center_right));

                        //Calculate Kp
                        Kp += nei_Area * Ws * Wc;
                        //Calculate N
                        Normal += nei_Area * Ws * Wc * nei_normal;
                    }
                }
                face_normal = Normal / Kp;
                face_normal = face_normal.normalize();
                /*mesh_->Reset_Bilateral_faceNormal(face, face_normal);*/
                newNormal[face.idx()] = face_normal;
            }
        }
         
      
         
        //step2 : Calculte NewN(f)
        for (int i = 0;i < vertex_iteration;i++)
        {
            for (SurfaceMesh::Vertex vertex : mesh_->vertices())
            {
                vec3 dx(0, 0, 0);
                vec3 Normal(0, 0, 0);
                int Nei_count = 0;
                vec3 x_i = mesh_->position(vertex);
                //if (!mesh_->is_border(vertex))
                {
                    for (SurfaceMesh::Face nei_vertex_face : mesh_->faces(vertex))
                    {
                        Nei_count++;
                        vec3 face_center = mesh_->compute_face_gravityPoint(nei_vertex_face);
                        /*Normal = mesh_->Get_Bilateral_faceNormal(nei_vertex_face);*/
                        Normal = newNormal[nei_vertex_face.idx()];
                        vec3 diff_Vec = face_center - mesh_->position(vertex);
                        //dx += Normal * (Normal.x * diff_Vec.x + Normal.y * diff_Vec.y + Normal.z * diff_Vec.z);
                        //dx += (Normal.length() * Normal.length()) * (face_center - mesh_->position(vertex));
                        dx += Normal * ((face_center[0] - x_i[0]) * Normal[0] + (face_center[1] - x_i[1]) * Normal[1] + (face_center[2] - x_i[2]) * Normal[2]);
                    }
                    x_i += dx / Nei_count;
                    mesh_->position(vertex) = x_i;
                }
            }
        }
       
        double ees = ErrorEstimator(FNormal,newNormal);
        std::cout << ees << std::endl;
    }
    double SurfaceMeshSmoothing::ErrorEstimator(std::vector<vec3> FNormal_, std::vector<vec3>New_normal)//传的是oldnormals与新normals比较
    {
        double thetasqsum = 0.0;
        double innerproduct = 0.0;
        double temp = 0.0;
        vec3 n1;
        vec3 n2;
        for (int i = 0;i < mesh_->n_faces();i++)
        {
            n1 = FNormal_[i];
            n2 = New_normal[i];

            innerproduct = dot(n1, n2);

            double costheta = innerproduct / (n1.norm() * n2.norm() + 1e-8);

            if (abs(costheta - 1) <= EPS)
            {
                costheta = 1;
            }
            if (abs(costheta + 1) <= EPS)
            {
                costheta = -1;
            }
            temp = acos(costheta);

            thetasqsum += pow(temp, 2);
        }
        return thetasqsum / (mesh_->n_faces());
    }
} // namespace easy3d
