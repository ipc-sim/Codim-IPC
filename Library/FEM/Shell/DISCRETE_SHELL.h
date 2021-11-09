#pragma once

#include <Math/AMGCL_SOLVER.h>

#include <FEM/Shell/Rod/DISCRETE_ROD.h>
#include <FEM/Shell/DISCRETE_PARTICLE.h>

#include <FEM/Shell/IMPLICIT_EULER.h>
#include <FEM/Shell/SPLITTING_IE.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim = 3>
VECTOR<int, 4> Add_Discrete_Shell_3D(
    const std::string& filePath,
    const VECTOR<T, dim>& trans,
    const VECTOR<T, dim>& scale,
    const VECTOR<T, dim>& rotCenter,
    const VECTOR<T, dim>& rotAxis,
    T rotAngDeg,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_ELEM<dim - 1>& Elem, // the mid-surface triangles
    std::vector<int>& compNodeRange)
{
    MESH_NODE<T, dim> newX;
    MESH_ELEM<dim - 1> newElem;
    VECTOR<int, 4> counter = Read_TriMesh_Obj(filePath, newX, newElem);
    counter[0] += X.size;
    counter[2] += X.size;
    counter[1] += Elem.size;
    counter[3] += Elem.size;

    T rotAngRad = rotAngDeg / 180 * M_PI;
    newX.Par_Each([&](int id, auto data){
        auto &[X] = data;
        if constexpr (dim == 3) {
            const Eigen::Matrix3d rotMtr = Eigen::AngleAxis<T>(rotAngRad,
                Eigen::Vector3d(rotAxis[0], rotAxis[1], rotAxis[2])).toRotationMatrix();
            Eigen::Vector3d x((X - rotCenter).data);
            x = x.cwiseProduct(Eigen::Vector3d(scale.data));
            const Eigen::Vector3d rotx = rotMtr * x;
            for (int i = 0; i < dim; ++i) {
                X[i] = rotx[i] + rotCenter[i] + trans[i];
            }
        }
        else {
            //TODO
        }
    });
    newElem.Par_Each([&](int id, auto data){
        auto &[elemVInd] = data;
        for (int i = 0; i < dim; ++i) {
            elemVInd[i] += X.size;
        }
    });

    Append_Attribute(newX, X);
    Append_Attribute(newElem, Elem);

    compNodeRange.emplace_back(X.size);

    return counter;
}

template <class T, int dim = 3>
void Add_Garment_3D(
    const std::string& filePath,
    const VECTOR<T, dim>& trans,
    const VECTOR<T, dim>& scale,
    const VECTOR<T, dim>& rotCenter,
    const VECTOR<T, dim>& rotAxis,
    T rotAngDeg,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE<T, dim>& X_stage, // mid-surface node coordinates
    MESH_ELEM<dim - 1>& Elem, // the mid-surface triangles
    std::vector<VECTOR<int, 3>>& stitchNodes, 
    std::vector<T>& stitchRatio,
    std::vector<int>& compNodeRange)
{
    MESH_NODE<T, dim> newX, newXTex;
    MESH_ELEM<dim - 1> newElem, newElemTex;
    Read_TriMesh_Tex_Obj(filePath, newX, newXTex, newElem, newElemTex, stitchNodes, stitchRatio);

    T rotAngRad = rotAngDeg / 180 * M_PI;
    newX.Par_Each([&](int id, auto data){
        auto &[X] = data;
        if constexpr (dim == 3) {
            const Eigen::Matrix3d rotMtr = Eigen::AngleAxis<T>(rotAngRad,
                Eigen::Vector3d(rotAxis[0], rotAxis[1], rotAxis[2])).toRotationMatrix();
            Eigen::Vector3d x((X - rotCenter).data);
            x = x.cwiseProduct(Eigen::Vector3d(scale.data));
            const Eigen::Vector3d rotx = rotMtr * x;
            for (int i = 0; i < dim; ++i) {
                X[i] = rotx[i] + rotCenter[i] + trans[i];
            }
        }
        else {
            //TODO
        }
    });
    newElem.Par_Each([&](int id, auto data){
        auto &[elemVInd] = data;
        for (int i = 0; i < dim; ++i) {
            elemVInd[i] += X.size;
        }
    });

    Append_Attribute(newXTex, X);
    Append_Attribute(newX, X_stage);
    Append_Attribute(newElem, Elem);

    compNodeRange.emplace_back(X.size);
}

template <class T, int dim>
void Initialize_Garment(
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE<T, dim>& X_stage) // mid-surface node coordinates
{
    if (X_stage.size > X.size) {
        std::cout << "garment mesh node size mismatch with global mesh" << std::endl;
        exit(-1);
    }
    X_stage.Par_Each([&](int id, auto data){
        auto &[xI] = data;
        std::get<0>(X.Get_Unchecked(id)) = xI;
    });
}

template <class T, int dim, bool KL>
void Compute_Discrete_Shell_Inv_Basis(
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim - 1>& Elem,
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 2>>& edge,
    std::vector<VECTOR<int, 4>>& edgeStencil,
    std::vector<VECTOR<T, 3>>& edgeInfo,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr)
{
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        elemAttr = MESH_ELEM_ATTR<T, dim - 1>(Elem.size);
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));

            const VECTOR<T, dim> E01 = X2 - X1;
            const VECTOR<T, dim> E02 = X3 - X1;
            MATRIX<T, dim - 1> IB; // for first fundamental form
            IB(0, 0) = E01.length2();
            IB(1, 0) = IB(0, 1) = E01.dot(E02);
            IB(1, 1) = E02.length2();
            // IB.invert();

            Eigen::Matrix<T, dim, 1> cNormal;
            Eigen::Matrix<T, dim, 1> oppNormals[3];
            T mnorms[3];
            MATRIX<T, dim - 1> D; // for second fundamental form
            Compute_SFF(X, Elem, edge2tri, id, cNormal, oppNormals, mnorms, D);
            
            elemAttr.Append(IB, D);
        });

        if constexpr (!KL) {
            edgeInfo.resize(0);
            edgeStencil.resize(0);
            for (int eI = 0; eI < edge.size(); ++eI) {
                const auto triFinder = edge2tri.find(std::pair<int, int>(edge[eI][0], edge[eI][1]));
                if (triFinder != edge2tri.end()) {
                    const VECTOR<int, 3>& elemVInd_tri = std::get<0>(Elem.Get_Unchecked(triFinder->second));
                    VECTOR<int, 3> elemVInd;
                    for (int j = 0; j < 3; ++j) {
                        if (elemVInd_tri[j] == edge[eI][1]) {
                            elemVInd[0] = elemVInd_tri[(j + 1) % 3];
                            elemVInd[1] = edge[eI][0];
                            elemVInd[2] = edge[eI][1];
                            break;
                        }
                    }
                    const auto finder = edge2tri.find(std::pair<int, int>(elemVInd[2], elemVInd[1]));
                    if (finder != edge2tri.end()) {
                        const VECTOR<int, 3>& oppElemVInd = std::get<0>(Elem.Get_Unchecked(finder->second));
                        int v3I = 0;
                        for (int j = 0; j < 3; ++j) {
                            if (oppElemVInd[j] == elemVInd[1]) {
                                v3I = oppElemVInd[(j + 1) % 3];
                                break;
                            }
                        }
                        edgeStencil.emplace_back(VECTOR<int, 4>(elemVInd[0], elemVInd[1], elemVInd[2], v3I));

                        const VECTOR<T, dim>& X0 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                        const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                        const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                        const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(v3I));
                        Eigen::Matrix<T, dim, 1> X0e(X0.data), X1e(X1.data), X2e(X2.data), X3e(X3.data);
                        
                        edgeInfo.resize(edgeInfo.size() + 1);
                        Compute_Dihedral_Angle(X0e, X1e, X2e, X3e, edgeInfo.back()[0]);
                        edgeInfo.back()[1] = (X1 - X2).length();
                        VECTOR<T, 3> n1 = cross(X1 - X0, X2 - X0);
                        VECTOR<T, 3> n2 = cross(X2 - X3, X1 - X3);
                        edgeInfo.back()[2] = (n1.length() + n2.length()) / (edgeInfo.back()[1] * 6); 
                    }
                }
            }
            std::cout << edgeStencil.size() << " hinges" << std::endl;
        }
    }
    std::cout << "IB and D computed" << std::endl;
}

template <class T, int dim, bool KL, bool elasticIPC>
T Initialize_Discrete_Shell(
    T rho0, T E, T nu, T thickness, T h, T dHat2,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_ELEM<dim - 1>& Elem,
    std::vector<VECTOR<int, 2>>& seg,
    std::map<std::pair<int, int>, int>& edge2tri,
    std::vector<VECTOR<int, 4>>& edgeStencil,
    std::vector<VECTOR<T, 3>>& edgeInfo,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix
    const VECTOR<T, dim>& gravity,
    std::vector<T>& b, // body force
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    VECTOR<T, 3>& kappa)
{
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        MESH_ELEM<dim - 1> filteredElem(Elem.size);
        Elem.Each([&](int id, auto data){
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));

            const VECTOR<T, dim> E01 = X2 - X1;
            const VECTOR<T, dim> E02 = X3 - X1;
            MATRIX<T, dim - 1> IB; // for first fundamental form
            IB(0, 0) = E01.length2();
            IB(1, 0) = IB(0, 1) = E01.dot(E02);
            IB(1, 1) = E02.length2();
            if(IB.determinant() != 0) {
                filteredElem.Append(elemVInd);
            }
        });
        filteredElem.deep_copy_to(Elem);

        nodeAttr = MESH_NODE_ATTR<T, dim>(X.size);
        for (int i = 0; i < X.size; ++i) {
            nodeAttr.Append(std::get<0>(X.Get_Unchecked(i)), VECTOR<T, dim>(0, 0), VECTOR<T, dim>(), 0);
        }

        edge2tri.clear();
        Elem.Each([&](int id, auto data){
            auto &[elemVInd] = data;
            edge2tri[std::pair<int, int>(elemVInd[0], elemVInd[1])] = id;
            edge2tri[std::pair<int, int>(elemVInd[1], elemVInd[2])] = id;
            edge2tri[std::pair<int, int>(elemVInd[2], elemVInd[0])] = id;
        });

        std::vector<int> boundaryNode;
        std::vector<VECTOR<int, 2>> edge;
        std::vector<VECTOR<int, 3>> boundaryTri;
        if constexpr (!KL) {
            Find_Surface_Primitives(X.size, Elem, boundaryNode, edge, boundaryTri);
        }
        Compute_Discrete_Shell_Inv_Basis<T, dim, KL>(X, Elem, edge2tri, edge, edgeStencil, edgeInfo, elemAttr);

        // mass matrix and body force
        std::vector<Eigen::Triplet<T>> triplets;
        b.resize(0);
        b.resize(X.size * dim, 0);
        T massPortionMean = 0;
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
            T& m1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(elemVInd[0]));
            T& m2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(elemVInd[1]));
            T& m3 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(elemVInd[2]));

            const T massPortion = cross(X2 - X1, X3 - X1).length() / 2 * thickness * rho0 / 3;
            massPortionMean += massPortion;
            m1 += massPortion; m2 += massPortion; m3 += massPortion;
            for (int endI = 0; endI < dim; ++endI) {
                for (int dimI = 0; dimI < dim; ++dimI) {
                    triplets.emplace_back(elemVInd[endI] * dim + dimI, elemVInd[endI] * dim + dimI, massPortion);
                    b[elemVInd[endI] * dim + dimI] += massPortion * gravity[dimI];
                }
            }
        });
        if (Elem.size) {
            massPortionMean /= Elem.size;
        }
        for (const auto& segI : seg) {
            // const VECTOR<T, dim> X0 = std::get<0>(X.Get_Unchecked(segI[0]));
            // const VECTOR<T, dim> X1 = std::get<0>(X.Get_Unchecked(segI[1]));
            // const T massPortion = (X0 - X1).length() * M_PI * thickness * thickness / 4 * rho0 / 2;
            for (int endI = 0; endI < 2; ++endI) {
                for (int dimI = 0; dimI < dim; ++dimI) {
                    // triplets.emplace_back(segI[endI] * dim + dimI, segI[endI] * dim + dimI, massPortion);
                    triplets.emplace_back(segI[endI] * dim + dimI, segI[endI] * dim + dimI, massPortionMean * 3);
                }
                // b[segI[endI] * dim + 1] += massPortion * gravity[1];
            }
        }
        M.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
        //NOTE: for implicit, need to project Matrix for Dirichlet boundary condition

        // quadratures
        elasticityAttr = FIXED_COROTATED<T, dim - 1>(Elem.size);
        const T lambda = E * nu / ((T)1 - nu * nu);
        const T mu = E / ((T)2 * ((T)1 + nu));
        T areaSum = 0;
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
            T area = cross(X2 - X1, X3 - X1).length() / 2;
            T vol = area * thickness;
            elasticityAttr.Append(MATRIX<T, dim - 1>(), vol, lambda, mu);
            areaSum += area;
        });
        if constexpr (!KL) {
            if (elemAttr.size) {
                T& k = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(0))(0, 0);
                k = E * std::pow(thickness, 3) / (24 * ((T)1 - nu * nu));
                std::cout << "hinge k = " << k << std::endl;
            }
        }

        if constexpr (elasticIPC) {
            T h2vol = h * h;
            kappa[0] = h2vol * mu;
            kappa[1] = h2vol * lambda;
            kappa[2] = nu;
            dHat2 = thickness * thickness;
        }
        else {
            //TODO: adaptive kappa
        }
    }
    std::cout << "shell initialized" << std::endl;
    return dHat2;
}

template <class T, int dim = 3>
void Update_Normal_Flow_Neumann(
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_ELEM<dim - 1>& Elem,
    CSR_MATRIX<T>& M, // mass matrix
    T magnitude,
    std::vector<T>& b) // body force
{
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        b.assign(X.size * dim, 0);
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
            VECTOR<T, dim> n = cross(X2 - X1, X3 - X1);
            for (int vI = 0; vI < dim; ++vI) {
                for (int d = 0; d < dim; ++d) {
                    b[elemVInd[vI] * dim + d] += n[d];
                }
            }
        });
        for (int vI = 0; vI < X.size; ++vI) {
            T sqnorm = 0;
            for (int d = 0; d < dim; ++d) {
                sqnorm += b[vI * dim + d] * b[vI * dim + d];
            }
            T w = magnitude * M.Get_Matrix().coeff(vI * dim, vI * dim) / std::sqrt(sqnorm);
            for (int d = 0; d < dim; ++d) {
                b[vI * dim + d] *= w;
            }
        }
    }
}

template <class T, int dim>
T Update_Material_With_Tex_Shell(
    const std::string& filePath, 
    int vIndStart, int triIndStart, 
    T rho0, T thickness,
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    std::vector<VECTOR<T, 3>>& edgeInfo,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix
    const VECTOR<T, dim>& gravity,
    std::vector<T>& b, // body force
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        MESH_NODE<T, dim> X, X_tex;
        MESH_ELEM<2> triangles, triangles_tex;
        std::vector<VECTOR<int, 3>> stitchNodes;
        std::vector<T> stitchRatio;
        Read_TriMesh_Tex_Obj(filePath, X, X_tex, triangles, triangles_tex, stitchNodes, stitchRatio);

        // membrane
        triangles_tex.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X_tex.Get_Unchecked(elemVInd[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X_tex.Get_Unchecked(elemVInd[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X_tex.Get_Unchecked(elemVInd[2]));

            const VECTOR<T, dim> E01 = X2 - X1;
            const VECTOR<T, dim> E02 = X3 - X1;
            MATRIX<T, dim - 1> IB; // for first fundamental form
            IB(0, 0) = E01.length2();
            IB(1, 0) = IB(0, 1) = E01.dot(E02);
            IB(1, 1) = E02.length2();
            std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(triIndStart + id)) = IB;

            T area = cross(X2 - X1, X3 - X1).length() / 2;
            T vol = area * thickness;
            std::get<FIELDS<FIXED_COROTATED<T, dim>>::VOL>(elasticityAttr.Get_Unchecked(triIndStart + id)) = vol;
        });
        printf("membrane updated\n");

        // bending
        for (int eI = 0; eI < edgeStencil.size(); ++eI) {
            bool needsUpdate = false;
            for (int i = 0; i < 4; ++i) {
                if (edgeStencil[eI][i] >= vIndStart && edgeStencil[eI][i] < vIndStart + X.size) {
                    needsUpdate = true;
                    break;
                }
            }
            if (!needsUpdate) {
                continue;
            }

            const auto tri0Finder = edge2tri.find(std::pair<int, int>(edgeStencil[eI][1], edgeStencil[eI][2]));
            const auto tri1Finder = edge2tri.find(std::pair<int, int>(edgeStencil[eI][2], edgeStencil[eI][1]));
            if (tri0Finder == edge2tri.end() || tri1Finder == edge2tri.end()) {
                printf("edge tri pair not found!\n");
                exit(-1);
            }
            const VECTOR<int, 3>& tri0VInd = std::get<0>(triangles.Get_Unchecked(tri0Finder->second - triIndStart));
            const VECTOR<int, 3>& tri1VInd = std::get<0>(triangles.Get_Unchecked(tri1Finder->second - triIndStart));
            int tri0start = -1;
            for (int i = 0; i < 3; ++i) {
                if (tri0VInd[i] + vIndStart == edgeStencil[eI][1]) {
                    tri0start = i;
                    break;
                }
            }
            if (tri0start == -1) {
                printf("vert not found in tri\n");
                exit(-1);
            }

            const VECTOR<int, 3>& tri0VInd_tex = std::get<0>(triangles_tex.Get_Unchecked(tri0Finder->second - triIndStart));
            const VECTOR<int, 3>& tri1VInd_tex = std::get<0>(triangles_tex.Get_Unchecked(tri1Finder->second - triIndStart));
            const VECTOR<T, dim>& tri0X0 = std::get<0>(X_tex.Get_Unchecked(tri0VInd_tex[tri0start]));
            const VECTOR<T, dim>& tri0X1 = std::get<0>(X_tex.Get_Unchecked(tri0VInd_tex[(tri0start + 1) % 3]));
            const VECTOR<T, dim>& tri0X2 = std::get<0>(X_tex.Get_Unchecked(tri0VInd_tex[(tri0start + 2) % 3]));
            const VECTOR<T, dim>& tri1X0 = std::get<0>(X_tex.Get_Unchecked(tri1VInd_tex[0]));
            const VECTOR<T, dim>& tri1X1 = std::get<0>(X_tex.Get_Unchecked(tri1VInd_tex[1]));
            const VECTOR<T, dim>& tri1X2 = std::get<0>(X_tex.Get_Unchecked(tri1VInd_tex[2]));

            edgeInfo[eI][0] = 0;
            edgeInfo[eI][1] = (tri0X0 - tri0X1).length();
            VECTOR<T, 3> n1 = cross(tri0X1 - tri0X0, tri0X2 - tri0X0);
            VECTOR<T, 3> n2 = cross(tri1X1 - tri1X0, tri1X2 - tri1X0);
            edgeInfo[eI][2] = (n1.length() + n2.length()) / (edgeInfo[eI][1] * 6); 
        }
        printf("bending updated\n");
        
        // mass matrix and body force
        std::vector<T> b_cur(X.size * dim, 0);
        for (int i = vIndStart; i < vIndStart + X.size; ++i) {
            std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(i)) = 0;
        }
        triangles.Join(triangles_tex).Each([&](int id, auto data) {
            auto &[elemVInd, elemVInd_tex] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X_tex.Get_Unchecked(elemVInd_tex[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X_tex.Get_Unchecked(elemVInd_tex[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X_tex.Get_Unchecked(elemVInd_tex[2]));
            T& m1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(vIndStart + elemVInd[0]));
            T& m2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(vIndStart + elemVInd[1]));
            T& m3 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(vIndStart + elemVInd[2]));

            const T massPortion = cross(X2 - X1, X3 - X1).length() / 2 * thickness * rho0 / 3;
            m1 += massPortion; m2 += massPortion; m3 += massPortion;
            for (int endI = 0; endI < dim; ++endI) {
                for (int dimI = 0; dimI < dim; ++dimI) {
                    b_cur[elemVInd[endI] * dim + dimI] += massPortion * gravity[dimI];
                }
            }
        });

        nodeAttr.Par_Each([&](int id, auto data) {
            auto &[x0, v, g, m] = data;
            if (id >= vIndStart && id < vIndStart + X.size) {
                M.Get_Matrix().coeffRef(id * dim, id * dim) = m;
                M.Get_Matrix().coeffRef(id * dim + 1, id * dim + 1) = m;
                M.Get_Matrix().coeffRef(id * dim + 2, id * dim + 2) = m;
            }
        });
        for (int i = 0; i < b_cur.size(); ++i) {
            b[vIndStart * dim + i] = b_cur[i];
        }
        printf("mass and body force updated\n");
    }
}

template <class T, int dim>
void Adjust_Material(
    const VECTOR<int, 4>& meshCounter,
    T densityMult, T YoungsMult,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix
    std::vector<T>& b, // body force
    FIXED_COROTATED<T, dim>& tetElasticityAttr)
{
    for (int vI = meshCounter[0]; vI < meshCounter[2]; ++vI) {
        std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(vI)) *= densityMult;

        for (int d = 0; d < dim; ++d) {
            M.Get_Matrix().coeffRef(vI * dim + d, vI * dim + d) *= densityMult;
            b[vI * dim + d] *= densityMult;
        }
    }

    for (int tI = meshCounter[1]; tI < meshCounter[3]; ++tI) {
        std::get<FIELDS<FIXED_COROTATED<T, dim>>::LAMBDA>(tetElasticityAttr.Get_Unchecked(tI)) *= YoungsMult;
        std::get<FIELDS<FIXED_COROTATED<T, dim>>::MU>(tetElasticityAttr.Get_Unchecked(tI)) *= YoungsMult;
    }
}

template <class T, bool elasticIPC = true, int dim = 3>
T Initialize_EIPC(T E, T nu, T thickness, T h, 
    CSR_MATRIX<T>& M, // mass matrix
    VECTOR<T, 3>& kappa, T stiffMult = 1)
{
    const T lambda = E * nu / ((T)1 - nu * nu);
    const T mu = E / ((T)2 * ((T)1 + nu));
    T h2 = h * h;
    kappa[0] = h2 * mu;
    kappa[1] = h2 * lambda;
    kappa[2] = nu;
    T dHat2 = thickness * thickness;

    if constexpr (!elasticIPC) {
        T H_b, kappaVec[] = {1, kappa[1], kappa[2]};
        Barrier_Hessian<elasticIPC>(1.0e-16, dHat2, kappaVec, H_b);
        kappa[0] = stiffMult * 1.0e11 * M.Get_Matrix().diagonal().mean() * dim / (4.0e-16 * H_b);
        kappa[1] = 100 * kappa[0];
        printf("original IPC kappa = %le\n", kappa[0]);
    }

    return dHat2;
}

template <class T, int dim = 3>
T Initialize_OIPC_VecM(T dHat2, 
    MESH_NODE_ATTR<T, dim>& nodeAttr, // mass
    VECTOR<T, 3>& kappa, 
    T stiffMult = 1)
{
    T avgmass = 0;
    nodeAttr.Each([&](int id, auto data) {
        auto &[x0, v, g, m] = data;
        avgmass += m;
    });
    avgmass /= nodeAttr.size;

    T H_b, kappaVec[] = {1, kappa[1], kappa[2]};
    Barrier_Hessian<false>(1.0e-16, dHat2, kappaVec, H_b);
    kappa[0] = stiffMult * 1.0e11 * avgmass / (4.0e-16 * H_b);
    kappa[1] = 100 * kappa[0];
    printf("original IPC kappa = %le\n", kappa[0]);

    return dHat2;
}

template<class T, int dim>
void Compute_Max_And_Avg_Stretch(
    MESH_ELEM<dim - 1>& Elem, T h,
    const VECTOR<T, 4>& fiberStiffMult,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    T& maxs, T& avgs, T& minc, T& avgc)
{
    TIMER_FLAG("Compute_Max_And_Avg_Stretch");
    maxs = 1.0, avgs = 0.0;
    minc = 1.0, avgc = 0.0;
    int stretchedAmt = 0, compressedAmt = 0;
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F_, vol, lambda, mu] = data;
            if (!(DBCb[elemVInd[0]] && DBCb[elemVInd[1]] && DBCb[elemVInd[2]])) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                const MATRIX<T, dim - 1>& Bsqr = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));

                MATRIX<T, dim - 1> B;
                B(0, 0) = std::sqrt(Bsqr(0, 0));
                B(1, 0) = 0;
                B(0, 1) = Bsqr(0, 1) / B(0, 0);
                B(1, 1) = std::sqrt(Bsqr(1, 1) - Bsqr(0, 1) * Bsqr(0, 1) / Bsqr(0, 0));
                MATRIX<T, dim - 1> A;
                A(0, 0) = (x2 - x1).norm();
                A(1, 0) = 0;
                A(0, 1) = (x2 - x1).dot(x3 - x1) / A(0, 0);
                A(1, 1) = cross(x2 - x1, x3 - x1).norm() / A(0, 0);
                MATRIX<T, dim - 1> F = A * B.inverse();

                MATRIX<T, dim - 1> U(1), V(1);
                VECTOR<T, dim - 1> sigma;
                Singular_Value_Decomposition(F, U, sigma, V);
                T maxsI = sigma[0], mincI = sigma[1];
                if (maxsI < sigma[1]) {
                    maxsI = sigma[1];
                    mincI = sigma[0];
                }

                if (maxsI > maxs) {
                    maxs = maxsI;
                }
                if (maxsI > 1) {
                    ++stretchedAmt;
                    avgs += maxsI;
                }

                if (mincI < minc) {
                    minc = mincI;
                }
                if (mincI < 1) {
                    ++compressedAmt;
                    avgc += mincI;
                }
            }
        });
    }

    if (stretchedAmt) {
        avgs /= stretchedAmt;
    }
    if (compressedAmt) {
        avgc /= compressedAmt;
    }
}

template<class T, int dim>
void Compute_Stretch_From_File(
    const std::string& restShapePath,
    const std::string& folderPath,
    int beginFrame, int endFrame)
{
    MESH_NODE<T, dim> X_rest;
    MESH_ELEM<dim - 1> Elem_rest;
    Read_TriMesh_Obj(restShapePath, X_rest, Elem_rest);

    for (int frameI = beginFrame; frameI <= endFrame; ++frameI) {
        printf("processing frame %d...\n", frameI);

        std::string filePath = folderPath + std::to_string(frameI) + ".obj";

        FILE *in = fopen(filePath.c_str(), "r");
        if (!in) {
            printf("frame %d not found!\n", frameI);
            continue;
        }
        else {
            MESH_NODE<T, dim> X;
            MESH_ELEM<dim - 1> Elem;
            Read_TriMesh_Obj(filePath, X, Elem);
            if (Elem.size != Elem_rest.size || X.size != X_rest.size) {
                printf("frame do not match with rest shape!\n");
                fclose(in);
                exit(-1);
            }

            T maxs = 1.0, avgs = 0.0, minc = 1.0, avgc = 0.0;
            int stretchedAmt = 0, compressedAmt = 0;
            if constexpr (dim == 2) {
                //TODO
            }
            else {
                //TODO: parallelize
                Elem.Each([&](int id, auto data) {
                    auto &[elemVInd] = data;
                    const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                    const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                    const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                    const VECTOR<T, dim>& X1 = std::get<0>(X_rest.Get_Unchecked(elemVInd[0]));
                    const VECTOR<T, dim>& X2 = std::get<0>(X_rest.Get_Unchecked(elemVInd[1]));
                    const VECTOR<T, dim>& X3 = std::get<0>(X_rest.Get_Unchecked(elemVInd[2]));

                    MATRIX<T, dim - 1> B;
                    B(0, 0) = (X2 - X1).norm();
                    B(1, 0) = 0;
                    B(0, 1) = (X2 - X1).dot(X3 - X1) / B(0, 0);
                    B(1, 1) = cross(X2 - X1, X3 - X1).norm() / B(0, 0);
                    MATRIX<T, dim - 1> A;
                    A(0, 0) = (x2 - x1).norm();
                    A(1, 0) = 0;
                    A(0, 1) = (x2 - x1).dot(x3 - x1) / A(0, 0);
                    A(1, 1) = cross(x2 - x1, x3 - x1).norm() / A(0, 0);
                    MATRIX<T, dim - 1> F = A * B.inverse();

                    MATRIX<T, dim - 1> U(1), V(1);
                    VECTOR<T, dim - 1> sigma;
                    Singular_Value_Decomposition(F, U, sigma, V);
                    T maxsI = sigma[0], mincI = sigma[1];
                    if (maxsI < sigma[1]) {
                        maxsI = sigma[1];
                        mincI = sigma[0];
                    }

                    if (maxsI > maxs) {
                        maxs = maxsI;
                    }
                    if (maxsI > 1) {
                        ++stretchedAmt;
                        avgs += maxsI;
                    }

                    if (mincI < minc) {
                        minc = mincI;
                    }
                    if (mincI < 1) {
                        ++compressedAmt;
                        avgc += mincI;
                    }
                });
            }

            if (stretchedAmt) {
                avgs /= stretchedAmt;
            }
            if (compressedAmt) {
                avgc /= compressedAmt;
            }

            int lastSlash = folderPath.find_last_of('/');
            FILE *out = fopen((folderPath.substr(0, lastSlash + 1) + "stretch.txt").c_str(), "a+");
            if (!out) {
                printf("file creation error!\n");
                fclose(in);
                exit(-2);
            }
            else {
                fprintf(out, "%d %le %le %le %le\n", frameI, maxs, avgs, minc, avgc);
                fclose(out);
            }

            fclose(in);
        }
    }
}

template <class T, int dim = 3>
void XZ_As_Texture(
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim - 1>& Elem,
    const std::string& filePath)
{
    FILE *out = fopen(filePath.c_str(), "w+");
    if (!out) {
        printf("cannot create %s\n", filePath.c_str());
        exit(-1);
    }
    else {
        X.Each([&](int id, auto data) {
            auto &[x] = data;
            fprintf(out, "v %le %le %le\n", x[0], x[1], x[2]);
        });
        X.Each([&](int id, auto data) {
            auto &[x] = data;
            fprintf(out, "vt %le %le\n", x[0], x[2]);
        });
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            fprintf(out, "f %d/%d %d/%d %d/%d\n", 
                elemVInd[0] + 1, elemVInd[0] + 1,
                elemVInd[1] + 1, elemVInd[1] + 1, 
                elemVInd[2] + 1, elemVInd[2] + 1);
        });

        fclose(out);
    }
}

template <class T, int dim>
void Construct_Surface_Mesh(
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim - 1>& Elem,
    T thickness,
    MESH_NODE<T, dim>& X_surf,
    MESH_ELEM<dim - 1>& Elem_surf)
{
    if constexpr (dim == 3) {
        std::vector<VECTOR<T, dim>> nodeNormals(X.size, VECTOR<T, dim>(0, 0, 0));
        std::map<VECTOR<int, 2>, int> edgeID;
        std::vector<VECTOR<T, dim>> edgeNormals;
        std::vector<VECTOR<T, dim>> triNormals(Elem.size);
        std::map<VECTOR<int, 3>, int> triID;
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            triID[elemVInd] = id;

            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));

            triNormals[id] = cross(X2 - X1, X3 - X1);

            for (int i = 0; i < 3; ++i) {
                nodeNormals[elemVInd[i]] += triNormals[id];

                int j = (i + 1) % 3;
                auto finder = edgeID.find(VECTOR<int, 2>(elemVInd[j], elemVInd[i]));
                if (finder == edgeID.end()) {
                    edgeID[VECTOR<int, 2>(elemVInd[i], elemVInd[j])] = edgeNormals.size();
                    edgeNormals.emplace_back(triNormals[id]);
                }
                else {
                    edgeNormals[finder->second] += triNormals[id];
                }
            }

            triNormals[id] *= thickness / 2 / triNormals[id].norm();
        });
        for (auto& nnI : nodeNormals) {
            nnI *= thickness / 2 / nnI.norm();
        }
        for (auto& enI : edgeNormals) {
            enI *= thickness / 2 / enI.norm();
        }


        //TODO: boundary triangles
        //TODO: use contact to update vertex position: slackness, unilateral, modularize
        std::vector<int> boundaryNode;
        std::vector<VECTOR<int, 2>> boundaryEdge;
        std::vector<VECTOR<int, 3>> boundaryTri;
        std::vector<T> BNArea, BEArea, BTArea;
        Find_Surface_Primitives_And_Compute_Area(X, Elem, boundaryNode, boundaryEdge, boundaryTri, BNArea, BEArea, BTArea);
        
        MESH_NODE_ATTR<T, dim> nodeAttr(X.size);
        for (int i = 0; i < X.size; ++i) {
            nodeAttr.Append(std::get<0>(X.Get_Unchecked(i)), VECTOR<T, dim>(), VECTOR<T, dim>(), 0);
        }
        std::vector<VECTOR<int, dim + 1>> constraintSet;
        std::vector<VECTOR<int, 2>> constraintSetPTEE;
        std::vector<VECTOR<T, 2>> stencilInfo;
        Compute_Constraint_Set<T, dim, false, true>(X, nodeAttr, boundaryNode, boundaryEdge, boundaryTri,
            std::vector<int>(), std::vector<VECTOR<int, 2>>(), std::map<int, std::set<int>>(), BNArea, BEArea, BTArea,
            VECTOR<int, 2>(boundaryNode.size(), boundaryNode.size()),
            std::vector<bool>(X.size, false), thickness * thickness, T(0), false, constraintSet, constraintSetPTEE, stencilInfo);
        //TODO: rod and particles?

        T minDist2;
        std::vector<T> dist2;
        Compute_Min_Dist2<T, dim, true>(X, constraintSet, thickness, dist2, minDist2);
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            const VECTOR<int, 4>& cIVInd = constraintSet[cI];
            assert(cIVInd[1] >= 0);
            if (cIVInd[0] >= 0) {
                // EE
                if (cIVInd[3] >= 0 && cIVInd[2] >= 0) {
                    // ++++ EE, no mollification
                    // const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                    // const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    // const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    // const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(cIVInd[3]));

                    for (int i = 0; i < 4; ++i) {
                        const T scale = std::sqrt(dist2[cI]) / 2 / nodeNormals[cIVInd[i]].norm();
                        if (scale < 1) {
                            nodeNormals[cIVInd[i]] *= scale;
                        }
                    }
                }
                else {
                    // EE, PE, or PP with mollification
                    std::array<int, 4> edgeVInd;
                    if (cIVInd[3] >= 0) {
                        // ++-+ EE with mollification
                        edgeVInd = {cIVInd[0], cIVInd[1], -cIVInd[2] - 1, cIVInd[3]};
                        // const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        // const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        // const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        // const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));

                        for (int i = 0; i < 4; ++i) {
                            const T scale = std::sqrt(dist2[cI]) / 2 / nodeNormals[edgeVInd[i]].norm();
                            if (scale < 1) {
                                nodeNormals[edgeVInd[i]] *= scale;
                            }
                        }
                    }
                    else if (cIVInd[2] >= 0) {
                        // +++- PE with mollification, multiplicity 1
                        // edgeVInd = {cIVInd[0], -cIVInd[3] - 1, cIVInd[1], cIVInd[2]};
                        // const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        // const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        // const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        // const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));

                        for (int i = 0; i < 3; ++i) {
                            const T scale = std::sqrt(dist2[cI]) / 2 / nodeNormals[cIVInd[i]].norm();
                            if (scale < 1) {
                                nodeNormals[cIVInd[i]] *= scale;
                            }
                        }
                    }
                    else {
                        // ++-- PP with mollification, multiplicity 1
                        // edgeVInd = {cIVInd[0], -cIVInd[2] - 1, cIVInd[1], -cIVInd[3] - 1};
                        // const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        // const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        // const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        // const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));

                        for (int i = 0; i < 2; ++i) {
                            const T scale = std::sqrt(dist2[cI]) / 2 / nodeNormals[cIVInd[i]].norm();
                            if (scale < 1) {
                                nodeNormals[cIVInd[i]] *= scale;
                            }
                        }
                    }
                }
            }
            else {
                // PT, PE, and PP
                int vI = -cIVInd[0] - 1;
                // const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(vI));

                const T scale = std::sqrt(dist2[cI]) / 2 / nodeNormals[vI].norm();
                if (scale < 1) {
                    nodeNormals[vI] *= scale;
                }

                if (cIVInd[3] >= 0) {
                    // -+++ PT 
                    // assert(cIVInd[2] >= 0);
                    // const VECTOR<T, 3>& Xt0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    // const VECTOR<T, 3>& Xt1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    // const VECTOR<T, 3>& Xt2 = std::get<0>(X.Get_Unchecked(cIVInd[3]));

                    for (int i = 1; i < 4; ++i) {
                        const T scale = std::sqrt(dist2[cI]) / 2 / nodeNormals[cIVInd[i]].norm();
                        if (scale < 1) {
                            nodeNormals[cIVInd[i]] *= scale;
                        }
                    }
                }
                else if (cIVInd[2] >= 0) {
                    // -++[-] PE, last digit stores muliplicity
                    // const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    // const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));

                    for (int i = 1; i < 3; ++i) {
                        const T scale = std::sqrt(dist2[cI]) / 2 / nodeNormals[cIVInd[i]].norm();
                        if (scale < 1) {
                            nodeNormals[cIVInd[i]] *= scale;
                        }
                    }
                }
                else {
                    // -+-[-] PP, last digit stores muliplicity
                    // const VECTOR<T, 3>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const T scale = std::sqrt(dist2[cI]) / 2 / nodeNormals[cIVInd[1]].norm();
                    if (scale < 1) {
                        nodeNormals[cIVInd[1]] *= scale;
                    }
                }
            }
        }


        // X_surf = std::move(MESH_NODE<T, dim>((nodeNormals.size() + edgeNormals.size() + triNormals.size()) * 2));
        // X.Each([&](int id, auto data) {
        //     auto &[x] = data;
        //     X_surf.Append(x + nodeNormals[id]);
        //     X_surf.Append(x - nodeNormals[id]);
        // });
        // std::vector<VECTOR<T, dim>> edgeNodes(edgeID.size() * 2);
        // for (const auto enI : edgeID) {
        //     const VECTOR<T, dim>& xa = std::get<0>(X.Get_Unchecked(enI.first[0]));
        //     const VECTOR<T, dim>& xb = std::get<0>(X.Get_Unchecked(enI.first[1]));
        //     edgeNodes[enI.second * 2] = (xa + xb) / 2 + edgeNormals[enI.second];
        //     edgeNodes[enI.second * 2 + 1] = (xa + xb) / 2 - edgeNormals[enI.second];
        // }
        // for (const auto& enI : edgeNodes) {
        //     X_surf.Append(enI);
        // }
        // Elem.Each([&](int id, auto data) {
        //     auto &[elemVInd] = data;

        //     const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
        //     const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
        //     const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));

        //     X_surf.Append((X1 + X2 + X3) / 3 + triNormals[id]);
        //     X_surf.Append((X1 + X2 + X3) / 3 - triNormals[id]);
        // });

        // Elem_surf = std::move(MESH_ELEM<dim - 1>(Elem.size * 12 + 0));
        // Elem.Each([&](int id, auto data) {
        //     auto &[elemVInd] = data;

        //     int vt[3], et[3];
        //     int trit = X.size * 2 + edgeID.size() * 2 + id * 2;
        //     for (int i = 0; i < 3; ++i) {
        //         vt[i] = elemVInd[i] * 2;

        //         int j = (i + 1) % 3;
        //         auto finder = edgeID.find(VECTOR<int, 2>(elemVInd[i], elemVInd[j]));
        //         if (finder == edgeID.end()) {
        //             finder = edgeID.find(VECTOR<int, 2>(elemVInd[j], elemVInd[i]));
        //             if (finder == edgeID.end()) {
        //                 std::cout << "edge record error!" << std::endl;
        //                 exit(-1);
        //             }
        //         }
        //         et[i] = X.size * 2 + finder->second * 2;
        //     }

        //     Elem_surf.Append(VECTOR<int, dim>(vt[0], et[0], trit));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[0], trit, et[2]));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[1], et[1], trit));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[1], trit, et[0]));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[2], et[2], trit));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[2], trit, et[1]));

        //     Elem_surf.Append(VECTOR<int, dim>(vt[0] + 1, trit + 1, et[0] + 1));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[0] + 1, et[2] + 1, trit + 1));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[1] + 1, trit + 1, et[1] + 1));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[1] + 1, et[0] + 1, trit + 1));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[2] + 1, trit + 1, et[2] + 1));
        //     Elem_surf.Append(VECTOR<int, dim>(vt[2] + 1, et[1] + 1, trit + 1));
        // });

        X_surf = std::move(MESH_NODE<T, dim>(nodeNormals.size() * 2));
        X.Each([&](int id, auto data) {
            auto &[x] = data;
            X_surf.Append(x + nodeNormals[id]);
            X_surf.Append(x - nodeNormals[id]);
        });
        Elem_surf = std::move(MESH_ELEM<dim - 1>(Elem.size * 2 + 0));
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            Elem_surf.Append(VECTOR<int, dim>(elemVInd[0] * 2, elemVInd[1] * 2, elemVInd[2] * 2));
            Elem_surf.Append(VECTOR<int, dim>(elemVInd[0] * 2 + 1, elemVInd[2] * 2 + 1, elemVInd[1] * 2 + 1));
        });
    }
}

void Export_Discrete_Shell(py::module& m) {
    py::module shell_m = m.def_submodule("DiscreteShell", "A submodule of JGSL for FEM discrete shell simulation");

    shell_m.def("Add_Garment", &Add_Garment_3D<double>);
    shell_m.def("Add_Shell", &Add_Discrete_Shell_3D<double>);
    shell_m.def("Make_Rod", &Make_Rod<double, 3>);
    shell_m.def("Make_Rod_Net", &Make_Rod_Net<double, 3>);
    shell_m.def("Add_Discrete_Particles", &Add_Discrete_Particles<double, 3>);
    shell_m.def("Initialize_Shell", &Initialize_Discrete_Shell<double, 3, true, false>);
    shell_m.def("Initialize_Garment", &Initialize_Garment<double, 3>);
    shell_m.def("Initialize_Shell_Hinge", &Initialize_Discrete_Shell<double, 3, false, false>);
    shell_m.def("Update_Normal_Flow_Neumann", &Update_Normal_Flow_Neumann<double, 3>);
    shell_m.def("Update_Material_With_Tex_Shell", &Update_Material_With_Tex_Shell<double, 3>);
    shell_m.def("Initialize_Shell_EIPC", &Initialize_Discrete_Shell<double, 3, true, true>);
    shell_m.def("Initialize_Shell_Hinge_EIPC", &Initialize_Discrete_Shell<double, 3, false, true>);
    shell_m.def("Initialize_Discrete_Rod", &Initialize_Discrete_Rod<double, 3>);
    shell_m.def("Initialize_Discrete_Particle", &Initialize_Discrete_Particle<double, 3>);
    shell_m.def("Initialize_EIPC", &Initialize_EIPC<double>);
    shell_m.def("Initialize_OIPC", &Initialize_EIPC<double, false>);
    shell_m.def("Initialize_OIPC_VM", &Initialize_OIPC_VecM<double>);
    // shell_m.def("Init_Dirichlet", &Init_Dirichlet_Shell<double, 3>);
    // shell_m.def("Initialize_Displacement", &Initialize_Displacement<double, 3>);
    shell_m.def("Advance_One_Step_IE", &Advance_One_Step_IE_Discrete_Shell<double, 3, true, false>);
    shell_m.def("Advance_One_Step_IE_Hinge", &Advance_One_Step_IE_Discrete_Shell<double, 3, false, false>);
    shell_m.def("Advance_One_Step_IE_EIPC", &Advance_One_Step_IE_Discrete_Shell<double, 3, true, true>);
    shell_m.def("Advance_One_Step_IE_Hinge_EIPC", &Advance_One_Step_IE_Discrete_Shell<double, 3, false, true>);
    shell_m.def("Advance_One_Step_IE_Flow", &Advance_One_Step_IE_Discrete_Shell<double, 3, false, false, true>);

    shell_m.def("Advance_One_Step_SIE", &Advance_One_Step_SIE_Discrete_Shell<double, 3, true, false>);
    shell_m.def("Advance_One_Step_SIE_Hinge", &Advance_One_Step_SIE_Discrete_Shell<double, 3, false, false>);
    shell_m.def("Advance_One_Step_SIE_EIPC", &Advance_One_Step_SIE_Discrete_Shell<double, 3, true, true>);
    shell_m.def("Advance_One_Step_SIE_Hinge_EIPC", &Advance_One_Step_SIE_Discrete_Shell<double, 3, false, true>);

    shell_m.def("Construct_Surface_Mesh", &Construct_Surface_Mesh<double, 3>);

    shell_m.def("Compute_Stretch_From_File", &Compute_Stretch_From_File<double, 3>);
    shell_m.def("XZ_As_Texture", &XZ_As_Texture<double, 3>);

    shell_m.def("Adjust_Material", &Adjust_Material<double, 3>);
}

}