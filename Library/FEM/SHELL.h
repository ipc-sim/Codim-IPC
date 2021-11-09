#pragma once

#include <Physics/FIXED_COROTATED.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim>
void Compute_Inv_Basis(
    std::vector<T>& gamma,
    std::vector<T>& lambdaq,
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim - 1>& Elem,
    MESH_ELEM_ATTR<T, dim>& elemAttr)
{
    if constexpr (dim == 2) {
        elemAttr.Reserve(Elem.size * gamma.size() * lambdaq.size());
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2));
            const VECTOR<T, dim>& N1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2 + 1));
            const VECTOR<T, dim>& N2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2 + 1));

            for (int lI = 0; lI < lambdaq.size(); ++lI) {
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    MATRIX<T, dim> IB;
                    IB(0, 0) = -X1[0] + X2[0] - gamma[gI] * (N1[0] - N2[0]);
                    IB(1, 0) = -X1[1] + X2[1] - gamma[gI] * (N1[1] - N2[1]);
                    IB(0, 1) = (1 - lambdaq[lI]) * N1[0] + lambdaq[lI] * N2[0];
                    IB(1, 1) = (1 - lambdaq[lI]) * N1[1] + lambdaq[lI] * N2[1];
                    IB.invert();
                    elemAttr.Append(IB, MATRIX<T, dim>());
                }
            }
        });
    }
    else {
        // lambdaq: [lambda1_1, lambda2_1, lambda1_2, lambda_2_2, ...]
        elemAttr.Reserve(Elem.size * gamma.size() * lambdaq.size() / 2);
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2] * 2));
            const VECTOR<T, dim>& N1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2 + 1));
            const VECTOR<T, dim>& N2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2 + 1));
            const VECTOR<T, dim>& N3 = std::get<0>(X.Get_Unchecked(elemVInd[2] * 2 + 1));

            for (int lI = 0; lI < lambdaq.size() / 2; ++lI) {
                T lambda1 = lambdaq[lI * 2];
                T lambda2 = lambdaq[lI * 2 + 1];
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    MATRIX<T, dim> IB;
                    IB(0, 0) = -X1[0] + X2[0] - gamma[gI] * (N1[0] - N2[0]);
                    IB(1, 0) = -X1[1] + X2[1] - gamma[gI] * (N1[1] - N2[1]);
                    IB(2, 0) = -X1[2] + X2[2] - gamma[gI] * (N1[2] - N2[2]);
                    IB(0, 1) = -X1[0] + X3[0] - gamma[gI] * (N1[0] - N3[0]);
                    IB(1, 1) = -X1[1] + X3[1] - gamma[gI] * (N1[1] - N3[1]);
                    IB(2, 1) = -X1[2] + X3[2] - gamma[gI] * (N1[2] - N3[2]);
                    IB(0, 2) = N1[0] + lambda1 * (N2[0] - N1[0]) + lambda2 * (N3[0] - N1[0]);
                    IB(1, 2) = N1[1] + lambda1 * (N2[1] - N1[1]) + lambda2 * (N3[1] - N1[1]);
                    IB(2, 2) = N1[2] + lambda1 * (N2[2] - N1[2]) + lambda2 * (N3[2] - N1[2]);
                    IB.invert();
                    elemAttr.Append(IB, MATRIX<T, dim>());
                }
            }
        });
    }
    std::cout << "IB computed" << std::endl;
}

template <class T, int dim = 2>
void Add_Shell_2D(
    int shellType,
    int numElem, T length, 
    T thickness0, T thickness1,
    VECTOR<T, dim>& trans,
    T rotDeg,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates and normal fibers, # = 2 * (#segments + 1)
    MESH_ELEM<dim - 1>& Elem, // the segments
    MESH_NODE<T, dim>& X_mesh, // for rendering
    MESH_ELEM<dim>& triangles, // for rendering
    MESH_NODE_ATTR<T, dim>& nodeAttr_mesh)
{
    // mesh and node
    auto f = [&](T x) {
        switch (shellType) {
        case 1:
            return 0.1 * std::sin(x / length * M_PI * 4);

        case 2:
            return 0.05 * std::sin(x / length * M_PI * 4);

        case 3:
            return 0.025 * std::sin(x / length * M_PI * 4);

        case 4:
            return 0.0125 * std::sin(x / length * M_PI * 4);

        case 5:
            return 0.00625 * std::sin(x / length * M_PI * 4);

        case 0:
        default:
            return 0.0;
        }
    };
    auto g = [&](T x) {
        switch (shellType) {
        case 1:
            return 0.1 / length * M_PI * 4 * std::cos(x / length * M_PI * 4);

        case 2:
            return 0.05 / length * M_PI * 4 * std::cos(x / length * M_PI * 4);

        case 3:
            return 0.025 / length * M_PI * 4 * std::cos(x / length * M_PI * 4);

        case 4:
            return 0.0125 / length * M_PI * 4 * std::cos(x / length * M_PI * 4);

        case 5:
            return 0.00625 / length * M_PI * 4 * std::cos(x / length * M_PI * 4);

        case 0:
        default:
            return 0.0;
        }
    };

    MATRIX<T, dim> rotMtr;
    rotMtr(0, 0) = rotMtr(1, 1) = std::cos(rotDeg / 180 * M_PI);
    rotMtr(0, 1) = -std::sin(rotDeg / 180 * M_PI);
    rotMtr(1, 0) = -rotMtr(0, 1);

    int newXBegin = X.size;
    int newXMeshBegin = X_mesh.size;
    X.Reserve(X.size + 2 * (numElem + 1));
    X_mesh.Reserve(X_mesh.size + 3 * (numElem + 1));
    Elem.Reserve(Elem.size + numElem);
    triangles.Reserve(triangles.size + numElem * 4);
    VECTOR<T, dim> x0(0, f(0));
    x0 = rotMtr * x0 + trans;
    X.Append(x0); // x
    VECTOR<T, dim> normal(-g(0), 1);
    VECTOR<T, dim> n0 = normal / normal.length() * thickness0 / 2;
    n0 = rotMtr * n0;
    X.Append(n0); // n
    X_mesh.Append(x0 + n0);
    X_mesh.Append(x0);
    X_mesh.Append(x0 - n0);
    const T segLen = length / numElem;
    const T thickInc = (thickness1 - thickness0) / numElem;
    for (int i = 0; i < numElem; ++i) {
        VECTOR<T, dim> xI(segLen * (i + 1), f(segLen * (i + 1)));
        xI = rotMtr * xI + trans;
        X.Append(xI); // x
        VECTOR<T, dim> normal(-g(segLen * (i + 1)), 1);
        VECTOR<T, dim> nI = normal / normal.length() * (thickness0 + thickInc * (i + 1)) / 2;
        nI = rotMtr * nI;
        X.Append(nI); // n

        Elem.Append(VECTOR<int, dim>(newXBegin / 2 + i, newXBegin / 2 + i + 1));

        X_mesh.Append(xI + nI);
        X_mesh.Append(xI);
        X_mesh.Append(xI - nI);
        const int iStart = i * 3 + newXMeshBegin;
        triangles.Append(VECTOR<int, dim + 1>(iStart, iStart + 1, iStart + 4));
        triangles.Append(VECTOR<int, dim + 1>(iStart, iStart + 4, iStart + 3));
        triangles.Append(VECTOR<int, dim + 1>(iStart + 1, iStart + 2, iStart + 4));
        triangles.Append(VECTOR<int, dim + 1>(iStart + 2, iStart + 5, iStart + 4));
    }

    nodeAttr_mesh.Reserve(X_mesh.size);
    for (int i = newXMeshBegin; i < X_mesh.size; ++i) {
        nodeAttr_mesh.Append(std::get<0>(X_mesh.Get_Unchecked(i)), VECTOR<T, dim>(), VECTOR<T, dim>(), 0);
    }
}

template <class T, int dim = 3>
void Add_Shell_3D(
    const std::string& filePath,
    T thickness,
    VECTOR<T, dim>& trans,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates and normal fibers
    MESH_ELEM<dim - 1>& Elem, // the mid-surface triangles
    MESH_NODE<T, dim>& X_mesh, // for rendering
    MESH_ELEM<dim - 1>& triangles, // for rendering
    MESH_NODE_ATTR<T, dim>& nodeAttr_mesh)
{
    MESH_NODE<T, dim> X_mid;
    MESH_ELEM<dim - 1> newElem;
    Read_TriMesh_Obj(filePath, X_mid, newElem);

    int newXBegin = X.size;
    int newXMeshBegin = X_mesh.size;
    X.Reserve(X.size + X_mid.size * 2);
    X_mesh.Reserve(X_mesh.size + X_mid.size * 2); // top and bottom surface nodes
    Elem.Reserve(Elem.size + newElem.size);
    triangles.Reserve(triangles.size + newElem.size * 2); // top and bottom surface triangles 
    // shell boundary faces later
    
    // node allocation and initialization
    X_mid.Each([&](int id, auto data) {
        auto &[XmidI] = data;
        X.Append(XmidI + trans);
        X.Append(VECTOR<T, dim>(0, 0, 0)); // N to be computed later

        X_mesh.Append(VECTOR<T, dim>()); // top
        X_mesh.Append(VECTOR<T, dim>()); // bottom
    });

    // element, rendering triangles, and normal computation
    newElem.Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        Elem.Append(VECTOR<int, 3>(elemVInd[0] + newXBegin / 2,
            elemVInd[1] + newXBegin / 2, elemVInd[2] + newXBegin / 2));

        triangles.Append(VECTOR<int, 3>(newXMeshBegin + elemVInd[0] * 2,
            newXMeshBegin + elemVInd[1] * 2, newXMeshBegin + elemVInd[2] * 2)); // top
        triangles.Append(VECTOR<int, 3>(newXMeshBegin + elemVInd[0] * 2 + 1,
            newXMeshBegin + elemVInd[2] * 2 + 1, newXMeshBegin + elemVInd[1] * 2 + 1)); // bottom, needs reordering
        
        const VECTOR<T, dim>& X1 = std::get<0>(X_mid.Get_Unchecked(elemVInd[0]));
        const VECTOR<T, dim>& X2 = std::get<0>(X_mid.Get_Unchecked(elemVInd[1]));
        const VECTOR<T, dim>& X3 = std::get<0>(X_mid.Get_Unchecked(elemVInd[2]));
        VECTOR<T, dim> normal = cross(X2 - X1, X3 - X1);
        std::get<0>(X.Get_Unchecked(newXBegin + elemVInd[0] * 2 + 1)) += normal;
        std::get<0>(X.Get_Unchecked(newXBegin + elemVInd[1] * 2 + 1)) += normal;
        std::get<0>(X.Get_Unchecked(newXBegin + elemVInd[2] * 2 + 1)) += normal;
    });

    // director and rendering nodes coordinates
    X.Par_Each([&](int id, auto data) {
        if (id >= newXBegin && id % 2 == 1) {
            auto &[N] = data;
            N *= thickness / 2 / N.norm();

            const VECTOR<T, dim>& XI = std::get<0>(X.Get_Unchecked(id - 1));
            std::get<0>(X_mesh.Get_Unchecked(id - newXBegin + newXMeshBegin - 1)) = XI + N; // top
            std::get<0>(X_mesh.Get_Unchecked(id - newXBegin + newXMeshBegin)) = XI - N; // bottom
        }
    });

    // shell boundary faces for rendering and contact
    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    Find_Boundary_Edge_And_Node(X_mid.size, newElem, boundaryNode, boundaryEdge);
    triangles.Reserve(triangles.size + boundaryEdge.size() * 2);
    for (const auto& eI : boundaryEdge) {
        triangles.Append(VECTOR<int, 3>(newXMeshBegin + eI[0] * 2,
            newXMeshBegin + eI[0] * 2 + 1, newXMeshBegin + eI[1] * 2 + 1));
        triangles.Append(VECTOR<int, 3>(newXMeshBegin + eI[0] * 2,
            newXMeshBegin + eI[1] * 2 + 1, newXMeshBegin + eI[1] * 2));
    }

    nodeAttr_mesh.Reserve(X_mesh.size);
    for (int i = newXMeshBegin; i < X_mesh.size; ++i) {
        nodeAttr_mesh.Append(std::get<0>(X_mesh.Get_Unchecked(i)), VECTOR<T, dim>(), VECTOR<T, dim>(), 0);
    }
}

template <class T, int dim>
void Initialize_Shell(
    T rho0, T E, T nu,
    int gammaAmt, int lambdaAmt,
    std::vector<T>& gamma,
    std::vector<T>& lambdaq,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates and normal fibers, # = 2 * (#segments + 1)
    MESH_ELEM<dim - 1>& Elem, // the segments
    MESH_NODE_ATTR<T, dim>& nodeAttr, // # = 2 * (#segments + 1)
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    const VECTOR<T, dim>& gravity,
    std::vector<T>& b, // body force, # = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim>& elemAttr, // # = #segments * gammaAmt * lambdaAmt
    FIXED_COROTATED<T, dim>& elasticityAttr) // # = #segments * gammaAmt * lambdaAmt
{
    if constexpr (dim == 2) {
        // quadrature
        gamma.resize(gammaAmt);
        if (gammaAmt == 2) {
            // use Gaussian quadrature
            gamma[0] = 1.0 / std::sqrt(3.0);
            gamma[1] = -gamma[0];
        }
        else {
            if (gammaAmt % 2) {
                gamma[gammaAmt / 2] = 0;
            }
            for (int i = 0; i < gammaAmt / 2; ++i) {
                gamma[i] = (T)(gammaAmt - 1 - 2 * i) / gammaAmt;
                gamma[gammaAmt - 1 - i] = -gamma[i];
            }
        }
        lambdaq.resize(lambdaAmt);
        if (lambdaAmt == 2) {
            // use Gaussian quadrature
            lambdaq[0] = (-1.0 / std::sqrt(3.0) + 1.0) / 2.0;
            lambdaq[1] = (1.0 / std::sqrt(3.0) + 1.0) / 2.0;
        }
        else {
            lambdaq[0] = (T)1 / (2 * lambdaAmt);
            for (int i = 1; i < lambdaAmt; ++i) {
                lambdaq[i] = lambdaq[0] + 2 * i * lambdaq[0];
            }
        }
        //TODO: support more than 2 Gaussian quadrature

        nodeAttr.Reserve(X.size);
        for (int i = 0; i < X.size; ++i) {
            nodeAttr.Append(std::get<0>(X.Get_Unchecked(i)), VECTOR<T, dim>(0, 0), VECTOR<T, dim>(), 0);
        }

        Compute_Inv_Basis(gamma, lambdaq, X, Elem, elemAttr);

        // mass matrix and body force
        std::vector<Eigen::Triplet<T>> triplets;
        b.resize(0);
        b.resize(X.size * dim, 0);
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2));
            const VECTOR<T, dim>& N1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2 + 1));
            const VECTOR<T, dim>& N2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2 + 1));

            T m[4][4];
            m[0][0] = rho0 / 6 * det(X2 - X1, N2 + N1 * 3);
            m[0][1] = rho0 / 6 * det(X2 - X1, N2 + N1);
            m[0][2] = rho0 * 2 / 9 * det(N2, N1);
            m[0][3] = m[0][2] / 2;
            m[1][0] = m[0][1];
            m[1][1] = rho0 / 6 * det(X2 - X1, N2 * 3 + N1);
            m[1][2] = m[0][3];
            m[1][3] = m[0][2];
            m[2][0] = m[0][2];
            m[2][1] = m[1][2];
            m[2][2] = m[0][0] / 3;
            m[2][3] = m[0][1] / 3;
            m[3][0] = m[0][3];
            m[3][1] = m[1][3];
            m[3][2] = m[2][3];
            m[3][3] = m[1][1] / 3;

            int ind[4] = {elemVInd[0] * 2, elemVInd[1] * 2, elemVInd[0] * 2 + 1, elemVInd[1] * 2 + 1};
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    triplets.emplace_back(ind[i] * dim, ind[j] * dim, m[i][j]);
                    triplets.emplace_back(ind[i] * dim + 1, ind[j] * dim + 1, m[i][j]);
                }
            }

            // body force
            b[ind[0] * dim + 1] += gravity[1] * rho0 / 3 * det(X2 - X1, N2 + N1 * 2);
            b[ind[1] * dim + 1] += gravity[1] * rho0 / 3 * det(X2 - X1, N2 * 2 + N1);
            b[ind[2] * dim + 1] += gravity[1] * rho0 / 3 * det(N2, N1);
            b[ind[3] * dim + 1] += gravity[1] * rho0 / 3 * det(N2, N1);
        });
        M.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
        //NOTE: for implicit, need to project Matrix for Dirichlet boundary condition

        // quadratures
        elasticityAttr.Reserve(Elem.size * gammaAmt * lambdaAmt);
        const T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        const T mu = E / ((T)2 * ((T)1 + nu));
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2));
            const VECTOR<T, dim>& N1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2 + 1));
            const VECTOR<T, dim>& N2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2 + 1));
            for (int lI = 0; lI < lambdaq.size(); ++lI) {
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    T wq = 2.0 / gamma.size(); //TODO: support more than 2 Gaussian quadrature
                    T wp = 1.0 / lambdaq.size(); //TODO: support more than 2 Gaussian quadrature
                    T vol = wp * wq * (det(N2, N1) * gamma[gI] + det(X2 - X1, (N2 - N1) * lambdaq[lI] + N1));
                    elasticityAttr.Append(MATRIX<T, dim>(), vol, lambda, mu);
                }
            }
        });
    }
    else {
        // quadrature
        gamma.resize(gammaAmt);
        if (gammaAmt == 2) {
            // use Gaussian quadrature
            gamma[0] = 1.0 / std::sqrt(3.0);
            gamma[1] = -gamma[0];
        }
        else {
            //TODO: support more than 2 Gaussian quadrature
            if (gammaAmt % 2) {
                gamma[gammaAmt / 2] = 0;
            }
            for (int i = 0; i < gammaAmt / 2; ++i) {
                gamma[i] = (T)(gammaAmt - 1 - 2 * i) / gammaAmt;
                gamma[gammaAmt - 1 - i] = -gamma[i];
            }
        }
        lambdaAmt = 1;//TODO: lambdaAmt gaussian quadrature for triangles
        lambdaq.resize(lambdaAmt * 2);
        lambdaq[0] = lambdaq[1] = 1.0 / 3.0;
        // lambdaAmt = 2;
        // lambdaq.resize(4);
        // lambdaq[0] = 0.5; lambdaq[1] = 1.0 / 6.0;
        // lambdaq[2] = 1.0 / 6.0; lambdaq[3] = 0.5;
        // lambdaAmt = 3;
        // lambdaq.resize(6);
        // lambdaq[0] = lambdaq[1] = 5.0 / 24.0;
        // lambdaq[2] = 5.0 / 24.0; lambdaq[3] = 7.0 / 12.0;
        // lambdaq[4] = 7.0 / 12.0; lambdaq[5] = 5.0 / 24.0;

        nodeAttr.Reserve(X.size);
        for (int i = 0; i < X.size; ++i) {
            nodeAttr.Append(std::get<0>(X.Get_Unchecked(i)), VECTOR<T, dim>(0, 0), VECTOR<T, dim>(), 0);
        }

        Compute_Inv_Basis(gamma, lambdaq, X, Elem, elemAttr);

        // mass matrix and body force
        std::vector<Eigen::Triplet<T>> triplets;
        b.resize(0);
        b.resize(X.size * dim, 0);
        //TODO: separate write conflict and parallelize
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2] * 2));
            const VECTOR<T, dim>& N1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2 + 1));
            const VECTOR<T, dim>& N2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2 + 1));
            const VECTOR<T, dim>& N3 = std::get<0>(X.Get_Unchecked(elemVInd[2] * 2 + 1));

            const VECTOR<T, dim> X12cX13 = cross(X2 - X1, X3 - X1);
            const T X12cX13dN3 = X12cX13.dot(N3);
            const T X12cX13dN2 = X12cX13.dot(N2);
            const T X12cX13dN1 = X12cX13.dot(N1);
            const T N2cN3dN1 = cross(N2, N3).dot(N1);
            const T X12cN3dN1 = cross(X2 - X1, N3).dot(N1);
            const T X13cN2dN1 = cross(X3 - X1, N2).dot(N1);
            const T X12cN13dN2 = cross(X2 - X1, N3 - N1).dot(N2);
            const T X13cN12dN3 = cross(X3 - X1, N2 - N1).dot(N3);

            T m[6][6]; // symmetric
            m[0][0] = rho0 / 90 * (3 * (X12cX13dN3 + X12cX13dN2 + 3 * X12cX13dN1) + 5 * N2cN3dN1);
            m[0][1] = rho0 / 180 * (3 * (X12cX13dN3 + 2 * X12cX13dN2 + 2 * X12cX13dN1) + 5 * N2cN3dN1);
            m[0][2] = rho0 / 180 * (3 * (2 * X12cX13dN3 + X12cX13dN2 + 2 * X12cX13dN1) + 5 * N2cN3dN1);
            m[0][3] = rho0 / 18 * (X12cN3dN1 + X13cN2dN1);
            m[0][4] = rho0 / 180 * (4 * X12cN3dN1 + 5 * X13cN2dN1 + X12cN13dN2);
            m[0][5] = rho0 / 180 * (5 * X12cN3dN1 + 4 * X13cN2dN1 + X13cN12dN3);
            
            m[1][1] = rho0 / 90 * (3 * (X12cX13dN3 + 3 * X12cX13dN2 + X12cX13dN1) + 5 * N2cN3dN1);
            m[1][2] = rho0 / 180 * (3 * (2 * X12cX13dN3 + 2 * X12cX13dN2 + X12cX13dN1) + 5 * N2cN3dN1);
            m[1][3] = m[0][4];
            m[1][4] = rho0 / 90 * (3 * X12cN3dN1 + 5 * X13cN2dN1 + 2 * X12cN13dN2);
            m[1][5] = m[0][3] / 2;

            m[2][2] = rho0 / 90 * (3 * (3 * X12cX13dN3 + X12cX13dN2 + X12cX13dN1) + 5 * N2cN3dN1);
            m[2][3] = m[0][5];
            m[2][4] = m[1][5];
            m[2][5] = rho0 / 90 * (5 * X12cN3dN1 + 3 * X13cN2dN1 + 2 * X13cN12dN3);

            m[3][3] = rho0 / 90 * ((X12cX13dN3 + X12cX13dN2 + 3 * X12cX13dN1) + 3 * N2cN3dN1);
            m[3][4] = rho0 / 180 * ((X12cX13dN3 + 2 * X12cX13dN2 + 2 * X12cX13dN1) + 3 * N2cN3dN1);
            m[3][5] = rho0 / 180 * ((2 * X12cX13dN3 + X12cX13dN2 + 2 * X12cX13dN1) + 3 * N2cN3dN1);

            m[4][4] = rho0 / 90 * ((X12cX13dN3 + 3 * X12cX13dN2 + X12cX13dN1) + 3 * N2cN3dN1);
            m[4][5] = rho0 / 180 * ((2 * X12cX13dN3 + 2 * X12cX13dN2 + X12cX13dN1) + 3 * N2cN3dN1);

            m[5][5] = rho0 / 90 * ((3 * X12cX13dN3 + X12cX13dN2 + X12cX13dN1) + 3 * N2cN3dN1);

            int ind[6] = {elemVInd[0] * 2, elemVInd[1] * 2, elemVInd[2] * 2,
                elemVInd[0] * 2 + 1, elemVInd[1] * 2 + 1, elemVInd[2] * 2 + 1};
            for (int i = 0; i < 6; ++i) {
                triplets.emplace_back(ind[i] * dim, ind[i] * dim, m[i][i]);
                triplets.emplace_back(ind[i] * dim + 1, ind[i] * dim + 1, m[i][i]);
                triplets.emplace_back(ind[i] * dim + 2, ind[i] * dim + 2, m[i][i]);
                for (int j = i + 1; j < 6; ++j) {
                    triplets.emplace_back(ind[i] * dim, ind[j] * dim, m[i][j]);
                    triplets.emplace_back(ind[i] * dim + 1, ind[j] * dim + 1, m[i][j]);
                    triplets.emplace_back(ind[i] * dim + 2, ind[j] * dim + 2, m[i][j]);

                    triplets.emplace_back(ind[j] * dim, ind[i] * dim, m[i][j]);
                    triplets.emplace_back(ind[j] * dim + 1, ind[i] * dim + 1, m[i][j]);
                    triplets.emplace_back(ind[j] * dim + 2, ind[i] * dim + 2, m[i][j]);
                }
            }

            // body force
            const T X12cN13dN12 = cross(X2 - X1, N3 - N1).dot(N2 - N1);
            const T X13cN12dN13 = cross(X3 - X1, N2 - N1).dot(N3 - N1);
            b[ind[0] * dim + 1] += gravity[1] * rho0 / 36 * (3 * (X12cX13dN3 + X12cX13dN2 + 2 * X12cX13dN1) + 4 * N2cN3dN1);
            b[ind[1] * dim + 1] += gravity[1] * rho0 / 36 * (3 * (X12cX13dN3 + 2 * X12cX13dN2 + X12cX13dN1) + 4 * N2cN3dN1);
            b[ind[2] * dim + 1] += gravity[1] * rho0 / 36 * (3 * (2 * X12cX13dN3 + X12cX13dN2 + X12cX13dN1) + 4 * N2cN3dN1);
            b[ind[3] * dim + 1] += gravity[1] * rho0 / 9 * (X12cN3dN1 + X13cN2dN1);
            b[ind[4] * dim + 1] += gravity[1] * rho0 / 36 * (4 * (X12cN3dN1 + X13cN2dN1) + X12cN13dN12);
            b[ind[5] * dim + 1] += gravity[1] * rho0 / 36 * (4 * (X12cN3dN1 + X13cN2dN1) + X13cN12dN13);
        });
        M.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
        //NOTE: for implicit, need to project Matrix for Dirichlet boundary condition

        // quadratures
        elasticityAttr.Reserve(Elem.size * gammaAmt * lambdaAmt);
        const T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        const T mu = E / ((T)2 * ((T)1 + nu));
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2));
            const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2));
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked(elemVInd[2] * 2));
            const VECTOR<T, dim>& N1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2 + 1));
            const VECTOR<T, dim>& N2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2 + 1));
            const VECTOR<T, dim>& N3 = std::get<0>(X.Get_Unchecked(elemVInd[2] * 2 + 1));
            for (int lI = 0; lI < lambdaq.size() / 2; ++lI) {
                T lambda1 = lambdaq[lI * 2];
                T lambda2 = lambdaq[lI * 2 + 1];
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    T wq = 2.0 / gamma.size(); //TODO: support more than 2 Gaussian quadrature
                    T wp = 1.0 / (lambdaq.size() / 2) / 2.0; //TODO: support more than 2 Gaussian quadrature
                    T vol = wp * wq * cross(X2 - X1 + gamma[gI] * (N2 - N1), X3 - X1 + gamma[gI] * (N3 - N1)).dot(
                        N1 + lambda1 * (N2 - N1) + lambda2 * (N3 - N1));
                    elasticityAttr.Append(MATRIX<T, dim>(), vol, lambda, mu);
                }
            }
        });
    }
    std::cout << "shell initialized" << std::endl;
}

template <class T, int dim>
void Init_Dirichlet_Shell(
    MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& X_mesh,
    const VECTOR<T, dim>& relBoxMin,
    const VECTOR<T, dim>& relBoxMax,
    const VECTOR<T, dim>& v,
    const VECTOR<T, dim>& rotCenter,
    const VECTOR<T, dim>& rotAxis,
    T angVelDeg,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    DBC_MOTION<T, dim>& DBCMotion)
{
    VECTOR_STORAGE<T, dim + 1> DBC_mesh;
    DBC_MOTION<T, dim> DBCMotion_mesh;
    Init_Dirichlet(X_mesh, relBoxMin, relBoxMax, v, rotCenter, rotAxis, angVelDeg, DBC_mesh, DBCMotion_mesh);
    
    if constexpr (dim == 2) {
        int DBCSize0 = DBC.size;
        DBC_mesh.Each([&](int id, auto data) {
            const auto &[dbcI] = data;
            if (int(dbcI[0]) % 3 == 1) {
                int xI = int(dbcI[0]) / 3 * 2;
                const VECTOR<T, dim>& x = std::get<0>(X.Get_Unchecked(xI));
                DBC.Append(VECTOR<T, dim + 1>(xI, x[0], x[1]));
            }
        });
        DBCMotion.Append(VECTOR<int, 2>(DBCSize0, DBC.size), v, rotCenter, rotAxis, angVelDeg);

        DBCSize0 = DBC.size;
        DBC_mesh.Each([&](int id, auto data) {
            const auto &[dbcI] = data;
            if (int(dbcI[0]) % 3 == 1) {
                int xI = int(dbcI[0]) / 3 * 2;
                const VECTOR<T, dim>& n = std::get<0>(X.Get_Unchecked(xI + 1));
                DBC.Append(VECTOR<T, dim + 1>(xI + 1, n[0], n[1]));
            }
        });
        DBCMotion.Append(VECTOR<int, 2>(DBCSize0, DBC.size), VECTOR<T, dim>(0), 
            VECTOR<T, dim>(0), rotAxis, angVelDeg);
    }
    else {
        int DBCSize0 = DBC.size;
        DBC_mesh.Each([&](int id, auto data) {
            const auto &[dbcI] = data;
            if (int(dbcI[0]) % 2 == 0) {
                const VECTOR<T, dim>& x = std::get<0>(X.Get_Unchecked(int(dbcI[0])));
                DBC.Append(VECTOR<T, dim + 1>(dbcI[0], x[0], x[1], x[2]));
            }
        });
        DBCMotion.Append(VECTOR<int, 2>(DBCSize0, DBC.size), v, rotCenter, rotAxis, angVelDeg);

        DBCSize0 = DBC.size;
        DBC_mesh.Each([&](int id, auto data) {
            const auto &[dbcI] = data;
            if (int(dbcI[0]) % 2 == 0) {
                const VECTOR<T, dim>& n = std::get<0>(X.Get_Unchecked(int(dbcI[0]) + 1));
                DBC.Append(VECTOR<T, dim + 1>(dbcI[0] + 1, n[0], n[1], n[2]));
            }
        });
        DBCMotion.Append(VECTOR<int, 2>(DBCSize0, DBC.size), VECTOR<T, dim>(0), 
            VECTOR<T, dim>(0), rotAxis, angVelDeg);
    }
}

template<class T, int dim>
void Initialize_Displacement(
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim - 1>& Elem,
    int caseI)
{
    switch(caseI) {
    case 0:
        break;

    case 1:
        X.Par_Each([&](int id, auto data) {
            if (id % 2 == 0) {
                auto &[x] = data;
                x[0] *= 1.5;
            }
        });
        break;

    case 2:
        X.Par_Each([&](int id, auto data) {
            if (id % 2 == 0) {
                auto &[x] = data;
                x[1] += x[0] * 0.1;
            }
        });
        break;

    default:
        break;
    }
}

template <class T, int dim>
void Compute_Deformation_Gradient(
    std::vector<T>& gamma,
    std::vector<T>& lambdaq,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates and normal fibers, # = 2 * (#segments + 1)
    MESH_ELEM<dim - 1>& Elem, // the segments
    MESH_ELEM_ATTR<T, dim>& elemAttr, // # = #segments * gammaAmt * lambdaAmt
    FIXED_COROTATED<T, dim>& elasticityAttr) // # = #segments * gammaAmt * lambdaAmt
{
    TIMER_FLAG("Compute_Deformation_Gradient");

    if constexpr (dim == 2) {
        Elem.Par_Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2));
            const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2));
            const VECTOR<T, dim>& n1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2 + 1));
            const VECTOR<T, dim>& n2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2 + 1));
            
            for (int lI = 0; lI < lambdaq.size(); ++lI) {
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    MATRIX<T, dim>& F = std::get<FIELDS<FIXED_COROTATED<T, dim>>::F>(elasticityAttr.Get_Unchecked(
                        id * gamma.size() * lambdaq.size() + lI * gamma.size() + gI));
                    MATRIX<T, dim>& IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(
                        id * gamma.size() * lambdaq.size() + lI * gamma.size() + gI));
                    
                    F(0, 0) = -x1[0] + x2[0] - gamma[gI] * (n1[0] - n2[0]);
                    F(1, 0) = -x1[1] + x2[1] - gamma[gI] * (n1[1] - n2[1]);
                    F(0, 1) = (1 - lambdaq[lI]) * n1[0] + lambdaq[lI] * n2[0];
                    F(1, 1) = (1 - lambdaq[lI]) * n1[1] + lambdaq[lI] * n2[1];
                    F = F * IB;
                }
            }
        });
    }
    else {
        // lambdaq: [lambda1_1, lambda2_1, lambda1_2, lambda_2_2, ...]
        Elem.Par_Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2));
            const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2));
            const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2] * 2));
            const VECTOR<T, dim>& n1 = std::get<0>(X.Get_Unchecked(elemVInd[0] * 2 + 1));
            const VECTOR<T, dim>& n2 = std::get<0>(X.Get_Unchecked(elemVInd[1] * 2 + 1));
            const VECTOR<T, dim>& n3 = std::get<0>(X.Get_Unchecked(elemVInd[2] * 2 + 1));

            for (int lI = 0; lI < lambdaq.size() / 2; ++lI) {
                T lambda1 = lambdaq[lI * 2];
                T lambda2 = lambdaq[lI * 2 + 1];
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    MATRIX<T, dim>& F = std::get<FIELDS<FIXED_COROTATED<T, dim>>::F>(elasticityAttr.Get_Unchecked(
                        id * gamma.size() * lambdaq.size() / 2 + lI * gamma.size() + gI));
                    MATRIX<T, dim>& IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(
                        id * gamma.size() * lambdaq.size() / 2 + lI * gamma.size() + gI));
                    
                    F(0, 0) = -x1[0] + x2[0] - gamma[gI] * (n1[0] - n2[0]);
                    F(1, 0) = -x1[1] + x2[1] - gamma[gI] * (n1[1] - n2[1]);
                    F(2, 0) = -x1[2] + x2[2] - gamma[gI] * (n1[2] - n2[2]);
                    F(0, 1) = -x1[0] + x3[0] - gamma[gI] * (n1[0] - n3[0]);
                    F(1, 1) = -x1[1] + x3[1] - gamma[gI] * (n1[1] - n3[1]);
                    F(2, 1) = -x1[2] + x3[2] - gamma[gI] * (n1[2] - n3[2]);
                    F(0, 2) = n1[0] + lambda1 * (n2[0] - n1[0]) + lambda2 * (n3[0] - n1[0]);
                    F(1, 2) = n1[1] + lambda1 * (n2[1] - n1[1]) + lambda2 * (n3[1] - n1[1]);
                    F(2, 2) = n1[2] + lambda1 * (n2[2] - n1[2]) + lambda2 * (n3[2] - n1[2]);
                    F = F * IB;
                }
            }
        });
    }
}

template <class T, int dim>
void Compute_Elasticity_Gradient_From_Stress(
    std::vector<T>& gamma,
    std::vector<T>& lambdaq,
    MESH_ELEM<dim - 1>& Elem, // the segments
    MESH_ELEM_ATTR<T, dim>& elemAttr, // # = #segments * gammaAmt * lambdaAmt
    MESH_NODE_ATTR<T, dim>& nodeAttr) // # = #segments * gammaAmt * lambdaAmt)
{
    TIMER_FLAG("Compute_Elasticity_Gradient_From_Stress");

    if constexpr (dim == 2) {
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            VECTOR<T, dim>& g_x1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[0] * 2));
            VECTOR<T, dim>& g_x2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[1] * 2));
            VECTOR<T, dim>& g_n1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[0] * 2 + 1));
            VECTOR<T, dim>& g_n2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[1] * 2 + 1));
            
            for (int lI = 0; lI < lambdaq.size(); ++lI) {
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    MATRIX<T, dim>& P = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(
                        id * gamma.size() * lambdaq.size() + lI * gamma.size() + gI));
                    MATRIX<T, dim>& IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(
                        id * gamma.size() * lambdaq.size() + lI * gamma.size() + gI));

                    g_x1[0] += -IB(0, 0) * P(0, 0) - IB(0, 1) * P(0, 1);
                    g_x1[1] += -IB(0, 0) * P(1, 0) - IB(0, 1) * P(1, 1);
                    g_x2[0] += IB(0, 0) * P(0, 0) + IB(0, 1) * P(0, 1);
                    g_x2[1] += IB(0, 0) * P(1, 0) + IB(0, 1) * P(1, 1);
                    g_n1[0] += (-gamma[gI] * IB(0, 0) + (1 - lambdaq[lI]) * IB(1, 0)) * P(0, 0) + (-gamma[gI] * IB(0, 1) + (1 - lambdaq[lI]) * IB(1, 1)) * P(0, 1);
                    g_n1[1] += (-gamma[gI] * IB(0, 0) + (1 - lambdaq[lI]) * IB(1, 0)) * P(1, 0) + (-gamma[gI] * IB(0, 1) + (1 - lambdaq[lI]) * IB(1, 1)) * P(1, 1);
                    g_n2[0] += (gamma[gI] * IB(0, 0) + lambdaq[lI] * IB(1, 0)) * P(0, 0) + (gamma[gI] * IB(0, 1) + lambdaq[lI] * IB(1, 1)) * P(0, 1);
                    g_n2[1] += (gamma[gI] * IB(0, 0) + lambdaq[lI] * IB(1, 0)) * P(1, 0) + (gamma[gI] * IB(0, 1) + lambdaq[lI] * IB(1, 1)) * P(1, 1);
                }
            }
        });
    }
    else {
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            VECTOR<T, dim>& g_x1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[0] * 2));
            VECTOR<T, dim>& g_x2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[1] * 2));
            VECTOR<T, dim>& g_x3 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[2] * 2));
            VECTOR<T, dim>& g_n1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[0] * 2 + 1));
            VECTOR<T, dim>& g_n2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[1] * 2 + 1));
            VECTOR<T, dim>& g_n3 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[2] * 2 + 1));
            
            for (int lI = 0; lI < lambdaq.size() / 2; ++lI) {
                T lambda1 = lambdaq[lI * 2];
                T lambda2 = lambdaq[lI * 2 + 1];
                T lambda3 = 1 - lambda1 - lambda2;
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    MATRIX<T, dim>& P = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(
                        id * gamma.size() * lambdaq.size() / 2 + lI * gamma.size() + gI));
                    MATRIX<T, dim>& IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(
                        id * gamma.size() * lambdaq.size() / 2 + lI * gamma.size() + gI));

                    const T g_x20 = IB(0, 0) * P(0, 0) + IB(0, 1) * P(0, 1) + IB(0, 2) * P(0, 2);
                    const T g_x21 = IB(0, 0) * P(1, 0) + IB(0, 1) * P(1, 1) + IB(0, 2) * P(1, 2);
                    const T g_x22 = IB(0, 0) * P(2, 0) + IB(0, 1) * P(2, 1) + IB(0, 2) * P(2, 2);
                    const T g_x30 = IB(1, 0) * P(0, 0) + IB(1, 1) * P(0, 1) + IB(1, 2) * P(0, 2);
                    const T g_x31 = IB(1, 0) * P(1, 0) + IB(1, 1) * P(1, 1) + IB(1, 2) * P(1, 2);
                    const T g_x32 = IB(1, 0) * P(2, 0) + IB(1, 1) * P(2, 1) + IB(1, 2) * P(2, 2);
                    g_x1[0] += -g_x20 - g_x30;
                    g_x1[1] += -g_x21 - g_x31;
                    g_x1[2] += -g_x22 - g_x32;
                    g_x2[0] += g_x20;
                    g_x2[1] += g_x21;
                    g_x2[2] += g_x22;
                    g_x3[0] += g_x30;
                    g_x3[1] += g_x31;
                    g_x3[2] += g_x32;
                    
                    const T gIB00_p_l1IB20 = gamma[gI] * IB(0, 0) + lambda1 * IB(2, 0);
                    const T gIB01_p_l1IB21 = gamma[gI] * IB(0, 1) + lambda1 * IB(2, 1);
                    const T gIB02_p_l1IB22 = gamma[gI] * IB(0, 2) + lambda1 * IB(2, 2);
                    const T g_n20 = gIB00_p_l1IB20 * P(0, 0) + gIB01_p_l1IB21 * P(0, 1) + gIB02_p_l1IB22 * P(0, 2);
                    const T g_n21 = gIB00_p_l1IB20 * P(1, 0) + gIB01_p_l1IB21 * P(1, 1) + gIB02_p_l1IB22 * P(1, 2);
                    const T g_n22 = gIB00_p_l1IB20 * P(2, 0) + gIB01_p_l1IB21 * P(2, 1) + gIB02_p_l1IB22 * P(2, 2);

                    const T gIB10_p_l2IB20 = gamma[gI] * IB(1, 0) + lambda2 * IB(2, 0);
                    const T gIB11_p_l2IB21 = gamma[gI] * IB(1, 1) + lambda2 * IB(2, 1);
                    const T gIB12_p_l2IB22 = gamma[gI] * IB(1, 2) + lambda2 * IB(2, 2);
                    const T g_n30 = gIB10_p_l2IB20 * P(0, 0) + gIB11_p_l2IB21 * P(0, 1) + gIB12_p_l2IB22 * P(0, 2);
                    const T g_n31 = gIB10_p_l2IB20 * P(1, 0) + gIB11_p_l2IB21 * P(1, 1) + gIB12_p_l2IB22 * P(1, 2);
                    const T g_n32 = gIB10_p_l2IB20 * P(2, 0) + gIB11_p_l2IB21 * P(2, 1) + gIB12_p_l2IB22 * P(2, 2);

                    g_n1[0] += IB(2, 0) * P(0, 0) + IB(2, 1) * P(0, 1) + IB(2, 2) * P(0, 2) - g_n20 - g_n30;
                    g_n1[1] += IB(2, 0) * P(1, 0) + IB(2, 1) * P(1, 1) + IB(2, 2) * P(1, 2) - g_n21 - g_n31;
                    g_n1[2] += IB(2, 0) * P(2, 0) + IB(2, 1) * P(2, 1) + IB(2, 2) * P(2, 2) - g_n22 - g_n32;
                    g_n2[0] += g_n20;
                    g_n2[1] += g_n21;
                    g_n2[2] += g_n22;
                    g_n3[0] += g_n30;
                    g_n3[1] += g_n31;
                    g_n3[2] += g_n32;
                }
            }
        });
    }
}

template <class T, int dim>
void Compute_IncPotential(
    T h, MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xtilde,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr,
    MESH_NODE<T, dim>& X_mesh,
    MESH_NODE_ATTR<T, dim>& nodeAttr_mesh,
    bool withCollision,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    T dHat2, T kappa[],
    bool staticSolve,
    const std::vector<T>& b,
    double& value)
{
    TIMER_FLAG("computeIncPotential");
    value = 0;

    // elasticity
    FIXED_COROTATED_FUNCTOR<T, dim>::Compute_Psi(elasticityAttr, 
        staticSolve ? 1.0 : (h * h), elemAttr, value);
    
    if (staticSolve) {
        std::vector<T> xb(X.size, T(0));
        X.Par_Each([&](int id, auto data) {
            auto &[x] = data;
            xb[id] += x[0] * b[id * dim];
            xb[id] += x[1] * b[id * dim + 1];
            if constexpr (dim == 3) {
                xb[id] += x[2] * b[id * dim + 2];
            }
        });
        value -= std::accumulate(xb.begin(), xb.end(), T(0));
    }
    else {
        // inertia
        Eigen::VectorXd xDiff(X.size * dim);
        X.Join(Xtilde).Par_Each([&](int id, auto data) {
            auto &[x, xtilde] = data;
            xDiff[id * dim] = x[0] - xtilde[0];
            xDiff[id * dim + 1] = x[1] - xtilde[1];
            if constexpr (dim == 3) {
                xDiff[id * dim + 2] = x[2] - xtilde[2];
            }
        });
        Eigen::VectorXd MXDiff = M.Get_Matrix() * xDiff;
        value += 0.5 * MXDiff.dot(xDiff);
    }

    if (withCollision) {
        // IPC
        Compute_Barrier(X_mesh, nodeAttr_mesh, constraintSet, 
            std::vector<VECTOR<T, 2>>(constraintSet.size(), VECTOR<T, 2>(1, dHat2)),
            dHat2, kappa, T(0), value);
    }
}

template <class T, int dim>
void Compute_IncPotential_Gradient(
    std::vector<T>& gamma,
    std::vector<T>& lambdaq,
    MESH_ELEM<dim - 1>& Elem,
    T h, MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xtilde,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 2 * dim * (#segments + 1)
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    MESH_NODE<T, dim>& X_mesh,
    MESH_NODE_ATTR<T, dim>& nodeAttr_mesh,
    bool withCollision,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    T dHat2, T kappa[],
    bool staticSolve,
    const std::vector<T>& b,
    FIXED_COROTATED<T, dim>& elasticityAttr)
{
    TIMER_FLAG("computeIncPotentialGradient");
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));

    // elasticity
    FIXED_COROTATED_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(elasticityAttr, 
        staticSolve ? 1.0 : (h * h), elemAttr);
    Compute_Elasticity_Gradient_From_Stress(gamma, lambdaq, Elem, elemAttr, nodeAttr); // g initialized inside
    
    // inertia
    if (staticSolve) {
        nodeAttr.Par_Each([&](int id, auto data){
            auto &[x0, v, g, m] = data;
            g[0] -= b[id * dim];
            g[1] -= b[id * dim + 1];
            if constexpr (dim == 3) {
                g[2] -= b[id * dim + 2];
            }
        });
    }
    else {
        Eigen::VectorXd xDiff(X.size * dim);
        X.Join(Xtilde).Par_Each([&](int id, auto data) {
            auto &[x, xtilde] = data;
            xDiff[id * dim] = x[0] - xtilde[0];
            xDiff[id * dim + 1] = x[1] - xtilde[1];
            if constexpr (dim == 3) {
                xDiff[id * dim + 2] = x[2] - xtilde[2];
            }
        });
        Eigen::VectorXd MXDiff = M.Get_Matrix() * xDiff;
        nodeAttr.Par_Each([&](int id, auto data){
            auto &[x0, v, g, m] = data;
            g[0] += MXDiff[id * dim];
            g[1] += MXDiff[id * dim + 1];
            if constexpr (dim == 3) {
                g[2] += MXDiff[id * dim + 2];
            }
        });
    }

    if (withCollision) {
        // IPC
        Compute_Barrier_Gradient_Shell(X_mesh, nodeAttr_mesh, constraintSet, dHat2, kappa, nodeAttr);
    }
}

template <class T, int dim>
void Compute_Elasticity_Hessian(
    std::vector<T>& gamma,
    std::vector<T>& lambdaq, 
    MESH_ELEM<dim - 1>& Elem, T h,
    bool projectSPD,
    MESH_ELEM_ATTR<T, dim>& elemAttr, // # = #segments * gammaAmt * lambdaAmt
    FIXED_COROTATED<T, dim>& elasticityAttr, // # = #segments * gammaAmt * lambdaAmt
    std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Elasticity_Hessian");

    typename FIXED_COROTATED_FUNCTOR<T, dim>::DIFFERENTIAL dP_div_dF;
    for (int i = 0; i < elemAttr.size; ++i) {
        dP_div_dF.Insert(i, Eigen::Matrix<T, dim * dim, dim * dim>::Zero());
    }
    FIXED_COROTATED_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress_Derivative(elasticityAttr, h * h, projectSPD, dP_div_dF);
    
    if constexpr (dim == 2) {
        triplets.resize(Elem.size * gamma.size() * lambdaq.size() * 64);
        Elem.Par_Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            int indMap[4 * dim] = {
                elemVInd[0] * 2 * dim,
                elemVInd[0] * 2 * dim + 1,
                elemVInd[1] * 2 * dim,
                elemVInd[1] * 2 * dim + 1,
                (elemVInd[0] * 2 + 1) * dim,
                (elemVInd[0] * 2 + 1) * dim + 1,
                (elemVInd[1] * 2 + 1) * dim,
                (elemVInd[1] * 2 + 1) * dim + 1,
            };
            for (int lI = 0; lI < lambdaq.size(); ++lI) {
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    int qI = id * gamma.size() * lambdaq.size() + lI * gamma.size() + gI;
                    const auto& dPdF = std::get<0>(dP_div_dF.Get_Unchecked(qI));
                    const MATRIX<T, dim>& IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(qI));

                    // compute dfdx^T*dpdf*dfdx and make triplet
                    T intermediate[4 * dim][dim * dim];
                    for (int colI = 0; colI < dim * dim; ++colI) {
                        intermediate[0][colI] = -IB(0, 0) * dPdF(0, colI) - IB(0, 1) * dPdF(2, colI);
                        intermediate[1][colI] = -IB(0, 0) * dPdF(1, colI) - IB(0, 1) * dPdF(3, colI);
                        intermediate[2][colI] = IB(0, 0) * dPdF(0, colI) + IB(0, 1) * dPdF(2, colI);
                        intermediate[3][colI] = IB(0, 0) * dPdF(1, colI) + IB(0, 1) * dPdF(3, colI);
                        intermediate[4][colI] = (-gamma[gI] * IB(0, 0) + (1 - lambdaq[lI]) * IB(1, 0)) * dPdF(0, colI) +
                            (-gamma[gI] * IB(0, 1) + (1 - lambdaq[lI]) * IB(1, 1)) * dPdF(2, colI);
                        intermediate[5][colI] = (-gamma[gI] * IB(0, 0) + (1 - lambdaq[lI]) * IB(1, 0)) * dPdF(1, colI) +
                            (-gamma[gI] * IB(0, 1) + (1 - lambdaq[lI]) * IB(1, 1)) * dPdF(3, colI);
                        intermediate[6][colI] = (gamma[gI] * IB(0, 0) + lambdaq[lI] * IB(1, 0)) * dPdF(0, colI) +
                            (gamma[gI] * IB(0, 1) + lambdaq[lI] * IB(1, 1)) * dPdF(2, colI);
                        intermediate[7][colI] = (gamma[gI] * IB(0, 0) + lambdaq[lI] * IB(1, 0)) * dPdF(1, colI) +
                            (gamma[gI] * IB(0, 1) + lambdaq[lI] * IB(1, 1)) * dPdF(3, colI);
                    }
                    for (int rowI = 0; rowI < 4 * dim; ++rowI) {
                        int tIBegin = qI * 64 + rowI * 8;
                        triplets[tIBegin] = Eigen::Triplet<T>(indMap[rowI], indMap[0], -IB(0, 0) * intermediate[rowI][0] - IB(0, 1) * intermediate[rowI][2]);
                        triplets[tIBegin + 1] = Eigen::Triplet<T>(indMap[rowI], indMap[1], -IB(0, 0) * intermediate[rowI][1] - IB(0, 1) * intermediate[rowI][3]);
                        triplets[tIBegin + 2] = Eigen::Triplet<T>(indMap[rowI], indMap[2], IB(0, 0) * intermediate[rowI][0] + IB(0, 1) * intermediate[rowI][2]);
                        triplets[tIBegin + 3] = Eigen::Triplet<T>(indMap[rowI], indMap[3], IB(0, 0) * intermediate[rowI][1] + IB(0, 1) * intermediate[rowI][3]);
                        triplets[tIBegin + 4] = Eigen::Triplet<T>(indMap[rowI], indMap[4], (-gamma[gI] * IB(0, 0) + (1 - lambdaq[lI]) * IB(1, 0)) * intermediate[rowI][0] + 
                            (-gamma[gI] * IB(0, 1) + (1 - lambdaq[lI]) * IB(1, 1)) * intermediate[rowI][2]);
                        triplets[tIBegin + 5] = Eigen::Triplet<T>(indMap[rowI], indMap[5], (-gamma[gI] * IB(0, 0) + (1 - lambdaq[lI]) * IB(1, 0)) * intermediate[rowI][1] + 
                            (-gamma[gI] * IB(0, 1) + (1 - lambdaq[lI]) * IB(1, 1)) * intermediate[rowI][3]);
                        triplets[tIBegin + 6] = Eigen::Triplet<T>(indMap[rowI], indMap[6], (gamma[gI] * IB(0, 0) + lambdaq[lI] * IB(1, 0)) * intermediate[rowI][0] + 
                            (gamma[gI] * IB(0, 1) + lambdaq[lI] * IB(1, 1)) * intermediate[rowI][2]);
                        triplets[tIBegin + 7] = Eigen::Triplet<T>(indMap[rowI], indMap[7], (gamma[gI] * IB(0, 0) + lambdaq[lI] * IB(1, 0)) * intermediate[rowI][1] + 
                            (gamma[gI] * IB(0, 1) + lambdaq[lI] * IB(1, 1)) * intermediate[rowI][3]);
                    }
                }
            }
        });
    }
    else {
        triplets.resize(Elem.size * gamma.size() * lambdaq.size() / 2 * 18 * 18);
        Elem.Par_Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            int indMap[6 * dim] = {
                elemVInd[0] * 2 * dim,
                elemVInd[0] * 2 * dim + 1,
                elemVInd[0] * 2 * dim + 2,
                elemVInd[1] * 2 * dim,
                elemVInd[1] * 2 * dim + 1,
                elemVInd[1] * 2 * dim + 2,
                elemVInd[2] * 2 * dim,
                elemVInd[2] * 2 * dim + 1,
                elemVInd[2] * 2 * dim + 2,
                (elemVInd[0] * 2 + 1) * dim,
                (elemVInd[0] * 2 + 1) * dim + 1,
                (elemVInd[0] * 2 + 1) * dim + 2,
                (elemVInd[1] * 2 + 1) * dim,
                (elemVInd[1] * 2 + 1) * dim + 1,
                (elemVInd[1] * 2 + 1) * dim + 2,
                (elemVInd[2] * 2 + 1) * dim,
                (elemVInd[2] * 2 + 1) * dim + 1,
                (elemVInd[2] * 2 + 1) * dim + 2
            };

            for (int lI = 0; lI < lambdaq.size() / 2; ++lI) {
                T lambda1 = lambdaq[lI * 2];
                T lambda2 = lambdaq[lI * 2 + 1];
                T lambda3 = 1 - lambda1 - lambda2;
                for (int gI = 0; gI < gamma.size(); ++gI) {
                    int qI = id * gamma.size() * lambdaq.size() / 2 + lI * gamma.size() + gI;
                    const auto& dPdF = std::get<0>(dP_div_dF.Get_Unchecked(qI));
                    const MATRIX<T, dim>& IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(qI));

                    // compute dfdx^T*dpdf*dfdx and make triplet
                    T intermediate[6 * dim][dim * dim];
                    for (int colI = 0; colI < dim * dim; ++colI) {
                        MATRIX<T, dim> P; // for convenience of reusing the code
                        P(0, 0) = dPdF(0, colI); P(0, 1) = dPdF(3, colI); P(0, 2) = dPdF(6, colI); 
                        P(1, 0) = dPdF(1, colI); P(1, 1) = dPdF(4, colI); P(1, 2) = dPdF(7, colI); 
                        P(2, 0) = dPdF(2, colI); P(2, 1) = dPdF(5, colI); P(2, 2) = dPdF(8, colI);
                        
                        intermediate[3][colI] = IB(0, 0) * P(0, 0) + IB(0, 1) * P(0, 1) + IB(0, 2) * P(0, 2);
                        intermediate[4][colI] = IB(0, 0) * P(1, 0) + IB(0, 1) * P(1, 1) + IB(0, 2) * P(1, 2);
                        intermediate[5][colI] = IB(0, 0) * P(2, 0) + IB(0, 1) * P(2, 1) + IB(0, 2) * P(2, 2);
                        intermediate[6][colI] = IB(1, 0) * P(0, 0) + IB(1, 1) * P(0, 1) + IB(1, 2) * P(0, 2);
                        intermediate[7][colI] = IB(1, 0) * P(1, 0) + IB(1, 1) * P(1, 1) + IB(1, 2) * P(1, 2);
                        intermediate[8][colI] = IB(1, 0) * P(2, 0) + IB(1, 1) * P(2, 1) + IB(1, 2) * P(2, 2);
                        intermediate[0][colI] = -intermediate[3][colI] - intermediate[6][colI];
                        intermediate[1][colI] = -intermediate[4][colI] - intermediate[7][colI];
                        intermediate[2][colI] = -intermediate[5][colI] - intermediate[8][colI];

                        const T gIB00_p_l1IB20 = gamma[gI] * IB(0, 0) + lambda1 * IB(2, 0);
                        const T gIB01_p_l1IB21 = gamma[gI] * IB(0, 1) + lambda1 * IB(2, 1);
                        const T gIB02_p_l1IB22 = gamma[gI] * IB(0, 2) + lambda1 * IB(2, 2);
                        intermediate[12][colI] = gIB00_p_l1IB20 * P(0, 0) + gIB01_p_l1IB21 * P(0, 1) + gIB02_p_l1IB22 * P(0, 2);
                        intermediate[13][colI] = gIB00_p_l1IB20 * P(1, 0) + gIB01_p_l1IB21 * P(1, 1) + gIB02_p_l1IB22 * P(1, 2);
                        intermediate[14][colI] = gIB00_p_l1IB20 * P(2, 0) + gIB01_p_l1IB21 * P(2, 1) + gIB02_p_l1IB22 * P(2, 2);
                        
                        const T gIB10_p_l2IB20 = gamma[gI] * IB(1, 0) + lambda2 * IB(2, 0);
                        const T gIB11_p_l2IB21 = gamma[gI] * IB(1, 1) + lambda2 * IB(2, 1);
                        const T gIB12_p_l2IB22 = gamma[gI] * IB(1, 2) + lambda2 * IB(2, 2);
                        intermediate[15][colI] = gIB10_p_l2IB20 * P(0, 0) + gIB11_p_l2IB21 * P(0, 1) + gIB12_p_l2IB22 * P(0, 2);
                        intermediate[16][colI] = gIB10_p_l2IB20 * P(1, 0) + gIB11_p_l2IB21 * P(1, 1) + gIB12_p_l2IB22 * P(1, 2);
                        intermediate[17][colI] = gIB10_p_l2IB20 * P(2, 0) + gIB11_p_l2IB21 * P(2, 1) + gIB12_p_l2IB22 * P(2, 2);

                        intermediate[9][colI] = IB(2, 0) * P(0, 0) + IB(2, 1) * P(0, 1) + IB(2, 2) * P(0, 2) - 
                            intermediate[12][colI] - intermediate[15][colI];
                        intermediate[10][colI] = IB(2, 0) * P(1, 0) + IB(2, 1) * P(1, 1) + IB(2, 2) * P(1, 2) - 
                            intermediate[13][colI] - intermediate[16][colI];
                        intermediate[11][colI] = IB(2, 0) * P(2, 0) + IB(2, 1) * P(2, 1) + IB(2, 2) * P(2, 2) - 
                            intermediate[14][colI] - intermediate[17][colI];
                    }
                    for (int rowI = 0; rowI < 6 * dim; ++rowI) {
                        MATRIX<T, dim> P; // for convenience of reusing the code
                        P(0, 0) = intermediate[rowI][0]; P(0, 1) = intermediate[rowI][3]; P(0, 2) = intermediate[rowI][6]; 
                        P(1, 0) = intermediate[rowI][1]; P(1, 1) = intermediate[rowI][4]; P(1, 2) = intermediate[rowI][7]; 
                        P(2, 0) = intermediate[rowI][2]; P(2, 1) = intermediate[rowI][5]; P(2, 2) = intermediate[rowI][8];
                        
                        int tIBegin = qI * 18 * 18 + rowI * 18;
                        triplets[tIBegin + 3] = Eigen::Triplet<T>(indMap[rowI], indMap[3], 
                            IB(0, 0) * P(0, 0) + IB(0, 1) * P(0, 1) + IB(0, 2) * P(0, 2));
                        triplets[tIBegin + 4] = Eigen::Triplet<T>(indMap[rowI], indMap[4], 
                            IB(0, 0) * P(1, 0) + IB(0, 1) * P(1, 1) + IB(0, 2) * P(1, 2));
                        triplets[tIBegin + 5] = Eigen::Triplet<T>(indMap[rowI], indMap[5], 
                            IB(0, 0) * P(2, 0) + IB(0, 1) * P(2, 1) + IB(0, 2) * P(2, 2));
                        triplets[tIBegin + 6] = Eigen::Triplet<T>(indMap[rowI], indMap[6], 
                            IB(1, 0) * P(0, 0) + IB(1, 1) * P(0, 1) + IB(1, 2) * P(0, 2));
                        triplets[tIBegin + 7] = Eigen::Triplet<T>(indMap[rowI], indMap[7], 
                            IB(1, 0) * P(1, 0) + IB(1, 1) * P(1, 1) + IB(1, 2) * P(1, 2));
                        triplets[tIBegin + 8] = Eigen::Triplet<T>(indMap[rowI], indMap[8], 
                            IB(1, 0) * P(2, 0) + IB(1, 1) * P(2, 1) + IB(1, 2) * P(2, 2));
                        triplets[tIBegin] = Eigen::Triplet<T>(indMap[rowI], indMap[0],
                            -triplets[tIBegin + 3].value() - triplets[tIBegin + 6].value());
                        triplets[tIBegin + 1] = Eigen::Triplet<T>(indMap[rowI], indMap[1], 
                            -triplets[tIBegin + 4].value() - triplets[tIBegin + 7].value());
                        triplets[tIBegin + 2] = Eigen::Triplet<T>(indMap[rowI], indMap[2], 
                            -triplets[tIBegin + 5].value() - triplets[tIBegin + 8].value());
                        
                        const T gIB00_p_l1IB20 = gamma[gI] * IB(0, 0) + lambda1 * IB(2, 0);
                        const T gIB01_p_l1IB21 = gamma[gI] * IB(0, 1) + lambda1 * IB(2, 1);
                        const T gIB02_p_l1IB22 = gamma[gI] * IB(0, 2) + lambda1 * IB(2, 2);
                        triplets[tIBegin + 12] = Eigen::Triplet<T>(indMap[rowI], indMap[12], 
                            gIB00_p_l1IB20 * P(0, 0) + gIB01_p_l1IB21 * P(0, 1) + gIB02_p_l1IB22 * P(0, 2));
                        triplets[tIBegin + 13] = Eigen::Triplet<T>(indMap[rowI], indMap[13], 
                            gIB00_p_l1IB20 * P(1, 0) + gIB01_p_l1IB21 * P(1, 1) + gIB02_p_l1IB22 * P(1, 2));
                        triplets[tIBegin + 14] = Eigen::Triplet<T>(indMap[rowI], indMap[14], 
                            gIB00_p_l1IB20 * P(2, 0) + gIB01_p_l1IB21 * P(2, 1) + gIB02_p_l1IB22 * P(2, 2));
                        
                        const T gIB10_p_l2IB20 = gamma[gI] * IB(1, 0) + lambda2 * IB(2, 0);
                        const T gIB11_p_l2IB21 = gamma[gI] * IB(1, 1) + lambda2 * IB(2, 1);
                        const T gIB12_p_l2IB22 = gamma[gI] * IB(1, 2) + lambda2 * IB(2, 2);
                        triplets[tIBegin + 15] = Eigen::Triplet<T>(indMap[rowI], indMap[15], 
                            gIB10_p_l2IB20 * P(0, 0) + gIB11_p_l2IB21 * P(0, 1) + gIB12_p_l2IB22 * P(0, 2));
                        triplets[tIBegin + 16] = Eigen::Triplet<T>(indMap[rowI], indMap[16], 
                            gIB10_p_l2IB20 * P(1, 0) + gIB11_p_l2IB21 * P(1, 1) + gIB12_p_l2IB22 * P(1, 2));
                        triplets[tIBegin + 17] = Eigen::Triplet<T>(indMap[rowI], indMap[17], 
                            gIB10_p_l2IB20 * P(2, 0) + gIB11_p_l2IB21 * P(2, 1) + gIB12_p_l2IB22 * P(2, 2));
                        
                        triplets[tIBegin + 9] = Eigen::Triplet<T>(indMap[rowI], indMap[9], 
                            IB(2, 0) * P(0, 0) + IB(2, 1) * P(0, 1) + IB(2, 2) * P(0, 2) - 
                            triplets[tIBegin + 12].value() - triplets[tIBegin + 15].value());
                        triplets[tIBegin + 10] = Eigen::Triplet<T>(indMap[rowI], indMap[10], 
                            IB(2, 0) * P(1, 0) + IB(2, 1) * P(1, 1) + IB(2, 2) * P(1, 2) - 
                            triplets[tIBegin + 13].value() - triplets[tIBegin + 16].value());
                        triplets[tIBegin + 11] = Eigen::Triplet<T>(indMap[rowI], indMap[11], 
                            IB(2, 0) * P(2, 0) + IB(2, 1) * P(2, 1) + IB(2, 2) * P(2, 2) - 
                            triplets[tIBegin + 14].value() - triplets[tIBegin + 17].value());
                    }
                }
            }
        });
    }
}

template <class T, int dim>
void Advance_One_Step_SE_Shell(
    std::vector<T>& gamma,
    std::vector<T>& lambdaq,
    MESH_ELEM<dim - 1>& Elem, // the segments
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const std::vector<T>& b, T h,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates and normal fibers, # = 2 * (#segments + 1)
    MESH_NODE_ATTR<T, dim>& nodeAttr, // # = 2 * (#segments + 1)
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim>& elemAttr, // # = #segments * gammaAmt * lambdaAmt
    FIXED_COROTATED<T, dim>& elasticityAttr) // # = #segments * gammaAmt * lambdaAmt
{
    Compute_Deformation_Gradient(gamma, lambdaq, X, Elem, elemAttr, elasticityAttr);

    FIXED_COROTATED_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(elasticityAttr, 1.0, elemAttr);
    
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Elasticity_Gradient_From_Stress(gamma, lambdaq, Elem, elemAttr, nodeAttr);

    std::vector<T> a = b;
    {
        TIMER_FLAG("compute acceleration");
        nodeAttr.Par_Each([&](int id, auto data) {
            auto &[x0, v, g, m] = data;
            a[id * dim] -= g[0];
            a[id * dim + 1] -= g[1];
            if constexpr (dim == 3) {
                a[id * dim + 2] -= g[2];
            }
        });
        if (!Solve_Direct(M, a, a)) {
            std::cout << "mass matrix factorization failed!" << std::endl;
            exit(-1);
        }
    }

    {
        TIMER_FLAG("update V and X");
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            v[0] += h * a[id * dim];
            v[1] += h * a[id * dim + 1];
            if constexpr (dim == 3) {
                v[2] += h * a[id * dim + 2];
            }
            x += h * v;
        });

        DBC.Par_Each([&](int id, auto data) {
            auto &[dbcI] = data;
            VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
            x(0) = dbcI(1);
            x(1) = dbcI(2);
            if constexpr (dim == 3) {
                x(2) = dbcI(3);
            }
        });
    }
}

template <class T, int dim>
void Compute_Barrier_Gradient_Shell(MESH_NODE<T, dim>& X_mesh,
    MESH_NODE_ATTR<T, dim>& nodeAttr_mesh,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    T dHat2, T kappa[],
    MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    nodeAttr_mesh.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Barrier_Gradient(X_mesh, constraintSet, 
        std::vector<VECTOR<T, 2>>(constraintSet.size(), VECTOR<T, 2>(1, dHat2)),
        dHat2, kappa, T(0), nodeAttr_mesh);

    if constexpr (dim == 2) {
        nodeAttr_mesh.Par_Each([&](int id, auto data) {
            if (id % 3 == 0) {
                const auto &[_, __, g_top, ___] = data;
                const VECTOR<T, dim>& g_mid = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(
                    nodeAttr_mesh.Get_Unchecked(id + 1));
                const VECTOR<T, dim>& g_bottom = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(
                    nodeAttr_mesh.Get_Unchecked(id + 2));
                VECTOR<T, dim>& g_x = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(
                    nodeAttr.Get_Unchecked(id / 3 * 2));
                VECTOR<T, dim>& g_n = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(
                    nodeAttr.Get_Unchecked(id / 3 * 2 + 1));
                
                g_x += g_top + g_bottom + g_mid;
                g_n += g_top - g_bottom;
            }
        });
    }
    else {
        nodeAttr_mesh.Par_Each([&](int id, auto data) {
            if (id % 2 == 0) {
                const auto &[_, __, g_top, ___] = data;
                const VECTOR<T, dim>& g_bottom = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(
                    nodeAttr_mesh.Get_Unchecked(id + 1));
                VECTOR<T, dim>& g_x = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(
                    nodeAttr.Get_Unchecked(id));
                VECTOR<T, dim>& g_n = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(
                    nodeAttr.Get_Unchecked(id + 1));
                
                g_x += g_top + g_bottom;
                g_n += g_top - g_bottom;
            }
        });
    }
}

template <class T, int dim>
void Compute_Barrier_Hessian_Shell(MESH_NODE<T, dim>& X_mesh,
    MESH_NODE_ATTR<T, dim>& nodeAttr_mesh,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    T dHat2, T kappa[],
    std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Barrier_Hessian_Shell");

    std::vector<Eigen::Triplet<T>> triplets_mesh;
    Compute_Barrier_Hessian(X_mesh, nodeAttr_mesh, constraintSet,
        std::vector<VECTOR<T, 2>>(constraintSet.size(), VECTOR<T, 2>(1, dHat2)),
        dHat2, kappa, T(0), true, triplets_mesh);

    int newTripletBegin = triplets.size();
    triplets.resize(triplets.size() + 4 * triplets_mesh.size());
    BASE_STORAGE<int> threads(triplets_mesh.size());
    for (int i = 0; i < triplets_mesh.size(); ++i) {
        threads.Append(i);
    }
    threads.Par_Each([&](int i, auto data) {
        if constexpr (dim == 2) {
            bool topT_bottomF[2] = {
                (triplets_mesh[i].row() / 2 % 3) == 0,
                (triplets_mesh[i].col() / 2 % 3) == 0
            };
            bool isMid[2] = {
                (triplets_mesh[i].row() / 2 % 3) == 1,
                (triplets_mesh[i].col() / 2 % 3) == 1
            };
            int xRowMtrInd = (triplets_mesh[i].row() / 2 / 3 * 2) * 2 + triplets_mesh[i].row() % 2;
            int nRowMtrInd = (triplets_mesh[i].row() / 2 / 3 * 2 + 1) * 2 + triplets_mesh[i].row() % 2;
            int xColMtrInd = (triplets_mesh[i].col() / 2 / 3 * 2) * 2 + triplets_mesh[i].col() % 2;
            int nColMtrInd = (triplets_mesh[i].col() / 2 / 3 * 2 + 1) * 2 + triplets_mesh[i].col() % 2;
            triplets[newTripletBegin + i * 4] = Eigen::Triplet<T>(xRowMtrInd, xColMtrInd, triplets_mesh[i].value());
            triplets[newTripletBegin + i * 4 + 1] = Eigen::Triplet<T>(xRowMtrInd, nColMtrInd, 
                isMid[1] ? T(0) : (topT_bottomF[1] ? triplets_mesh[i].value(): -triplets_mesh[i].value()));
            triplets[newTripletBegin + i * 4 + 2] = Eigen::Triplet<T>(nRowMtrInd, xColMtrInd, 
                isMid[0] ? T(0) : (topT_bottomF[0] ? triplets_mesh[i].value(): -triplets_mesh[i].value()));
            triplets[newTripletBegin + i * 4 + 3] = Eigen::Triplet<T>(nRowMtrInd, nColMtrInd, 
                (isMid[0] | isMid[1]) ? T(0) : ((topT_bottomF[0]^topT_bottomF[1]) ? -triplets_mesh[i].value(): triplets_mesh[i].value()));
        }
        else {
            bool topT_bottomF[2] = {
                (triplets_mesh[i].row() / dim % 2) == 0,
                (triplets_mesh[i].col() / dim % 2) == 0
            };
            int xRowMtrInd = topT_bottomF[0] ? triplets_mesh[i].row() : (triplets_mesh[i].row() - dim);
            int nRowMtrInd = topT_bottomF[0] ? (triplets_mesh[i].row() + dim) : triplets_mesh[i].row();
            int xColMtrInd = topT_bottomF[1] ? triplets_mesh[i].col() : (triplets_mesh[i].col() - dim);
            int nColMtrInd = topT_bottomF[1] ? (triplets_mesh[i].col() + dim) : triplets_mesh[i].col();
            triplets[newTripletBegin + i * 4] = Eigen::Triplet<T>(xRowMtrInd, xColMtrInd, triplets_mesh[i].value());
            triplets[newTripletBegin + i * 4 + 1] = Eigen::Triplet<T>(xRowMtrInd, nColMtrInd, 
                topT_bottomF[1] ? triplets_mesh[i].value(): -triplets_mesh[i].value());
            triplets[newTripletBegin + i * 4 + 2] = Eigen::Triplet<T>(nRowMtrInd, xColMtrInd, 
                topT_bottomF[0] ? triplets_mesh[i].value(): -triplets_mesh[i].value());
            triplets[newTripletBegin + i * 4 + 3] = Eigen::Triplet<T>(nRowMtrInd, nColMtrInd, 
                (topT_bottomF[0]^topT_bottomF[1]) ? -triplets_mesh[i].value(): triplets_mesh[i].value());
        }
    });
}

template <class T, int dim>
int Advance_One_Step_IE_Shell(
    std::vector<T>& gamma,
    std::vector<T>& lambdaq,
    MESH_ELEM<dim - 1>& Elem, // the segments
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const std::vector<T>& b, 
    T h, T NewtonTol,
    bool withCollision,
    T dHat2, const VECTOR<T, 2>& kappaVec,
    bool staticSolve,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates and normal fibers, # = 2 * (#segments + 1)
    MESH_NODE_ATTR<T, dim>& nodeAttr, // # = 2 * (#segments + 1)
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim>& elemAttr, // # = #segments * gammaAmt * lambdaAmt
    FIXED_COROTATED<T, dim>& elasticityAttr, // # = #segments * gammaAmt * lambdaAmt
    MESH_NODE<T, dim>& X_mesh,
    MESH_ELEM<2>& triangles,
    MESH_NODE_ATTR<T, dim>& nodeAttr_mesh)
{
    TIMER_FLAG("implicitEuler");
    
    T kappa[] = {kappaVec[0], kappaVec[1]}; // dumb pybind does not support c array

    // record Xn and compute predictive pos Xtilde
    std::cout << "compute Xtilde" << std::endl;
    MESH_NODE<T, dim> Xn, Xtilde;
    if (!staticSolve) {
        Append_Attribute(X, Xn);
        Append_Attribute(X, Xtilde);
        //TODO: only once per sim
        std::vector<T> a;
        if (!Solve_Direct(M, b, a)) {
            std::cout << "mass matrix factorization failed!" << std::endl;
            exit(-1);
        }
        Xtilde.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            x[0] += h * v[0] + h * h * a[id * dim];
            x[1] += h * v[1] + h * h * a[id * dim + 1];
            if constexpr (dim == 3) {
                x[2] += h * v[2] + h * h * a[id * dim + 2];
            }
        });
    }
    
    // set Dirichlet boundary condition on X
    std::cout << "process DBC" << std::endl;
    std::vector<bool> DBCb(X.size, false);
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
        x(0) = dbcI(1);
        x(1) = dbcI(2);
        if constexpr (dim == 3) {
            x(2) = dbcI(3);
        }
        DBCb[dbcI(0)] = true; // bool array cannot be written in parallel by entries
    });
    if (withCollision) {
        Update_Render_Mesh_Node(X, X_mesh);
    }

    CSR_MATRIX<T> sysMtr;
    std::vector<T> rhs(X.size * dim), sol(X.size * dim);

    //TODO: only once
    // compute contact primitives
    std::cout << "find boundary primitives" << std::endl;
    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    std::vector<VECTOR<int, 3>> boundaryTri;
    std::vector<T> BNArea, BEArea, BTArea;
    if (withCollision) {
        if constexpr (dim == 2) {
            Find_Boundary_Edge_And_Node(X_mesh.size, triangles, boundaryNode, boundaryEdge);
        }
        else {
            Find_Surface_Primitives_And_Compute_Area(X_mesh, triangles, boundaryNode, boundaryEdge, boundaryTri, BNArea, BEArea, BTArea);
        }
    }
    //TODO: dHat relative to bbox, adapt kappa

    // Newton loop
    int PNIter = 0;
    T infNorm = 0.0;
    bool projectSPD = true;
    bool useGD = false;
    // compute deformation gradient, constraint set, and energy
    Compute_Deformation_Gradient(gamma, lambdaq, X, Elem, elemAttr, elasticityAttr);
    std::vector<VECTOR<int, dim + 1>> constraintSet;
    std::vector<VECTOR<int, 2>> constraintSetPTEE;
    std::vector<VECTOR<T, 2>> stencilInfo;
    if (withCollision) {
        Compute_Constraint_Set<T, dim, true>(X_mesh, nodeAttr_mesh, boundaryNode, boundaryEdge, boundaryTri, 
            std::vector<int>(), std::vector<VECTOR<int, 2>>(), std::map<int, std::set<int>>(), BNArea, BEArea, BTArea,
            VECTOR<int, 2>(boundaryNode.size(), boundaryNode.size()),
            DBCb, dHat2, T(0), false, constraintSet, constraintSetPTEE, stencilInfo);
    }
    T Eprev;
    Compute_IncPotential(h, X, Xtilde, M, elemAttr, elasticityAttr, 
        X_mesh, nodeAttr_mesh, withCollision, constraintSet, dHat2, kappa, staticSolve, b, Eprev);
    do {
        // compute gradient
        std::cout << "\ncompute gradient" << std::endl;
        Compute_IncPotential_Gradient(gamma, lambdaq, Elem, h, X, Xtilde, nodeAttr, M, elemAttr, 
            X_mesh, nodeAttr_mesh, withCollision, constraintSet, dHat2, kappa, staticSolve, b, elasticityAttr);
        // project rhs for Dirichlet boundary condition
        DBC.Par_Each([&](int id, auto data) {
            auto &[dbcI] = data;
            std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(dbcI(0))).setZero();
        });
        nodeAttr.Par_Each([&](int id, auto data) {
            auto &[x0, v, g, m] = data;
            rhs[id * dim] = -g[0];
            rhs[id * dim + 1] = -g[1];
            if constexpr (dim == 3) {
                rhs[id * dim + 2] = -g[2];
            }
        });

        // compute Hessian
        if (!useGD) {
            std::cout << "compute hessian" << std::endl;
            std::vector<Eigen::Triplet<T>> triplets;
            Compute_Elasticity_Hessian(gamma, lambdaq, Elem, staticSolve ? T(1) : h, 
                projectSPD, elemAttr, elasticityAttr, triplets);
            // projectSPD = false;
            if (withCollision) {
                Compute_Barrier_Hessian_Shell(X_mesh, nodeAttr_mesh, constraintSet, dHat2, kappa, triplets);
            }
            sysMtr.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
            if (!staticSolve) {
                TIMER_FLAG("add mass matrix");
                sysMtr.Get_Matrix() += M.Get_Matrix();
            }
            // project Matrix for Dirichlet boundary condition
            sysMtr.Project_DBC(DBCb, dim);
        }

        // compute search direction
        {
            TIMER_FLAG("linearSolve");

            // AMGCL
            // std::memset(sol.data(), 0, sizeof(T) * sol.size());
            // Solve(sysMtr, rhs, sol, 1.0e-3, 1000, Default_FEM_Params(dim), true); //TODO: output

            if (useGD) {
                printf("use gradient descent\n");
                std::memcpy(sol.data(), rhs.data(), sizeof(T) * rhs.size());
            }
            else {
                // direct factorization
                if(!Solve_Direct(sysMtr, rhs, sol)) {
                    useGD = true;
                    printf("use gradient descent\n");
                    std::memcpy(sol.data(), rhs.data(), sizeof(T) * rhs.size());
                }
            }
        }

        // line search
        std::cout << "line search" << std::endl;
        MESH_NODE<T, dim> Xprev;
        Append_Attribute(X, Xprev);
        T alpha = 1.0, E;
        if (withCollision) {
            std::vector<T> p_mesh(X_mesh.size * dim);
            if constexpr (dim == 2) {
                //TODO: parallelize
                for (int i = 0; i < sol.size() / 2; ++i) {
                    if (i % 2 == 0) {
                        T px_x = sol[i * 2];
                        T px_y = sol[i * 2 + 1];
                        T pn_x = sol[(i + 1) * 2];
                        T pn_y = sol[(i + 1) * 2 + 1];
                        int topNodeI = i / 2 * 3;
                        int midNodeI = i / 2 * 3 + 1;
                        int bottomNodeI = i / 2 * 3 + 2;
                        p_mesh[topNodeI * 2] = px_x + pn_x;
                        p_mesh[topNodeI * 2 + 1] = px_y + pn_y;
                        p_mesh[midNodeI * 2] = px_x;
                        p_mesh[midNodeI * 2 + 1] = px_y;
                        p_mesh[bottomNodeI * 2] = px_x - pn_x;
                        p_mesh[bottomNodeI * 2 + 1] = px_y - pn_y;
                    }
                }
            }
            else {
                //TODO: parallelize
                for (int i = 0; i < sol.size() / dim; ++i) {
                    if (i % 2 == 0) { // mid-surface node
                        T px_x = sol[i * dim];
                        T px_y = sol[i * dim + 1];
                        T px_z = sol[i * dim + 2];
                        T pn_x = sol[(i + 1) * dim];
                        T pn_y = sol[(i + 1) * dim + 1];
                        T pn_z = sol[(i + 1) * dim + 2];
                        p_mesh[i * dim] = px_x + pn_x;
                        p_mesh[i * dim + 1] = px_y + pn_y;
                        p_mesh[i * dim + 2] = px_z + pn_z;
                        p_mesh[(i + 1) * dim] = px_x - pn_x;
                        p_mesh[(i + 1) * dim + 1] = px_y - pn_y;
                        p_mesh[(i + 1) * dim + 2] = px_z - pn_z;
                    }
                }
            }
            Compute_Intersection_Free_StepSize<T, dim, true>(X_mesh, boundaryNode, boundaryEdge, boundaryTri, 
                std::vector<int>(), std::vector<VECTOR<int, 2>>(), std::map<int, std::set<int>>(), 
                VECTOR<int, 2>(boundaryNode.size(), boundaryNode.size()), 
                DBCb, p_mesh, T(0), alpha); // CCD
            printf("intersection free step size = %le\n", alpha);
        }
        int nBT = -1;
        do {
            X.Join(Xprev).Par_Each([&](int id, auto data) {
                auto &[x, xprev] = data;
                x[0] = xprev[0] + alpha * sol[id * dim];
                x[1] = xprev[1] + alpha * sol[id * dim + 1];
                if constexpr (dim == 3) {
                    x[2] = xprev[2] + alpha * sol[id * dim + 2];
                }
            });
            Compute_Deformation_Gradient(gamma, lambdaq, X, Elem, elemAttr, elasticityAttr);
            if (withCollision) {
                Update_Render_Mesh_Node(X, X_mesh);
                Compute_Constraint_Set<T, dim, true>(X_mesh, nodeAttr_mesh, boundaryNode, boundaryEdge, boundaryTri, 
                    std::vector<int>(), std::vector<VECTOR<int, 2>>(), std::map<int, std::set<int>>(), BNArea, BEArea, BTArea,
                    VECTOR<int, 2>(boundaryNode.size(), boundaryNode.size()),
                    DBCb, dHat2, T(0), false, constraintSet, constraintSetPTEE, stencilInfo);
            }
            Compute_IncPotential(h, X, Xtilde, M, elemAttr, elasticityAttr, 
                X_mesh, nodeAttr_mesh, withCollision, constraintSet, dHat2, kappa, staticSolve, b, E);
            alpha /= 2.0;
            ++nBT;
        } while (E > Eprev);
        if (nBT > 3) {
           projectSPD = true;
        }
        printf("alpha = %le\n", alpha * 2.0);
        Eprev = E;

        if (constraintSet.size()) {
            T minDist2;
            std::vector<T> dist2;
            Compute_Min_Dist2(X_mesh, constraintSet, T(0), dist2, minDist2);
            printf("minDist2 = %le\n", minDist2);
        }

        // stopping criteria
        infNorm = 0.0;
        for (int i = 0; i < sol.size(); ++i) {
            if (infNorm < std::abs(sol[i])) {
                infNorm = std::abs(sol[i]);
            }
        }
        infNorm /= (staticSolve ? 1 : h);
        printf("PNIter%d: Newton res = %le, tol = %le\n", PNIter++, infNorm, NewtonTol);

        if (useGD) {
            infNorm = NewtonTol * 10; // ensures not exit Newton loop
        }

        if (alpha * 2 < 1e-6) {
            if (!useGD) {
                useGD = true;
                Eigen::VectorXd pe(sol.size()), mge(rhs.size());
                std::memcpy(pe.data(), sol.data(), sizeof(T) * sol.size());
                std::memcpy(mge.data(), rhs.data(), sizeof(T) * rhs.size());
                printf("-gdotp = %le, -gpcos = %le\n", mge.dot(pe), 
                    mge.dot(pe) / std::sqrt(mge.squaredNorm() * pe.squaredNorm()));
                printf("linear solve relErr = %le\n", 
                    std::sqrt((sysMtr.Get_Matrix() * pe - mge).squaredNorm() / mge.squaredNorm()));
            }
            else {
                printf("GD tiny step size!\n");
            }
        }
        else {
            useGD = false;
        }
    } while (infNorm > NewtonTol); //TODO: newtonTol relative to bbox

    if (withCollision) {
        printf("contact #: %lu\n", constraintSet.size());
    }

    if (!staticSolve) {
        // update velocity
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            v = (x - std::get<0>(Xn.Get_Unchecked(id))) / h;
        });
    }

    return PNIter;
}

template<class T, int dim>
void Update_Render_Mesh_Node(
    MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& X_mesh)
{
    TIMER_FLAG("Update_Shell_Mesh_Node");

    X.Par_Each([&](int id, auto data) {
        if (id % 2 == 0) {
            auto &[x] = data;
            const VECTOR<T, dim>& n = std::get<0>(X.Get_Unchecked(id + 1));

            if constexpr (dim == 2) {
                std::get<0>(X_mesh.Get_Unchecked(id / 2 * 3)) = x + n;
                std::get<0>(X_mesh.Get_Unchecked(id / 2 * 3 + 1)) = x;
                std::get<0>(X_mesh.Get_Unchecked(id / 2 * 3 + 2)) = x - n;
            }
            else {
                std::get<0>(X_mesh.Get_Unchecked(id)) = x + n;
                std::get<0>(X_mesh.Get_Unchecked(id + 1)) = x - n;
            }
        }
    });
}

template<class T, int dim>
void Write_Node_Coordinate(
    MESH_NODE<T, dim>& X_mesh,
    const std::string& filePath)
{
    FILE *out = fopen(filePath.c_str(), "a+");
    VECTOR<T, dim> x = std::get<0>(X_mesh.Get_Unchecked(X_mesh.size / 2 + 1));
    if constexpr (dim == 2) {
        fprintf(out, "%lu %.20le %.20le\n", X_mesh.size, x[0], x[1]);
    }
    else {
        fprintf(out, "%lu %.20le %.20le %.20le\n", X_mesh.size, x[0], x[1], x[2]);
    }
    fclose(out);
}

template <class T, int dim>
void Check_Gradient(
    std::vector<T>& gamma,
    std::vector<T>& lambdaq,
    MESH_ELEM<dim - 1>& Elem,
    const VECTOR_STORAGE<T, dim + 1>& DBC,
    T h, MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xtilde,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr)
{
    T eps = 1.0e-6;

    Compute_Deformation_Gradient(gamma, lambdaq, X, Elem, elemAttr, elasticityAttr);
    T E0;
    Compute_IncPotential(h, X, Xtilde, M, elemAttr, elasticityAttr, E0);

    std::vector<T> grad_FD(X.size * dim);
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        Compute_Deformation_Gradient(gamma, lambdaq, Xperturb, Elem, elemAttr, elasticityAttr);
        T E;
        Compute_IncPotential(h, Xperturb, Xtilde, M, elemAttr, elasticityAttr, E);
        grad_FD[i] = (E - E0) / eps;
    }

    Compute_Deformation_Gradient(gamma, lambdaq, X, Elem, elemAttr, elasticityAttr);
    Compute_IncPotential_Gradient(gamma, lambdaq, Elem, h, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr);

    T err = 0.0, norm = 0.0;
    nodeAttr.Each([&](int id, auto data) {
        auto &[v, g, m] = data;

        err += std::pow(grad_FD[id * dim] - g[0], 2);
        err += std::pow(grad_FD[id * dim + 1] - g[1], 2);

        norm += std::pow(grad_FD[id * dim], 2);
        norm += std::pow(grad_FD[id * dim + 1], 2);

        if constexpr (dim == 3) {
            err += std::pow(grad_FD[id * dim + 2] - g[2], 2);
            norm += std::pow(grad_FD[id * dim + 2], 2);
        }
    });
    printf("err_abs = %le, err_rel = %le", err, err / norm);
}

void Export_Shell(py::module& m) {
    py::module shell_m = m.def_submodule("Shell", "A submodule of JGSL for FEM shell simulation");
    shell_m.def("Add_Shell", &Add_Shell_2D<double>);
    shell_m.def("Initialize_Shell", &Initialize_Shell<double, 2>);
    shell_m.def("Init_Dirichlet", &Init_Dirichlet_Shell<double, 2>);
    shell_m.def("Initialize_Displacement", &Initialize_Displacement<double, 2>);
    shell_m.def("Advance_One_Step_SE", &Advance_One_Step_SE_Shell<double, 2>);
    shell_m.def("Advance_One_Step_IE", &Advance_One_Step_IE_Shell<double, 2>);
    shell_m.def("Update_Render_Mesh_Node", &Update_Render_Mesh_Node<double, 2>);
    shell_m.def("Write_Node_Coordinate", &Write_Node_Coordinate<double, 2>);

    shell_m.def("Add_Shell", &Add_Shell_3D<double>);
    shell_m.def("Initialize_Shell", &Initialize_Shell<double, 3>);
    shell_m.def("Init_Dirichlet", &Init_Dirichlet_Shell<double, 3>);
    shell_m.def("Initialize_Displacement", &Initialize_Displacement<double, 3>);
    shell_m.def("Advance_One_Step_SE", &Advance_One_Step_SE_Shell<double, 3>);
    shell_m.def("Advance_One_Step_IE", &Advance_One_Step_IE_Shell<double, 3>);
    shell_m.def("Update_Render_Mesh_Node", &Update_Render_Mesh_Node<double, 3>);
    shell_m.def("Write_Node_Coordinate", &Write_Node_Coordinate<double, 3>);
}

}
