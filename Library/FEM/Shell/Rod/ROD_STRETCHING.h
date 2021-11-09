#pragma once

#include <FEM/Shell/Rod/MASS_SPRING_DERIVATIVES.h>

#include <Math/VECTOR.h>

namespace JGSL {

template <class T, int dim = 3>
void Compute_Rod_Spring_Energy(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo, // rodInfo: E_i, l_rest_i, thickness_i
    T h, T& E)
{
    TIMER_FLAG("Compute_Rod_Spring_Energy");

    int i = 0;
    for (const auto& segI : rod) {
        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(segI[0]));
        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(segI[1]));

        E += h * h * M_PI * rodInfo[i][2] * rodInfo[i][2] / 4 * rodInfo[i][1] * rodInfo[i][0] / 2 * 
            std::pow((x0 - x1).length() / rodInfo[i][1] - 1, 2);

        ++i;
    }
}

template <class T, int dim = 3>
void Compute_Rod_Spring_Gradient(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo, // rodInfo: E_i, l_rest_i, thickness_i
    T h, MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    TIMER_FLAG("Compute_Rod_Spring_Gradient");

    int i = 0;
    for (const auto& segI : rod) {
        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(segI[0]));
        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(segI[1]));

        T g[6];
        g_MS(rodInfo[i][0], rodInfo[i][1], 
            x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], g);
        
        T w = h * h * M_PI * rodInfo[i][2] * rodInfo[i][2] / 4;
        for (int endI = 0; endI < 2; ++endI) {
            VECTOR<T, dim>& grad = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(segI[endI]));
            for (int dimI = 0; dimI < dim; ++dimI) {
                grad[dimI] += w * g[endI * dim + dimI];
            }
        }

        ++i;
    }
}

template <class T, int dim = 3>
void Compute_Rod_Spring_Hessian(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo, // rodInfo: E_i, l_rest_i, thickness_i
    T h, bool projectSPD, std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Rod_Spring_Hessian");

    BASE_STORAGE<int> threads(rod.size());
    for (int i = 0; i < rod.size(); ++i) {
        threads.Append(triplets.size() + i * 36);
    }

    triplets.resize(triplets.size() + rod.size() * 36);
    threads.Par_Each([&](int i, auto data) {
        const auto& [tripletStartInd] = data;
        const VECTOR<int, 2>& segI = rod[i];
        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(segI[0]));
        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(segI[1]));

        Eigen::Matrix<T, 6, 6> hessian;
        H_MS(rodInfo[i][0], rodInfo[i][1], 
            x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], hessian.data());

        if (projectSPD) {
            makePD(hessian);
        }
        
        int globalInd[6] = { 
            segI[0] * dim,
            segI[0] * dim + 1,
            segI[0] * dim + 2,
            segI[1] * dim,
            segI[1] * dim + 1,
            segI[1] * dim + 2,
        };
        T w = h * h * M_PI * rodInfo[i][2] * rodInfo[i][2] / 4;
        for (int rowI = 0; rowI < 6; ++rowI) {
            for (int colI = 0; colI < 6; ++colI) {
                triplets[tripletStartInd + rowI * 6 + colI] = Eigen::Triplet<T>(
                    globalInd[rowI], globalInd[colI], w * hessian(rowI, colI)
                );
            }
        }
    });
}

template <class T, int dim>
void Check_Rod_Spring_Gradient(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo, // rodInfo: E_i, l_rest_i, thickness_i
    T h, MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    T eps = 1.0e-6;

    T E0 = 0;
    Compute_Rod_Spring_Energy(X, rod, rodInfo, h, E0);
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Rod_Spring_Gradient(X, rod, rodInfo, h, nodeAttr);

    std::vector<T> grad_FD(X.size * dim);
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        T E = 0;
        Compute_Rod_Spring_Energy(Xperturb, rod, rodInfo, h, E);
        grad_FD[i] = (E - E0) / eps;
    }

    T err = 0.0, norm = 0.0;
    nodeAttr.Each([&](int id, auto data) {
        auto &[x0, v, g, m] = data;

        err += std::pow(grad_FD[id * dim] - g[0], 2);
        err += std::pow(grad_FD[id * dim + 1] - g[1], 2);

        norm += std::pow(grad_FD[id * dim], 2);
        norm += std::pow(grad_FD[id * dim + 1], 2);

        if constexpr (dim == 3) {
            err += std::pow(grad_FD[id * dim + 2] - g[2], 2);
            norm += std::pow(grad_FD[id * dim + 2], 2);
        }
    });
    printf("err_abs = %le, sqnorm_FD = %le, err_rel = %le\n", err, norm, err / norm);
}

template <class T, int dim>
void Check_Rod_Spring_Hessian(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo, // rodInfo: E_i, l_rest_i, thickness_i
    T h, MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    T eps = 1.0e-6;

    MESH_NODE_ATTR<T, dim> nodeAttr0;
    nodeAttr.deep_copy_to(nodeAttr0);
    nodeAttr0.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Rod_Spring_Gradient(X, rod, rodInfo, h, nodeAttr0);
    std::vector<Eigen::Triplet<T>> HStriplets;
    Compute_Rod_Spring_Hessian(X, rod, rodInfo, h, false, HStriplets);
    CSR_MATRIX<T> HS;
    HS.Construct_From_Triplet(X.size * dim, X.size * dim, HStriplets);

    std::vector<Eigen::Triplet<T>> HFDtriplets;
    HFDtriplets.reserve(HStriplets.size());
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
        Compute_Rod_Spring_Gradient(Xperturb, rod, rodInfo, h, nodeAttr);
        for (int vI = 0; vI < X.size; ++vI) {
            const VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(vI));
            const VECTOR<T, dim>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr0.Get_Unchecked(vI));
            const VECTOR<T, dim> hFD = (g - g0) / eps;
            if (hFD.length2() != 0) {
                HFDtriplets.emplace_back(i, vI * dim, hFD[0]);
                HFDtriplets.emplace_back(i, vI * dim + 1, hFD[1]);
                if constexpr (dim == 3) {
                    HFDtriplets.emplace_back(i, vI * dim + 2, hFD[2]);
                }
            }
        }
    }
    CSR_MATRIX<T> HFD;
    HFD.Construct_From_Triplet(X.size * dim, X.size * dim, HFDtriplets);

    T err = (HS.Get_Matrix() - HFD.Get_Matrix()).squaredNorm(), norm = HFD.Get_Matrix().squaredNorm();
    printf("err_abs = %le, sqnorm_FD = %le, err_rel = %le\n", err, norm, err / norm);
}

}