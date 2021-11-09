#pragma once

#include <FEM/Shell/Rod/ROD_BENDING_DERIVATIVES.h>

#include <Math/VECTOR.h>

namespace JGSL {

template <class T, int dim = 3>
void Compute_Rod_Bending_Energy(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo, // rodInfo: E_i, lsum_rest_i, thickness_i
    T h, T& E)
{
    TIMER_FLAG("Compute_Rod_Bending_Energy");

    int i = 0;
    for (const auto& hingeI : rodHinge) {
        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(hingeI[0]));
        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(hingeI[1]));
        const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(hingeI[2]));

        const VECTOR<T, dim> e0 = x1 - x0;
        const VECTOR<T, dim> e1 = x2 - x1;

        const VECTOR<T, dim> kappa = cross(e0, e1) * 2 / (sqrt(e0.length2() * e1.length2()) + e0.dot(e1));
        const T alpha = rodHingeInfo[i][0] * std::pow(rodHingeInfo[i][2], 4) * M_PI / 64;
        E += h * h * alpha * kappa.length2() / rodHingeInfo[i][1];

        ++i;
    }
}

template <class T, int dim = 3>
void Compute_Rod_Bending_Gradient(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo, // rodInfo: E_i, lsum_rest_i, thickness_i
    T h, MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    TIMER_FLAG("Compute_Rod_Bending_Gradient");

    int i = 0;
    for (const auto& hingeI : rodHinge) {
        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(hingeI[0]));
        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(hingeI[1]));
        const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(hingeI[2]));

        T g_kappa2[9];
        g_RB(x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], g_kappa2);

        const T w = h * h / rodHingeInfo[i][1] *
            rodHingeInfo[i][0] * std::pow(rodHingeInfo[i][2], 4) * M_PI / 64;
        for (int endI = 0; endI < 3; ++endI) {
            VECTOR<T, dim>& grad = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(hingeI[endI]));
            for (int dimI = 0; dimI < dim; ++dimI) {
                grad[dimI] += w * g_kappa2[endI * dim + dimI];
            }
        }

        ++i;
    }
}

template <class T, int dim = 3>
void Compute_Rod_Bending_Hessian(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo, // rodInfo: E_i, lsum_rest_i, thickness_i
    T h, bool projectSPD, std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Rod_Bending_Hessian");

    BASE_STORAGE<int> threads(rodHinge.size());
    for (int i = 0; i < rodHinge.size(); ++i) {
        threads.Append(triplets.size() + i * 81);
    }

    triplets.resize(triplets.size() + rodHinge.size() * 81);
    threads.Par_Each([&](int i, auto data) {
        const auto& [tripletStartInd] = data;
        const VECTOR<int, 3>& hingeI = rodHinge[i];
        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(hingeI[0]));
        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(hingeI[1]));
        const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(hingeI[2]));

        Eigen::Matrix<T, 9, 9> hessian;
        H_RB(x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], hessian.data());

        if (projectSPD) {
            makePD(hessian);
        }

        int globalInd[9] = { 
            hingeI[0] * dim,
            hingeI[0] * dim + 1,
            hingeI[0] * dim + 2,
            hingeI[1] * dim,
            hingeI[1] * dim + 1,
            hingeI[1] * dim + 2,
            hingeI[2] * dim,
            hingeI[2] * dim + 1,
            hingeI[2] * dim + 2,
        };
        const T w = h * h / rodHingeInfo[i][1] *
            rodHingeInfo[i][0] * std::pow(rodHingeInfo[i][2], 4) * M_PI / 64;
        for (int rowI = 0; rowI < 9; ++rowI) {
            for (int colI = 0; colI < 9; ++colI) {
                triplets[tripletStartInd + rowI * 9 + colI] = Eigen::Triplet<T>(
                    globalInd[rowI], globalInd[colI], w * hessian(rowI, colI)
                );
            }
        }
    });
}

}