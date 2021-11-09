#pragma once

#include <Math/VECTOR.h>

namespace JGSL {

template <class T, int dim = 3>
void Compute_Stitch_Energy(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    const std::vector<bool>& DBCb,
    T k, T h, T& E)
{
    TIMER_FLAG("Compute_Stitch_Energy");

    int i = 0;
    for (const auto& stitchI : stitchInfo) {
        if (DBCb[stitchI[0]] || DBCb[stitchI[1]] || DBCb[stitchI[2]]) {
            continue;
        }

        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(stitchI[0]));
        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(stitchI[1]));
        const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(stitchI[2]));

        E += h * h * k / 2 * (x0 - (x1 * (1 - stitchRatio[i]) + x2 * stitchRatio[i])).length2();

        ++i;
    }
}

template <class T, int dim = 3>
void Compute_Stitch_Gradient(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    const std::vector<bool>& DBCb,
    T k, T h, MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    TIMER_FLAG("Compute_Stitch_Gradient");

    int i = 0;
    for (const auto& stitchI : stitchInfo) {
        if (DBCb[stitchI[0]] || DBCb[stitchI[1]] || DBCb[stitchI[2]]) {
            continue;
        }

        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(stitchI[0]));
        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(stitchI[1]));
        const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(stitchI[2]));

        const VECTOR<T, dim> distVec = x0 - (x1 * (1 - stitchRatio[i]) + x2 * stitchRatio[i]);
        T w = h * h * k;
        VECTOR<T, dim>& grad0 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(stitchI[0]));
        grad0 += w * distVec;
        VECTOR<T, dim>& grad1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(stitchI[1]));
        grad1 += w * (stitchRatio[i] - 1) * distVec;
        VECTOR<T, dim>& grad2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(stitchI[2]));
        grad2 += w * -stitchRatio[i] * distVec;

        ++i;
    }
}

template <class T, int dim = 3>
void Compute_Stitch_Hessian(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    const std::vector<bool>& DBCb,
    T k, T h, bool projectSPD, std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Stitch_Hessian");

    BASE_STORAGE<int> threads(stitchInfo.size());
    int counter = triplets.size();
    for (const auto& stitchI : stitchInfo) {
        if (DBCb[stitchI[0]] || DBCb[stitchI[1]] || DBCb[stitchI[2]]) {
            threads.Append(-1);
        }
        else {
            threads.Append(counter);
            counter += 81;
        }
    }

    triplets.resize(counter);
    threads.Par_Each([&](int i, auto data) {
        const auto& [tripletStartInd] = data;
        if (tripletStartInd >= 0) {
            const VECTOR<int, 3>& stitchI = stitchInfo[i];
            const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(stitchI[0]));
            const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(stitchI[1]));
            const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(stitchI[2]));

            Eigen::Matrix<T, 9, 9> hessian;
            hessian.setZero();
            hessian.template block<3, 3>(0, 0).diagonal().setConstant(1);
            hessian.template block<3, 3>(0, 3).diagonal().setConstant(stitchRatio[i] - 1);
            hessian.template block<3, 3>(3, 0).diagonal().setConstant(stitchRatio[i] - 1);
            hessian.template block<3, 3>(0, 6).diagonal().setConstant(-stitchRatio[i]);
            hessian.template block<3, 3>(6, 0).diagonal().setConstant(-stitchRatio[i]);
            hessian.template block<3, 3>(3, 3).diagonal().setConstant((stitchRatio[i] - 1) * (stitchRatio[i] - 1));
            hessian.template block<3, 3>(3, 6).diagonal().setConstant(stitchRatio[i] * (1 - stitchRatio[i]));
            hessian.template block<3, 3>(6, 3).diagonal().setConstant(stitchRatio[i] * (1 - stitchRatio[i]));
            hessian.template block<3, 3>(6, 6).diagonal().setConstant(stitchRatio[i] * stitchRatio[i]);

            if (projectSPD) {
                makePD(hessian);
            }
            
            int globalInd[9] = { 
                stitchI[0] * dim,
                stitchI[0] * dim + 1,
                stitchI[0] * dim + 2,
                stitchI[1] * dim,
                stitchI[1] * dim + 1,
                stitchI[1] * dim + 2,
                stitchI[2] * dim,
                stitchI[2] * dim + 1,
                stitchI[2] * dim + 2,
            };
            T w = h * h * k;
            for (int rowI = 0; rowI < 9; ++rowI) {
                for (int colI = 0; colI < 9; ++colI) {
                    triplets[tripletStartInd + rowI * 9 + colI] = Eigen::Triplet<T>(
                        globalInd[rowI], globalInd[colI], w * hessian(rowI, colI)
                    );
                }
            }
        }
    });
}

template <class T, int dim>
void Check_Stitch_Gradient(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    T k, T h, MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    T eps = 1.0e-6;

    T E0 = 0;
    Compute_Stitch_Energy(X, stitchInfo, stitchRatio, k, h, E0);
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Stitch_Gradient(X, stitchInfo, stitchRatio, k, h, nodeAttr);

    std::vector<T> grad_FD(X.size * dim);
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        T E = 0;
        Compute_Stitch_Energy(Xperturb, stitchInfo, stitchRatio, k, h, E);
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
void Check_Stitch_Hessian(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    T k, T h, MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    T eps = 1.0e-6;

    MESH_NODE_ATTR<T, dim> nodeAttr0;
    nodeAttr.deep_copy_to(nodeAttr0);
    nodeAttr0.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Stitch_Gradient(X, stitchInfo, stitchRatio, k, h, nodeAttr0);
    std::vector<Eigen::Triplet<T>> HStriplets;
    Compute_Stitch_Hessian(X, stitchInfo, stitchRatio, k, h, false, HStriplets);
    CSR_MATRIX<T> HS;
    HS.Construct_From_Triplet(X.size * dim, X.size * dim, HStriplets);

    std::vector<Eigen::Triplet<T>> HFDtriplets;
    HFDtriplets.reserve(HStriplets.size());
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
        Compute_Stitch_Gradient(Xperturb, stitchInfo, stitchRatio, k, h, nodeAttr);
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