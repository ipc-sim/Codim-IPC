#pragma once

namespace JGSL {

template <class T>
T eta(T input, T inputMax)
{
#ifdef ORTHO_STVK
    return input;
#else
    return -inputMax * std::log((inputMax - input) / inputMax);
#endif
}
template <class T>
T eta_g(T input, T inputMax)
{
#ifdef ORTHO_STVK
    return 1;
#else
    return inputMax / (inputMax - input);
#endif
}
template <class T>
T eta_H(T input, T inputMax)
{
#ifdef ORTHO_STVK
    return 0;
#else
    return inputMax / ((inputMax - input) * (inputMax - input));
#endif
}

template<class T, int dim>
bool Check_Fiber_Feasibility(
    MESH_ELEM<dim - 1>& Elem, T h,
    const VECTOR<T, 3>& fiberLimit,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
#ifdef ORTHO_STVK
    return true;
#else
    TIMER_FLAG("Check_Fiber_Feasibility");
    bool feasible = true;
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F_, vol, lambda, mu] = data;
            if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                const VECTOR<T, dim>& X1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& X2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& X3 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[2]));

                Eigen::Matrix<T, dim - 1, dim - 1> B;
                B(0, 0) = X2[0] - X1[0];
                B(1, 0) = X2[2] - X1[2];
                B(0, 1) = X3[0] - X1[0];
                B(1, 1) = X3[2] - X1[2];
                Eigen::Matrix<T, dim, dim - 1> A;
                for (int d = 0; d < 3; ++d) { 
                    A(d, 0) = x2[d] - x1[d];
                    A(d, 1) = x3[d] - x1[d];
                }
                Eigen::Matrix<T, dim, dim - 1> F = A * B.inverse();
                Eigen::Matrix<T, dim - 1, dim - 1> tildeE = 0.5 * (F.transpose() * F - Eigen::Matrix<T, dim - 1, dim - 1>::Identity());
                
                if (std::abs(tildeE(0, 0)) >= fiberLimit[0] ||
                    std::abs(tildeE(1, 1)) >= fiberLimit[1] ||
                    std::abs(tildeE(0, 1)) >= fiberLimit[2])
                {
                    feasible = false;
                }
            }
        });
    }
    return feasible;
#endif
}

template<class T, int dim>
void Compute_Fiber_Energy(
    MESH_ELEM<dim - 1>& Elem, T h,
    const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    T& E)
{
    TIMER_FLAG("Compute_Fiber_Energy");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F_, vol, lambda, mu] = data;
            if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                const VECTOR<T, dim>& X1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& X2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& X3 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[2]));

                Eigen::Matrix<T, dim - 1, dim - 1> B;
                B(0, 0) = X2[0] - X1[0];
                B(1, 0) = X2[2] - X1[2];
                B(0, 1) = X3[0] - X1[0];
                B(1, 1) = X3[2] - X1[2];
                Eigen::Matrix<T, dim, dim - 1> A;
                for (int d = 0; d < 3; ++d) { 
                    A(d, 0) = x2[d] - x1[d];
                    A(d, 1) = x3[d] - x1[d];
                }
                Eigen::Matrix<T, dim, dim - 1> F = A * B.inverse();
                Eigen::Matrix<T, dim - 1, dim - 1> tildeE = 0.5 * (F.transpose() * F - Eigen::Matrix<T, dim - 1, dim - 1>::Identity());
                
                E += h * h * vol * (fiberStiffMult[0] / 2 * eta(tildeE(0, 0) * tildeE(0, 0), fiberLimit[0] * fiberLimit[0]) + 
                    fiberStiffMult[1] * eta(tildeE(0, 0) * tildeE(1, 1), fiberLimit[0] * fiberLimit[1]) +
                    fiberStiffMult[2] / 2 * eta(tildeE(1, 1) * tildeE(1, 1), fiberLimit[1] * fiberLimit[1]) + 
                    fiberStiffMult[3] * eta(tildeE(0, 1) * tildeE(0, 1), fiberLimit[2] * fiberLimit[2]));
            }
        });
    }
}

template<class T, int dim>
void Compute_Fiber_Gradient(
    MESH_ELEM<dim - 1>& Elem, T h,
    const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    TIMER_FLAG("Compute_Fiber_Gradient");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F_, vol, lambda, mu] = data;
            if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                const VECTOR<T, dim>& X1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& X2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& X3 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[2]));

                Eigen::Matrix<T, dim - 1, dim - 1> B;
                B(0, 0) = X2[0] - X1[0];
                B(1, 0) = X2[2] - X1[2];
                B(0, 1) = X3[0] - X1[0];
                B(1, 1) = X3[2] - X1[2];
                Eigen::Matrix<T, dim - 1, dim - 1> IB = B.inverse();
                Eigen::Matrix<T, dim, dim - 1> A;
                for (int d = 0; d < 3; ++d) { 
                    A(d, 0) = x2[d] - x1[d];
                    A(d, 1) = x3[d] - x1[d];
                }
                Eigen::Matrix<T, dim, dim - 1> F = A * IB;
                Eigen::Matrix<T, dim - 1, dim - 1> tildeE = 0.5 * (F.transpose() * F - Eigen::Matrix<T, dim - 1, dim - 1>::Identity());

                Eigen::Matrix<T, 1, 4> dpsi_div_dtildeE;
                T eg_tE0011 = eta_g(tildeE(0, 0) * tildeE(1, 1), fiberLimit[0] * fiberLimit[1]);
                dpsi_div_dtildeE[0] = fiberStiffMult[0] * eta_g(tildeE(0, 0) * tildeE(0, 0), fiberLimit[0] * fiberLimit[0]) * tildeE(0, 0) + 
                    fiberStiffMult[1] * eg_tE0011 * tildeE(1, 1);
                dpsi_div_dtildeE[1] = 0;
                dpsi_div_dtildeE[2] = 2 * fiberStiffMult[3] * eta_g(tildeE(0, 1) * tildeE(0, 1), fiberLimit[2] * fiberLimit[2]) * tildeE(0, 1);
                dpsi_div_dtildeE[3] = fiberStiffMult[2] * eta_g(tildeE(1, 1) * tildeE(1, 1), fiberLimit[1] * fiberLimit[1]) * tildeE(1, 1) +
                    fiberStiffMult[1] * eg_tE0011 * tildeE(0, 0);
                dpsi_div_dtildeE *= h * h * vol;
                
                Eigen::Matrix<T, 4, 6> dtildeE_div_dF;
                dtildeE_div_dF.setZero();
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            dtildeE_div_dF(i + j * 2, k + i * 3) += 0.5 * F(k, j);
                            dtildeE_div_dF(i + j * 2, k + j * 3) += 0.5 * F(k, i);
                        }
                    }
                }

                Eigen::Matrix<T, 1, 6> dW_div_dF_vec = dpsi_div_dtildeE * dtildeE_div_dF;
                Eigen::Matrix<T, dim, dim - 1> dW_div_dF;
                dW_div_dF.col(0) = dW_div_dF_vec.template segment<dim>(0).transpose();
                dW_div_dF.col(1) = dW_div_dF_vec.template segment<dim>(3).transpose();

                Eigen::Matrix<T, dim * 3, 1> grad;
                grad.template segment<dim>(0) = -(IB(0, 0) + IB(1, 0)) * dW_div_dF.col(0) - (IB(0, 1) + IB(1, 1)) * dW_div_dF.col(1);
                grad.template segment<dim>(3) = IB(0, 0) * dW_div_dF.col(0) + IB(0, 1) * dW_div_dF.col(1);
                grad.template segment<dim>(6) = IB(1, 0) * dW_div_dF.col(0) + IB(1, 1) * dW_div_dF.col(1);
                for (int v = 0; v < 3; ++v) {
                    VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[v]));
                    for (int d = 0; d < dim; ++d) {
                        g[d] += grad[v * dim + d];
                    }
                }
            }
        });
    }
}

template<class T, int dim>
void Compute_Fiber_Hessian(
    MESH_ELEM<dim - 1>& Elem, 
    T h, bool projectSPD,
    const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Fiber_Hessian");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        std::vector<int> tripletStartInd(Elem.size);
        int tstartind = triplets.size();
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
                tripletStartInd[id] = tstartind;
                tstartind += 81;
            }
            else {
                tripletStartInd[id] = -1;
            }
        });
        triplets.resize(tstartind);

        Elem.Join(elasticityAttr).Par_Each([&](int id, auto data) {
            auto &[elemVInd, F_, vol, lambda, mu] = data;
            if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                const VECTOR<T, dim>& X1 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& X2 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& X3 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::x0>(nodeAttr.Get_Unchecked(elemVInd[2]));

                Eigen::Matrix<T, dim - 1, dim - 1> B;
                B(0, 0) = X2[0] - X1[0];
                B(1, 0) = X2[2] - X1[2];
                B(0, 1) = X3[0] - X1[0];
                B(1, 1) = X3[2] - X1[2];
                Eigen::Matrix<T, dim - 1, dim - 1> IB = B.inverse();
                Eigen::Matrix<T, dim, dim - 1> A;
                for (int d = 0; d < 3; ++d) { 
                    A(d, 0) = x2[d] - x1[d];
                    A(d, 1) = x3[d] - x1[d];
                }
                Eigen::Matrix<T, dim, dim - 1> F = A * IB;
                Eigen::Matrix<T, dim - 1, dim - 1> tildeE = 0.5 * (F.transpose() * F - Eigen::Matrix<T, dim - 1, dim - 1>::Identity());

                Eigen::Matrix<T, 1, 4> dpsi_div_dtildeE;
                T tE0000 = tildeE(0, 0) * tildeE(0, 0);
                T tE0011 = tildeE(0, 0) * tildeE(1, 1);
                T tE0101 = tildeE(0, 1) * tildeE(0, 1);
                T tE1111 = tildeE(1, 1) * tildeE(1, 1);
                T lim0000 = fiberLimit[0] * fiberLimit[0];
                T lim1111 = fiberLimit[1] * fiberLimit[1];
                T lim0011 = fiberLimit[0] * fiberLimit[1];
                T lim0101 = fiberLimit[2] * fiberLimit[2];
                T w = h * h * vol;
                dpsi_div_dtildeE[0] = fiberStiffMult[0] * eta_g(tE0000, lim0000) * tildeE(0, 0) + 
                    fiberStiffMult[1] * eta_g(tE0011, lim0011) * tildeE(1, 1);
                dpsi_div_dtildeE[1] = 0;
                dpsi_div_dtildeE[2] = 2 * fiberStiffMult[3] * eta_g(tE0101, lim0101) * tildeE(0, 1);
                dpsi_div_dtildeE[3] = fiberStiffMult[2] * eta_g(tE1111, lim1111) * tildeE(1, 1) +
                    fiberStiffMult[1] * eta_g(tE0011, lim0011) * tildeE(0, 0);
                dpsi_div_dtildeE *= w;
                Eigen::Matrix<T, 4, 4> d2psi_div_dtildeE2 = Eigen::Matrix<T, 4, 4>::Zero();
                d2psi_div_dtildeE2(0, 0) = w * (2 * fiberStiffMult[0] * eta_H(tE0000, lim0000) * tE0000 +
                    fiberStiffMult[0] * eta_g(tE0000, lim0000) + 
                    fiberStiffMult[1] * eta_H(tE0011, lim0011) * tE1111);
                d2psi_div_dtildeE2(0, 3) = d2psi_div_dtildeE2(3, 0) = w * fiberStiffMult[1] * (eta_H(tE0011, lim0011) * tE0011 + eta_g(tE0011, lim0011));
                d2psi_div_dtildeE2(3, 3) = w * (2 * fiberStiffMult[2] * eta_H(tE1111, lim1111) * tE1111 +
                    fiberStiffMult[2] * eta_g(tE1111, lim1111) + 
                    fiberStiffMult[1] * eta_H(tE0011, lim0011) * tE0000);
                d2psi_div_dtildeE2(2, 2) = w * 2 * fiberStiffMult[3] * (2 * eta_H(tE0101, lim0101) * tE0101 + eta_g(tE0101, lim0101));

                Eigen::Matrix<T, 4, 6> dtildeE_div_dF;
                Eigen::Matrix<T, 6, 6> d2tildeE_div_dF2[4];
                dtildeE_div_dF.setZero();
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        d2tildeE_div_dF2[i + j * 2].setZero();
                        for (int k = 0; k < 3; ++k) {
                            dtildeE_div_dF(i + j * 2, k + i * 3) += 0.5 * F(k, j);
                            dtildeE_div_dF(i + j * 2, k + j * 3) += 0.5 * F(k, i);

                            d2tildeE_div_dF2[i + j * 2](k + i * 3, k + j * 3) += 0.5;
                            d2tildeE_div_dF2[i + j * 2](k + j * 3, k + i * 3) += 0.5;
                        }
                    }
                }

                Eigen::Matrix<T, 6, 6> d2W_div_dF2 = dtildeE_div_dF.transpose() * d2psi_div_dtildeE2 * dtildeE_div_dF;
                for (int i = 0; i < 4; ++i) {
                    d2W_div_dF2 += dpsi_div_dtildeE[i] * d2tildeE_div_dF2[i];
                }
                if (projectSPD) {
                    makePD(d2W_div_dF2);
                }

                Eigen::Matrix<T, 9, 6> intermediate;
                for (int colI = 0; colI < 6; ++colI) {
                    intermediate.col(colI).template segment<dim>(0) = -(IB(0, 0) + IB(1, 0)) * d2W_div_dF2.col(colI).template segment<dim>(0) - 
                        (IB(0, 1) + IB(1, 1)) * d2W_div_dF2.col(colI).template segment<dim>(3);
                    intermediate.col(colI).template segment<dim>(3) = IB(0, 0) * d2W_div_dF2.col(colI).template segment<dim>(0) + 
                        IB(0, 1) * d2W_div_dF2.col(colI).template segment<dim>(3);
                    intermediate.col(colI).template segment<dim>(6) = IB(1, 0) * d2W_div_dF2.col(colI).template segment<dim>(0) + 
                        IB(1, 1) * d2W_div_dF2.col(colI).template segment<dim>(3);
                }
                Eigen::Matrix<T, 9, 9> Hessian;
                for (int i = 0; i < 9; ++i) {
                    Hessian.row(i).template segment<dim>(0) = -(IB(0, 0) + IB(1, 0)) * intermediate.row(i).template segment<dim>(0) - 
                        (IB(0, 1) + IB(1, 1)) * intermediate.row(i).template segment<dim>(3);
                    Hessian.row(i).template segment<dim>(3) = IB(0, 0) * intermediate.row(i).template segment<dim>(0) + 
                        IB(0, 1) * intermediate.row(i).template segment<dim>(3);
                    Hessian.row(i).template segment<dim>(6) = IB(1, 0) * intermediate.row(i).template segment<dim>(0) + 
                        IB(1, 1) * intermediate.row(i).template segment<dim>(3);
                }
                
                int startInd = tripletStartInd[id];
                for (int i = 0; i < 3; ++i) {
                    for (int id = 0; id < 3; ++id) {
                        for (int j = 0; j < 3; ++j) {
                            for (int jd = 0; jd < 3; ++jd) {
                                triplets[startInd + (i * 3 + id) * 9 + j * 3 + jd] = std::move(
                                    Eigen::Triplet<T>(elemVInd[i] * 3 + id, elemVInd[j] * 3 + jd, 
                                    Hessian(i * 3 + id, j * 3 + jd)));
                            }
                        }
                    }
                }
            }
        });
    }
}

template <class T, int dim>
void Check_Fiber_Gradient(
    MESH_ELEM<dim - 1>& Elem,
    const VECTOR_STORAGE<T, dim + 1>& DBC,
    T h, const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    T eps = 1.0e-6;

    T E0 = 0;
    Compute_Fiber_Energy(Elem, 1.0, fiberStiffMult, fiberLimit, DBCb, X, nodeAttr, elemAttr, elasticityAttr, E0); 
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Fiber_Gradient(Elem, 1.0, fiberStiffMult, fiberLimit, DBCb, X, nodeAttr, elemAttr, elasticityAttr); 

    std::vector<T> grad_FD(X.size * dim);
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        T E = 0;
        Compute_Fiber_Energy(Elem, 1.0, fiberStiffMult, fiberLimit, DBCb, Xperturb, nodeAttr, elemAttr, elasticityAttr, E);
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
    printf("err_abs = %le, err_rel = %le\n", err, err / norm);
}

template <class T, int dim>
void Check_Fiber_Hessian(
    MESH_ELEM<dim - 1>& Elem,
    const VECTOR_STORAGE<T, dim + 1>& DBC, 
    T h, const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    T eps = 1.0e-6;

    MESH_NODE_ATTR<T, dim> nodeAttr0;
    nodeAttr.deep_copy_to(nodeAttr0);
    nodeAttr0.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Fiber_Gradient(Elem, 1.0, fiberStiffMult, fiberLimit, DBCb, X, nodeAttr0, elemAttr, elasticityAttr);
    std::vector<Eigen::Triplet<T>> HStriplets;
    Compute_Fiber_Hessian(Elem, 1.0, false, fiberStiffMult, fiberLimit, DBCb, X, nodeAttr, elemAttr, elasticityAttr, HStriplets); 
    CSR_MATRIX<T> HS;
    HS.Construct_From_Triplet(X.size * dim, X.size * dim, HStriplets);

    std::vector<Eigen::Triplet<T>> HFDtriplets;
    HFDtriplets.reserve(HStriplets.size());
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
        Compute_Fiber_Gradient(Elem, 1.0, fiberStiffMult, fiberLimit, DBCb, Xperturb, nodeAttr, elemAttr, elasticityAttr);
        for (int vI = 0; vI < X.size; ++vI) {
            const VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(vI));
            const VECTOR<T, dim>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr0.Get_Unchecked(vI));
            const VECTOR<T, dim> hFD = (g - g0) / eps;
            if (hFD.length2() > eps) {
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
    printf("err_abs = %le, err_rel = %le\n", err, err / norm);
}

}