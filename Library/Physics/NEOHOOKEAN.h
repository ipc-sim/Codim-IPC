#pragma once

#include <meta/meta.hpp>

#include <Math/VECTOR.h>
#include <Math/SINGULAR_VALUE_DECOMPOSITION.h>
#include <FEM/DATA_TYPE.h>
#include <numeric>

namespace py = pybind11;
namespace JGSL {

template<class T>
using NEOHOOKEAN_2 = BASE_STORAGE<MATRIX<T, 2>, T, T, T>;

template<class T>
using NEOHOOKEAN_3 = BASE_STORAGE<MATRIX<T, 3>, T, T, T>;

template<class T, int dim>
using NEOHOOKEAN = meta::if_<
    meta::equal_to<meta::int_<dim>, meta::int_<2>>,
    NEOHOOKEAN_2<T>, // F, VOL, LAMBDA, MU, _padding_
    NEOHOOKEAN_3<T>  // F, VOL, LAMBDA, MU
>;

template <class T, int dim>
struct NH_HELPER {
    static void Insert(NEOHOOKEAN<T, dim>& nh, int i, T vol, T lambda, T mu) {
        nh.Insert(i, MATRIX<T, dim>(1), vol, lambda, mu);
    }
};

template <class T, int dim>
class NEOHOOKEAN_FUNCTOR {
public:
    using STORAGE = NEOHOOKEAN<T, dim>;
    using DIFFERENTIAL = BASE_STORAGE<Eigen::Matrix<T, dim*dim, dim*dim>>;
    static const bool useJ = false;
    static const bool projectable = true;

    static std::unique_ptr<STORAGE> Create() {
        return std::make_unique<STORAGE>();
    }

    static void Append(STORAGE& nh, const VECTOR<int, 2>& handle, T vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = handle(0); i < handle(1); ++i)
            NH_HELPER<T, dim>::Insert(nh, i, vol, lambda, mu);
    }

    static void Append_FEM(STORAGE& nh, const VECTOR<int, 4>& handle, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = handle(1); i < handle(3); ++i)
            NH_HELPER<T, dim>::Insert(nh, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void Append_All_FEM(STORAGE& nh, const VECTOR<int, 4>& handle, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = 0; i < handle(3); ++i)
            NH_HELPER<T, dim>::Insert(nh, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void All_Append_FEM(STORAGE& nh, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = 0; i < vol.size; ++i)
            NH_HELPER<T, dim>::Insert(nh, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void Compute_Kirchoff_Stress(STORAGE& nh, MPM_STRESS<T, dim>& stress) {
        puts("MPM related func, not implemented");
        exit(0);
    }

    static void Compute_Psi(STORAGE& nh, T w, MESH_ELEM_ATTR<T, dim>& elemAttr, T& Psi) {
        TIMER_FLAG("computePsi");
        std::vector<T> psi(elemAttr.size);
        nh.Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu] = data;

            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime

            Compute_E(sigma, mu, lambda, psi[i]);
            psi[i] *= w * vol;
        });

        Psi = std::accumulate(psi.begin(), psi.end(), Psi);
    }

    static void Compute_Psi(STORAGE& nh, T w, T& Psi) {
        puts("MPM related func, not implemented");
        exit(0);
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& nh, T w, MESH_ELEM_ATTR<T, dim>& elemAttr) {
        TIMER_FLAG("computeStress");
        nh.Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, u] = data;
            T J = F.determinant();
            MATRIX<T, dim> JFinvT = F.cofactor();
            MATRIX<T, dim> FInvT =  JFinvT / J;
            elemAttr.template Update_Component<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(i, w * vol * (u * (F - FInvT) + lambda * std::log(J) * FInvT));
        });
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& nh, T w, BASE_STORAGE<MATRIX<T, dim>>& P) {
        puts("MPM related func, not implemented");
        exit(0);
    }

    static void Compute_First_PiolaKirchoff_Stress_Derivative(STORAGE& nh, T w, bool projectSPD, DIFFERENTIAL& dP_div_dF) {
        TIMER_FLAG("computeStressDerivative");
        if(dP_div_dF.size != nh.size) {
            dP_div_dF.Reserve(nh.size);
            nh.Each([&](const int& idx, auto data)
            {
                dP_div_dF.Insert(idx, Eigen::Matrix<T, dim*dim, dim*dim>::Zero());
            });
        }
        nh.Join(dP_div_dF).Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu, dP_div_dFI] = data;

            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime
            
            Compute_DP_Div_DF(U, sigma, V, mu, lambda, w * vol, projectSPD, dP_div_dFI);
        });
    }

    static void Compute_E(const VECTOR<T, dim>& sigma, T u, T lambda, T& E)
    {
        const T sigma2Sum = sigma.length2();
        const T sigmaProd = sigma.prod();
        const T log_sigmaProd = std::log(sigmaProd);
        E = u / 2.0 * (sigma2Sum - dim) - (u - lambda / 2.0 * log_sigmaProd) * log_sigmaProd;
    }

    static void Compute_DE_Div_DSigma(const VECTOR<T, dim>& singularValues, T u, T lambda,
        VECTOR<T, dim>& dE_div_dsigma) 
    {
        const double log_sigmaProd = std::log(singularValues.prod());
        const double inv0 = 1.0 / singularValues[0];
        dE_div_dsigma[0] = u * (singularValues[0] - inv0) + lambda * inv0 * log_sigmaProd;
        const double inv1 = 1.0 / singularValues[1];
        dE_div_dsigma[1] = u * (singularValues[1] - inv1) + lambda * inv1 * log_sigmaProd;
        if constexpr (dim == 3) {
            const double inv2 = 1.0 / singularValues[2];
            dE_div_dsigma[2] = u * (singularValues[2] - inv2) + lambda * inv2 * log_sigmaProd;
        }
    }

    static void Compute_D2E_Div_DSigma2(const VECTOR<T, dim>& singularValues, T u, T lambda,
        MATRIX<T, dim>& d2E_div_dsigma2)
    {
        const double log_sigmaProd = std::log(singularValues.prod());
        const double inv2_0 = 1.0 / singularValues[0] / singularValues[0];
        d2E_div_dsigma2(0, 0) = u * (1.0 + inv2_0) - lambda * inv2_0 * (log_sigmaProd - 1.0);
        const double inv2_1 = 1.0 / singularValues[1] / singularValues[1];
        d2E_div_dsigma2(1, 1) = u * (1.0 + inv2_1) - lambda * inv2_1 * (log_sigmaProd - 1.0);
        d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda / singularValues[0] / singularValues[1];
        if constexpr (dim == 3) {
            const double inv2_2 = 1.0 / singularValues[2] / singularValues[2];
            d2E_div_dsigma2(2, 2) = u * (1.0 + inv2_2) - lambda * inv2_2 * (log_sigmaProd - 1.0);
            d2E_div_dsigma2(1, 2) = d2E_div_dsigma2(2, 1) = lambda / singularValues[1] / singularValues[2];
            d2E_div_dsigma2(2, 0) = d2E_div_dsigma2(0, 2) = lambda / singularValues[2] / singularValues[0];
        }
    }

    static void Compute_BLeftCoef(const VECTOR<T, dim>& singularValues, T u, T lambda,
        VECTOR<T, dim*(dim - 1) / 2>& BLeftCoef)
    {
        //TODO: right coef also has analytical form
        const double sigmaProd = singularValues.prod();
        if constexpr (dim == 2) {
            BLeftCoef[0] = (u + (u - lambda * std::log(sigmaProd)) / sigmaProd) / 2.0;
        }
        else {
            const double middle = u - lambda * std::log(sigmaProd);
            BLeftCoef[0] = (u + middle / singularValues[0] / singularValues[1]) / 2.0;
            BLeftCoef[1] = (u + middle / singularValues[1] / singularValues[2]) / 2.0;
            BLeftCoef[2] = (u + middle / singularValues[2] / singularValues[0]) / 2.0;
        }
    }

    template <class MAT>
    static void Compute_DP_Div_DF(const MATRIX<T, dim>& U, const VECTOR<T, dim>& sigma, const MATRIX<T, dim>& V,
        T u, T lambda, T w, bool projectSPD,
        MAT& dP_div_dF)
    {
        // compute A
        VECTOR<T, dim> dE_div_dsigma;
        Compute_DE_Div_DSigma(sigma, u, lambda, dE_div_dsigma);
        MATRIX<T, dim> d2E_div_dsigma2;
        Compute_D2E_Div_DSigma2(sigma, u, lambda, d2E_div_dsigma2);
        if (projectSPD) {
            d2E_div_dsigma2.makePD();
        }

        // compute B
        const int Cdim2 = dim * (dim - 1) / 2;
        VECTOR<T, Cdim2> BLeftCoef;
        Compute_BLeftCoef(sigma, u, lambda, BLeftCoef);
        MATRIX<T, 2> B[Cdim2];
        for (int cI = 0; cI < Cdim2; cI++) {
            int cI_post = (cI + 1) % dim;

            T rightCoef = dE_div_dsigma[cI] + dE_div_dsigma[cI_post];
            T sum_sigma = sigma[cI] + sigma[cI_post];
            const T eps = 1.0e-6;
            if (sum_sigma < eps) {
                rightCoef /= 2 * eps;
            }
            else {
                rightCoef /= 2 * sum_sigma;
            }

            const T& leftCoef = BLeftCoef[cI];
            B[cI](0, 0) = B[cI](1, 1) = leftCoef + rightCoef;
            B[cI](0, 1) = B[cI](1, 0) = leftCoef - rightCoef;
            if (projectSPD) {
                B[cI].makePD();
            }
        }

        // compute M using A(d2E_div_dsigma2) and B
        MAT M;
        M.setZero();
        if constexpr (dim == 2) {
            M(0, 0) = w * d2E_div_dsigma2(0, 0);
            M(0, 3) = w * d2E_div_dsigma2(0, 1);
            M(1, 1) = w * B[0](0, 0);
            M(1, 2) = w * B[0](0, 1);
            M(2, 1) = w * B[0](1, 0);
            M(2, 2) = w * B[0](1, 1);
            M(3, 0) = w * d2E_div_dsigma2(1, 0);
            M(3, 3) = w * d2E_div_dsigma2(1, 1);
        }
        else {
            // A
            M(0, 0) = w * d2E_div_dsigma2(0, 0);
            M(0, 4) = w * d2E_div_dsigma2(0, 1);
            M(0, 8) = w * d2E_div_dsigma2(0, 2);
            M(4, 0) = w * d2E_div_dsigma2(1, 0);
            M(4, 4) = w * d2E_div_dsigma2(1, 1);
            M(4, 8) = w * d2E_div_dsigma2(1, 2);
            M(8, 0) = w * d2E_div_dsigma2(2, 0);
            M(8, 4) = w * d2E_div_dsigma2(2, 1);
            M(8, 8) = w * d2E_div_dsigma2(2, 2);
            // B01
            M(1, 1) = w * B[0](0, 0);
            M(1, 3) = w * B[0](0, 1);
            M(3, 1) = w * B[0](1, 0);
            M(3, 3) = w * B[0](1, 1);
            // B12
            M(5, 5) = w * B[1](0, 0);
            M(5, 7) = w * B[1](0, 1);
            M(7, 5) = w * B[1](1, 0);
            M(7, 7) = w * B[1](1, 1);
            // B20
            M(2, 2) = w * B[2](1, 1);
            M(2, 6) = w * B[2](1, 0);
            M(6, 2) = w * B[2](0, 1);
            M(6, 6) = w * B[2](0, 0);
        }

        // compute dP_div_dF
        auto& wdP_div_dF = dP_div_dF;
        for (int j = 0; j < dim; j++) {
            int _dim_j = j * dim;
            for (int i = 0; i < dim; i++) {
                int ij = _dim_j + i;
                for (int s = 0; s < dim; s++) {
                    int _dim_s = s * dim;
                    for (int r = 0; r < dim; r++) {
                        int rs = _dim_s + r;
                        if (ij > rs) {
                            // bottom left, same as upper right
                            continue;
                        }

                        if constexpr (dim == 2) {
                            wdP_div_dF(ij, rs) = M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) + M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) + M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) + M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) + M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) + M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) + M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) + M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
                        }
                        else {
                            wdP_div_dF(ij, rs) = M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) + M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) + M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) + M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) + M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) + M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) + M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) + M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) + M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) + M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) + M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) + M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) + M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) + M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) + M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) + M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) + M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) + M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) + M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) + M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) + M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
                        }

                        if (ij < rs) {
                            wdP_div_dF(rs, ij) = wdP_div_dF(ij, rs);
                        }
                    }
                }
            }
        }
    }

    static void Check_Hessian()
    {
        T E = 1, nu = 0.3;
        T epsilon = T(1e-6);
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        Eigen::Matrix<T, dim, dim> eigenF = Eigen::Matrix<T, dim, dim>::Random();
        while (std::abs(eigenF.determinant()) < 0.1)
        {
            eigenF = Eigen::Matrix<T, dim, dim>::Random();
        }
        MATRIX<T, dim> F;
        for (int i = 0; i < dim; ++i) 
            for (int j = 0; j < dim; ++j)
                F(i,j) = eigenF(i,j);
        Eigen::Matrix<T, dim*dim, dim*dim> dPdF;
        Eigen::Matrix<T, dim*dim, dim*dim> dPdF_FD; 
        dPdF_FD.setZero();
        MATRIX<T, dim> U(1), V(1);
        VECTOR<T, dim> sigma;
        Singular_Value_Decomposition(F, U, sigma, V);
        for (int i = 0; i < dim; ++i)
            if (sigma(i) < 0) sigma(i) = -sigma(i);
        F = U * MATRIX<T, dim>(sigma) * V.transpose();
        T phi;
        Compute_E(sigma, mu, lambda, phi);
        Compute_DP_Div_DF(U, sigma, V, mu, lambda, 1, false, dPdF);
        for (int i = 0; i < dim; ++i) { for (int j = 0; j < dim; ++j) { for (int k = 0; k < dim; ++k) { for (int l = 0; l < dim; ++l) {
            int di = i + j * dim;
            int dj = k + l * dim;
            T phi_ij, phi_kl, phi_ijkl;
            auto F_move_ij = F;
            F_move_ij(i,j) += epsilon;
            U = MATRIX<T, dim>(1); sigma.setZero(); V = MATRIX<T, dim>(1);
            Singular_Value_Decomposition(F_move_ij, U, sigma, V);
            Compute_E(sigma, mu, lambda, phi_ij);
            auto F_move_kl = F;
            F_move_kl(k,l) += epsilon;
            U = MATRIX<T, dim>(1); sigma.setZero(); V = MATRIX<T, dim>(1);
            Singular_Value_Decomposition(F_move_kl, U, sigma, V);
            Compute_E(sigma, mu, lambda, phi_kl);
            auto F_move_ijkl = F;
            F_move_ijkl(i,j) += epsilon;
            F_move_ijkl(k,l) += epsilon;
            U = MATRIX<T, dim>(1); sigma.setZero(); V = MATRIX<T, dim>(1);
            Singular_Value_Decomposition(F_move_ijkl, U, sigma, V);
            Compute_E(sigma, mu, lambda, phi_ijkl);
            dPdF_FD(di, dj) = (phi_ijkl - phi_ij - phi_kl + phi) / (epsilon * epsilon);
        }}}}
        for (int i = 0; i < dim * dim; ++i) 
            for (int j = 0; j < dim * dim; ++j)
                std::cout << dPdF(i,j) << " " << dPdF_FD(i,j) << std::endl;
    }

    static void Check_Gradient()
    {
        T E = 1, nu = 0.3;
        T epsilon = T(1e-6);
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        Eigen::Matrix<T, dim, dim> eigenF = Eigen::Matrix<T, dim, dim>::Random();
        while (std::abs(eigenF.determinant()) < 0.1)
        {
            eigenF = Eigen::Matrix<T, dim, dim>::Random();
        }
        MATRIX<T, dim> F;
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                F(i,j) = eigenF(i,j);
        MATRIX<T, dim> U(1), V(1);
        VECTOR<T, dim> sigma;
        Singular_Value_Decomposition(F, U, sigma, V);

        for (int i = 0; i < dim; ++i)
            if (sigma(i) < 0) sigma(i) = -sigma(i);
        F = U * MATRIX<T, dim>(sigma) * V.transpose();

        VECTOR<T, dim> dE_div_dsigma;
        Compute_DE_Div_DSigma(sigma, mu, lambda, dE_div_dsigma);
        MATRIX<T, dim> P = U * MATRIX<T, dim >(dE_div_dsigma) * V.transpose();
        T phi;
        Compute_E(sigma, mu, lambda, phi);
        MATRIX<T, dim> P_FD;
        for (int i = 0; i < dim; ++i) { for (int j = 0; j < dim; ++j) {
                T phi_eps;
                auto F_moved = F;
                F_moved(i,j) += epsilon;
                U = MATRIX<T, dim>(1); sigma.setZero(); V = MATRIX<T, dim>(1);
                Singular_Value_Decomposition(F_moved, U, sigma, V);
                Compute_E(sigma, mu, lambda, phi_eps);
                P_FD(i,j) = (phi_eps - phi) / epsilon;
            }}
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                std::cout << P(i,j) << " " << P_FD(i,j) << std::endl;
    }
};

}
