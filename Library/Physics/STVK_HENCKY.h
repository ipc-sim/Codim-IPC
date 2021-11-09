#pragma once

#include <meta/meta.hpp>

#include <Math/VECTOR.h>
#include <Math/SINGULAR_VALUE_DECOMPOSITION.h>
#include <FEM/DATA_TYPE.h>
#include <numeric>

namespace py = pybind11;
namespace JGSL {

template<class T>
using STVK_HENCKY_2 = BASE_STORAGE<MATRIX<T, 2>, T, T, T>;

template<class T>
using STVK_HENCKY_3 = BASE_STORAGE<MATRIX<T, 3>, T, T, T>;

template<class T, int dim>
using STVK_HENCKY = meta::if_<
meta::equal_to<meta::int_<dim>, meta::int_<2>>,
STVK_HENCKY_2<T>, // F, VOL, LAMBDA, MU, _padding_
STVK_HENCKY_3<T>  // F, VOL, LAMBDA, MU
>;

template <class T, int dim>
struct SH_HELPER {
    static void Insert(STVK_HENCKY<T, dim>& sh, int i, T vol, T lambda, T mu) {
        sh.Insert(i, MATRIX<T, dim>(1), vol, lambda, mu);
    }
};

template <class T, int dim>
class STVK_HENCKY_FUNCTOR {
public:
    using STORAGE = STVK_HENCKY<T, dim>;
    using DIFFERENTIAL = BASE_STORAGE<Eigen::Matrix<T, dim*dim, dim*dim>>;
    static const bool useJ = false;
    static const bool projectable = true;

    static std::unique_ptr<STORAGE> Create() {
        return std::make_unique<STORAGE>();
    }

    static void Append(STORAGE& sh, const VECTOR<int, 2>& handle, T vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = handle(0); i < handle(1); ++i)
            SH_HELPER<T, dim>::Insert(sh, i, vol, lambda, mu);
    }

    static void Append_FEM(STORAGE& sh, const VECTOR<int, 4>& handle, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = handle(1); i < handle(3); ++i)
            SH_HELPER<T, dim>::Insert(sh, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void Append_All_FEM(STORAGE& sh, const VECTOR<int, 4>& handle, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = 0; i < handle(3); ++i)
            SH_HELPER<T, dim>::Insert(sh, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void All_Append_FEM(STORAGE& sh, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = 0; i < vol.size; ++i)
            SH_HELPER<T, dim>::Insert(sh, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void Compute_Kirchoff_Stress(STORAGE& sh, MPM_STRESS<T, dim>& stress) {
        puts("MPM related func, not implemented");
        exit(0);
    }

    static void Compute_Psi(STORAGE& sh, T w, MESH_ELEM_ATTR<T, dim>& elemAttr, T& Psi) {
        TIMER_FLAG("computePsi");
        std::vector<T> psi(elemAttr.size);
        sh.Par_Each([&](const int i, auto data) {
          auto& [F, vol, lambda, mu] = data;

          MATRIX<T, dim> U(1), V(1);
          VECTOR<T, dim> sigma;
          Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime

          Compute_E(sigma, mu, lambda, psi[i]);
          psi[i] *= w * vol;
        });

        Psi = std::accumulate(psi.begin(), psi.end(), Psi);
    }

    static void Compute_Psi(STORAGE& sh, T w, T& Psi) {
        puts("MPM related func, not implemented");
        exit(0);
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& sh, T w, MESH_ELEM_ATTR<T, dim>& elemAttr) {
        TIMER_FLAG("computeStress");
        sh.Par_Each([&](const int i, auto data) {
          auto& [F, vol, lambda, u] = data;

          MATRIX<T, dim> U(1), V(1);
          VECTOR<T, dim> sigma;
          Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime

          VECTOR<T, dim> dE_div_dsigma;
          Compute_DE_Div_DSigma(sigma, u, lambda, dE_div_dsigma);

          elemAttr.template Update_Component<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(i, w * vol * (U * MATRIX<T, dim >(dE_div_dsigma) * V.transpose()));
        });
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& sh, T w, BASE_STORAGE<MATRIX<T, dim>>& P) {
        puts("MPM related func, not implemented");
        exit(0);
    }

    static void Compute_First_PiolaKirchoff_Stress_Derivative(STORAGE& sh, T w, bool projectSPD, DIFFERENTIAL& dP_div_dF) {
        TIMER_FLAG("computeStressDerivative");
        if(dP_div_dF.size != sh.size) {
            dP_div_dF.Reserve(sh.size);
            sh.Each([&](const int& idx, auto data)
            {
              dP_div_dF.Insert(idx, Eigen::Matrix<T, dim*dim, dim*dim>::Zero());
            });
        }
        sh.Join(dP_div_dF).Par_Each([&](const int i, auto data) {
          auto& [F, vol, lambda, mu, dP_div_dFI] = data;

          MATRIX<T, dim> U(1), V(1);
          VECTOR<T, dim> sigma;
          Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime

          Compute_DP_Div_DF(U, sigma, V, mu, lambda, w * vol, projectSPD, dP_div_dFI);
        });
    }

    static void Compute_E(const VECTOR<T, dim>& sigma, T u, T lambda, T& E)
    {
        VECTOR<T, dim> log_sigma = sigma.log();
        T trace_log_sigma = log_sigma.Sum();
        E = u * log_sigma.square().Sum() + (T).5 * lambda * trace_log_sigma * trace_log_sigma;
    }

    static void Compute_DE_Div_DSigma(const VECTOR<T, dim>& sigma, T u, T lambda,
                                      VECTOR<T, dim>& dE_div_dsigma)
    {
        VECTOR<T, dim> logS = sigma.log();
        if constexpr (dim == 2) {
            T g = 2 * u + lambda;
            dE_div_dsigma[0] = (g * logS(0) + lambda * logS(1)) / sigma(0);
            dE_div_dsigma[1] = (g * logS(1) + lambda * logS(0)) / sigma(1);
        }
        else {
            T sum_log = logS(0) + logS(1) + logS(2);
            dE_div_dsigma[0] = (2 * u * logS(0) + lambda * sum_log) / sigma(0);
            dE_div_dsigma[1] = (2 * u * logS(1) + lambda * sum_log) / sigma(1);
            dE_div_dsigma[2] = (2 * u * logS(2) + lambda * sum_log) / sigma(2);
        }
    }

    static void Compute_D2E_Div_DSigma2(const VECTOR<T, dim>& sigma, T u, T lambda,
                                        MATRIX<T, dim>& d2E_div_dsigma2)
    {
        VECTOR<T, dim> logS = sigma.log();
        if constexpr (dim == 2) {
            T g = 2 * u + lambda;
            T prod = sigma(0) * sigma(1);
            d2E_div_dsigma2(0, 0) = (g * (1 - logS(0)) - lambda * logS(1)) / (sigma(0) * sigma(0));
            d2E_div_dsigma2(1, 1) = (g * (1 - logS(1)) - lambda * logS(0)) / (sigma(1) * sigma(1));
            d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda / prod;
        }
        else {
            T g = 2 * u + lambda;
            d2E_div_dsigma2(0, 0) = (g * (1 - logS(0)) - lambda * (logS(1) + logS(2))) / (sigma(0) * sigma(0));
            d2E_div_dsigma2(1, 1) = (g * (1 - logS(1)) - lambda * (logS(0) + logS(2))) / (sigma(1) * sigma(1));
            d2E_div_dsigma2(2, 2) = (g * (1 - logS(2)) - lambda * (logS(0) + logS(1))) / (sigma(2) * sigma(2));
            d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda / (sigma(0) * sigma(1));
            d2E_div_dsigma2(0, 2) = d2E_div_dsigma2(2, 0) = lambda / (sigma(0) * sigma(2));
            d2E_div_dsigma2(1, 2) = d2E_div_dsigma2(2, 1) = lambda / (sigma(1) * sigma(2));
        }
    }

    static void Compute_BLeftCoef(const VECTOR<T, dim>& sigma, T u, T lambda,
                                  VECTOR<T, dim*(dim - 1) / 2>& BLeftCoef)
    {
        VECTOR<T, dim> logS = sigma.log();
        T eps = (T)1e-8;
        if constexpr (dim == 2) {
            T prod = sigma(0) * sigma(1);
            T q = std::max(sigma(0) / sigma(1) - 1, -1 + eps);
            T h = (std::fabs(q) < eps) ? 1 : (std::log1p(q) / q);
            T t = h / sigma(1);
            T z = logS(1) - t * sigma(1);
            BLeftCoef[0] = -0.5 * (lambda * (logS(0) + logS(1)) + 2 * u * z) / prod;
        } else {
            T sum_log = logS(0) + logS(1) + logS(2);
            BLeftCoef[0] = -0.5 * (lambda * sum_log + 2 * u * MATH_TOOLS::diff_interlock_log_over_diff(sigma(0), sigma(1), logS(1), eps)) / (sigma(0) * sigma(1));
            BLeftCoef[1] = -0.5 * (lambda * sum_log + 2 * u * MATH_TOOLS::diff_interlock_log_over_diff(sigma(1), sigma(2), logS(2), eps)) / (sigma(1) * sigma(2));
            BLeftCoef[2] = -0.5 * (lambda * sum_log + 2 * u * MATH_TOOLS::diff_interlock_log_over_diff(sigma(2), sigma(0), logS(0), eps)) / (sigma(2) * sigma(0));
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
