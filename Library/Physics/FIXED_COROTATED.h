#pragma once

#include <meta/meta.hpp>

#include <Math/VECTOR.h>
#include <Math/SINGULAR_VALUE_DECOMPOSITION.h>
#include <FEM/DATA_TYPE.h>
#include <numeric>

namespace py = pybind11;
namespace JGSL {

template<class T>
using FIXED_COROTATED_2 = BASE_STORAGE<MATRIX<T, 2>, T, T, T>;

template <std::size_t OFFSET, class T>
struct FIELDS_WITH_OFFSET<OFFSET, FIXED_COROTATED_2<T>> {
    enum INDICES { F = OFFSET, VOL, LAMBDA, MU };
};

template<class T>
using FIXED_COROTATED_3 = BASE_STORAGE<MATRIX<T, 3>, T, T, T>;

template <std::size_t OFFSET, class T>
struct FIELDS_WITH_OFFSET<OFFSET, FIXED_COROTATED_3<T>> {
    enum INDICES { F = OFFSET, VOL, LAMBDA, MU };
};

template<class T, int dim>
using FIXED_COROTATED = meta::if_<
    meta::equal_to<meta::int_<dim>, meta::int_<2>>, 
    FIXED_COROTATED_2<T>, // F, VOL, LAMBDA, MU, _padding_
    FIXED_COROTATED_3<T>  // F, VOL, LAMBDA, MU
>;

template <class T, int dim>
struct FCR_HELPER {
    static void Insert(FIXED_COROTATED<T, dim>& fcr, int i, T vol, T lambda, T mu) {
        fcr.Insert(i, MATRIX<T, dim>(1), vol, lambda, mu);
    }
};

template <class T, int dim>
class FIXED_COROTATED_FUNCTOR {
public:
    using STORAGE = FIXED_COROTATED<T, dim>;
    using DIFFERENTIAL = BASE_STORAGE<Eigen::Matrix<T, dim*dim, dim*dim>>;
    static const bool useJ = false;
    static const bool projectable = true;

    static std::unique_ptr<STORAGE> Create() {
        return std::make_unique<STORAGE>();
    }

    static void Append(STORAGE& fcr, const VECTOR<int, 2>& handle, T vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = handle(0); i < handle(1); ++i)
            FCR_HELPER<T, dim>::Insert(fcr, i, vol, lambda, mu);
    }

    static void Append_FEM(STORAGE& fcr, const VECTOR<int, 4>& handle, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = handle(1); i < handle(3); ++i)
            FCR_HELPER<T, dim>::Insert(fcr, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void Append_All_FEM(STORAGE& fcr, const VECTOR<int, 4>& handle, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = 0; i < handle(3); ++i)
            FCR_HELPER<T, dim>::Insert(fcr, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void All_Append_FEM(STORAGE& fcr, const SCALAR_STORAGE<T>& vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = 0; i < vol.size; ++i)
            FCR_HELPER<T, dim>::Insert(fcr, i, std::get<0>(vol.Get_Unchecked_Const(i)), lambda, mu);
    }

    static void Compute_Kirchoff_Stress(STORAGE& fcr, MPM_STRESS<T, dim>& stress) {
        TIMER_FLAG("Compute Kirchoff Stress for Fixed Corotated");
        fcr.Join(stress).Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu, ks, __] = data;
            T J = F.determinant();
            MATRIX<T, dim> JFinvT = F.cofactor();
            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V);
            MATRIX<T, dim> R = U * V.transpose();
            MATRIX<T, dim> first_piola = (T)2 * mu * (F - R) + lambda * (J - 1) * JFinvT;
            ks += -vol * first_piola * F.transpose();
        });
    }

    static void Compute_Psi(STORAGE& fcr, T w, MESH_ELEM_ATTR<T, dim>& elemAttr, T& Psi) {
        TIMER_FLAG("computePsi");
        std::vector<T> psi(elemAttr.size);
        fcr.Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu] = data;

            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime

            Compute_E(sigma, mu, lambda, psi[i]);
            psi[i] *= w * vol;
        });

        Psi = std::accumulate(psi.begin(), psi.end(), Psi);
    }

    static void Compute_Psi(STORAGE& fcr, T w, T& Psi) {
        TIMER_FLAG("computePsi");
        std::vector<T> psi(fcr.size);
        fcr.Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu] = data;

            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime

            Compute_E(sigma, mu, lambda, psi[i]);
            psi[i] *= w * vol;
        });

        Psi = std::accumulate(psi.begin(), psi.end(), T(0));
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& fcr, T w, MESH_ELEM_ATTR<T, dim>& elemAttr) {
        TIMER_FLAG("computeStress");
        fcr.Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu] = data;
            T J = F.determinant();
            MATRIX<T, dim> JFinvT = F.cofactor();
            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime
            MATRIX<T, dim> R = U * V.transpose();
            elemAttr.template Update_Component<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(i, w * vol * (T)2 * mu * (F - R) + w * vol * lambda * (J - 1) * JFinvT);
        });
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& fcr, T w, BASE_STORAGE<MATRIX<T, dim>>& P) {
        TIMER_FLAG("computeStress");
        if(P.size != fcr.size) {
            P.Reserve(fcr.size);
            fcr.Each([&](const int& idx, auto data) {
                P.Insert(idx, MATRIX<T, dim>(0));
            });
        }
        fcr.Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu] = data;
            T J = F.determinant();
            MATRIX<T, dim> JFinvT = F.cofactor();
            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime
            MATRIX<T, dim> R = U * V.transpose();
            P.template Get_Component_Unchecked<0>(i) = w * vol * (T)2 * mu * (F - R) + w * vol * lambda * (J - 1) * JFinvT;
        });
    }

    static void Compute_First_PiolaKirchoff_Stress_Derivative(STORAGE& fcr, T w, bool projectSPD, DIFFERENTIAL& dP_div_dF) {
        TIMER_FLAG("computeStressDerivative");
        if(dP_div_dF.size != fcr.size) {
            dP_div_dF.Reserve(fcr.size);
            fcr.Each([&](const int& idx, auto data)
            {
                dP_div_dF.Insert(idx, Eigen::Matrix<T, dim*dim, dim*dim>::Zero());
            });
        }
        fcr.Join(dP_div_dF).Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu, dP_div_dFI] = data;

            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime
            
            Compute_DP_Div_DF(U, sigma, V, mu, lambda, w * vol, projectSPD, dP_div_dFI);
        });
    }

    static void Compute_E(const VECTOR<T, dim>& sigma, T u, T lambda, T& E)
    {
        const T sigmam12Sum = (sigma - sigma.Ones_Vector()).length2();
        const T sigmaProdm1 = sigma.prod() - 1;

        E = u * sigmam12Sum + lambda / 2 * sigmaProdm1 * sigmaProdm1;
    }

    static void Compute_DE_Div_DSigma(const VECTOR<T, dim>& singularValues, T u, T lambda,
        VECTOR<T, dim>& dE_div_dsigma) 
    {
        const T sigmaProdm1lambda = lambda * (singularValues.prod() - 1);
        VECTOR<T, dim> sigmaProd_noI;
        if constexpr (dim == 2) {
            sigmaProd_noI[0] = singularValues[1];
            sigmaProd_noI[1] = singularValues[0];
        }
        else {
            sigmaProd_noI[0] = singularValues[1] * singularValues[2];
            sigmaProd_noI[1] = singularValues[2] * singularValues[0];
            sigmaProd_noI[2] = singularValues[0] * singularValues[1];
        }

        T _2u = u * 2;
        dE_div_dsigma[0] = (_2u * (singularValues[0] - 1) + sigmaProd_noI[0] * sigmaProdm1lambda);
        dE_div_dsigma[1] = (_2u * (singularValues[1] - 1) + sigmaProd_noI[1] * sigmaProdm1lambda);
        if constexpr (dim == 3) {
            dE_div_dsigma[2] = (_2u * (singularValues[2] - 1) + sigmaProd_noI[2] * sigmaProdm1lambda);
        }
    }

    static void Compute_D2E_Div_DSigma2(const VECTOR<T, dim>& singularValues, T u, T lambda,
        MATRIX<T, dim>& d2E_div_dsigma2)
    {
        const T sigmaProd = singularValues.prod();
        VECTOR<T, dim> sigmaProd_noI;
        if constexpr (dim == 2) {
            sigmaProd_noI[0] = singularValues[1];
            sigmaProd_noI[1] = singularValues[0];
        }
        else {
            sigmaProd_noI[0] = singularValues[1] * singularValues[2];
            sigmaProd_noI[1] = singularValues[2] * singularValues[0];
            sigmaProd_noI[2] = singularValues[0] * singularValues[1];
        }

        double _2u = u * 2;
        d2E_div_dsigma2(0, 0) = _2u + lambda * sigmaProd_noI[0] * sigmaProd_noI[0];
        d2E_div_dsigma2(1, 1) = _2u + lambda * sigmaProd_noI[1] * sigmaProd_noI[1];
        if constexpr (dim == 3) {
            d2E_div_dsigma2(2, 2) = _2u + lambda * sigmaProd_noI[2] * sigmaProd_noI[2];
        }

        if constexpr (dim == 2) {
            d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda * ((sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[1]);
        }
        else {
            d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda * (singularValues[2] * (sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[1]);
            d2E_div_dsigma2(0, 2) = d2E_div_dsigma2(2, 0) = lambda * (singularValues[1] * (sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[2]);
            d2E_div_dsigma2(2, 1) = d2E_div_dsigma2(1, 2) = lambda * (singularValues[0] * (sigmaProd - 1) + sigmaProd_noI[2] * sigmaProd_noI[1]);
        }
    }

    static void Compute_BLeftCoef(const VECTOR<T, dim>& singularValues, T u, T lambda,
        VECTOR<T, dim*(dim - 1) / 2>& BLeftCoef)
    {
        const T sigmaProd = singularValues.prod();
        const T halfLambda = lambda / 2;
        if constexpr (dim == 2) {
            BLeftCoef[0] = u - halfLambda * (sigmaProd - 1);
        }
        else {
            BLeftCoef[0] = u - halfLambda * singularValues[2] * (sigmaProd - 1);
            BLeftCoef[1] = u - halfLambda * singularValues[0] * (sigmaProd - 1);
            BLeftCoef[2] = u - halfLambda * singularValues[1] * (sigmaProd - 1);
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
        T J = F.determinant();
        MATRIX<T, dim> JFinvT = F.cofactor();
        MATRIX<T, dim> R = U * V.transpose();
        MATRIX<T, dim> P = (T)2 * mu * (F - R) + lambda * (J - 1) * JFinvT;
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
