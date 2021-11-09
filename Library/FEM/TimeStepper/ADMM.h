#pragma once

#include <Physics/FIXED_COROTATED.h>
#include <FEM/DEFORMATION_GRADIENT.h>
#include <FEM/ELEM_TO_NODE.h>
#include <FEM/IPC.h>
#include <FEM/FRICTION.h>
#include <Math/CSR_MATRIX.h>
#include <Math/DIRECT_SOLVER.h>
#include <FEM/Energy/ENERGY.h>
#include <Utils/PARAMETER.h>

namespace py = pybind11;
namespace JGSL {

std::unique_ptr<Solver_Direct_Helper<double>> helper = nullptr;
std::vector<Eigen::Triplet<double>> rhs_corrector;

template <class T, int dim>
int Advance_One_Step_ADMM(MESH_ELEM<dim>& Elem,
                        VECTOR_STORAGE<T, dim + 1>& DBC,
                        const VECTOR<T, dim>& gravity, T h,
                        T NewtonTol, bool withCollision,
                        T dHat2, const VECTOR<T, 2>& kappaVec, //TODO: dHat as input and relative to bbox, adapt kappa
                        T mu, T epsv2,
                        std::string output_folder, int current_frame,
                        MESH_NODE<T, dim>& X,
                        MESH_NODE<T, dim>& X0,
                        MESH_NODE_ATTR<T, dim>& nodeAttr,
                        MESH_ELEM_ATTR<T, dim>& elemAttr,
                        FIXED_COROTATED<T, dim>& elasticityAttr)
{
    TIMER_ANALYZE("ADMM");

    // record Xn and compute predictive pos Xtilde
    MESH_NODE<T, dim> Xn, Xtilde;
    Append_Attribute(X, Xn);
    Append_Attribute(X, Xtilde);
    Xtilde.Join(nodeAttr).Par_Each([&](int id, auto data) {
        auto &[x, x0, v, g, m] = data;
        x += h * v + h * h * gravity;
    });

    T _area = PARAMETER::Get("Fixed_Corotated_area", (T)1.0);
    T _lambda = PARAMETER::Get("Fixed_Corotated_lambda", (T)1.0);
    T _mu = PARAMETER::Get("Fixed_Corotated_mu", (T)1.0);
    printf("=========================> %.10f\n", std::sqrt(_area * (_lambda + _mu * 2 / 3)));
    std::vector<MATRIX<T, dim>> W(Elem.size, MATRIX<T, dim>::Ones_Matrix() * std::sqrt(_area * (_lambda + _mu * 2 / 3)));

    std::vector<bool> isFixedVert(X.size, false);
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        isFixedVert[dbcI(0)] = true;
        REP(d, dim) std::get<0>(X.Get_Unchecked(dbcI(0)))(d) = dbcI(d + 1);
    });

    static bool first_run = true;
    if (first_run) {
        // precompute global matrix
        first_run = true;
        rhs_corrector.clear();

        Elem.Join(elemAttr).Each([&](int id, auto data) {
            auto &[elemVInd, A, P] = data;
            MATRIX<T, dim> F;
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
                int vI = elemVInd(p);
                F(i, j) += X2F(p, q, i, j, A) * std::get<0>(X.Get_Unchecked(vI))(q);
            }
            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V); //TODO: not redo everytime
            Eigen::Matrix<T, dim*dim, dim*dim> dP_div_dFI;
            FIXED_COROTATED_FUNCTOR<T, dim>::Compute_DP_Div_DF(U, sigma, V, _mu, _lambda, 1, false, dP_div_dFI);
            W[id] = MATRIX<T, dim>::Ones_Matrix() * std::sqrt(dP_div_dFI.norm() * h * h * std::get<1>(elasticityAttr.Get_Unchecked(id)));
            W[id] = MATRIX<T, dim>::Ones_Matrix() * std::sqrt(_lambda + _mu * 2 / 3) * std::get<1>(elasticityAttr.Get_Unchecked(id));
        });

        CSR_MATRIX<T> sysMtr;
        std::vector<Eigen::Triplet<T>> triplets;
        Elem.Join(elemAttr).Each([&](int id, auto data) {
            auto &[elemVInd, A, P] = data;
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) REP(pp, dim + 1) REP(qq, dim) {
                int vI = elemVInd(p) * dim + q;
                int vJ = elemVInd(pp) * dim + qq;
                double value = X2F(p, q, i, j, A) * F2X(i, j, pp, qq, A) * W[id](i, j) * W[id](i, j);
                if (!isFixedVert[vI / dim] && !isFixedVert[vJ / dim]) triplets.emplace_back(vI, vJ, value);
                if (!isFixedVert[vI / dim] && isFixedVert[vJ / dim]) rhs_corrector.emplace_back(vI, vJ, value);
            }
        });
        nodeAttr.Each([&](int id, auto data) {
            auto &[x0, v, g, m] = data;
            if (isFixedVert[id]) return;
            triplets.emplace_back(id * dim, id * dim, m);
            triplets.emplace_back(id * dim + 1, id * dim + 1, m);
            if constexpr (dim == 3) {
                triplets.emplace_back(id * dim + 2, id * dim + 2, m);
            }
        });
        DBC.Each([&](int id, auto data) {
            auto &[dbcI] = data;
            REP(d, dim) triplets.emplace_back(dbcI(0) * dim + d, dbcI(0) * dim + d, 1);
        });
        sysMtr.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
        helper = std::make_unique<Solver_Direct_Helper<T>>(sysMtr);
    }

    //initial guess
    std::vector<MATRIX<T, dim>> z(Elem.size, MATRIX<T, dim>());
    std::vector<MATRIX<T, dim>> u(Elem.size, MATRIX<T, dim>());
    Elem.Join(elemAttr).Each([&](int id, auto data) {
        auto &[elemVInd, A, P] = data;
        REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
            int vI = elemVInd(p);
            z[id](i, j) += X2F(p, q, i, j, A) * std::get<0>(X.Get_Unchecked(vI))(q);
        }
    });

    for (int iter = 0; iter < 20; ++iter) {
        // global step
        std::vector<T> rhs(X.size * dim), sol(X.size * dim);
        Elem.Join(elemAttr).Each([&](int id, auto data) {
            auto &[elemVInd, A, P] = data;
            MATRIX<T, dim> F = z[id] - u[id];
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
                int vI = elemVInd(p);
                rhs[vI * dim + q] += F2X(i, j, p, q, A) * F(i, j) * W[id](i, j) * W[id](i, j);
            }
        });
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            for (int d = 0; d < dim; ++d)
                rhs[id * dim + d] += m * (std::get<0>(Xtilde.Get_Unchecked(id)))(d);
        });
        for (auto tri : rhs_corrector)
            rhs[tri.row()] -= std::get<0>(X.Get_Unchecked(tri.col() / dim))(tri.col() % dim) * tri.value();
        DBC.Each([&](int id, auto data) {
            auto &[dbcI] = data;
            REP(d, dim) rhs[dbcI(0) * dim + d] = dbcI(d + 1);
        });
        helper->Solve(rhs, sol);
        X.Par_Each([&](int id, auto data) {
            auto &[x] = data;
            for (int d = 0; d < dim; ++d) x(d) = sol[id * dim + d];
        });
        // local step
        std::vector<MATRIX<T, dim>> old_z = z;
        Elem.Join(elemAttr).Par_Each([&](int id, auto data) {
            auto &[elemVInd, A, P] = data;
            MATRIX<T, dim> Dx_plus_u_mtr;
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
                int vI = elemVInd(p);
                Dx_plus_u_mtr(i, j) += X2F(p, q, i, j, A) * std::get<0>(X.Get_Unchecked(vI))(q);
            }
            Dx_plus_u_mtr += u[id];
            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(Dx_plus_u_mtr, U, sigma, V);
            VECTOR<T, dim> sigma_Dx_plus_u = sigma;
            VECTOR<T, dim> g;
            auto computeEnergyVal_zUpdate_SV = [&](VECTOR<T, dim>& sigma, T& E) {
                FIXED_COROTATED_FUNCTOR<T, dim>::Compute_E(sigma, _mu, _lambda, E);
                E *= h * h * std::get<1>(elasticityAttr.Get_Unchecked(id));
                E += (sigma_Dx_plus_u - sigma).length2() * W[id](0, 0) * W[id](0, 0) / 2.0;
            };
            auto computeGradient_zUpdate_SV = [&](VECTOR<T, dim>& sigma, VECTOR<T, dim>& g) {
                FIXED_COROTATED_FUNCTOR<T, dim>::Compute_DE_Div_DSigma(sigma, _mu, _lambda, g);
                g *= h * h * std::get<1>(elasticityAttr.Get_Unchecked(id));
                g -= (sigma_Dx_plus_u - sigma) * W[id](0, 0) * W[id](0, 0);
            };
            auto computeHessianProxy_zUpdate_SV = [&](VECTOR<T, dim>& sigma, MATRIX<T, dim>& P) {
                FIXED_COROTATED_FUNCTOR<T, dim>::Compute_D2E_Div_DSigma2(sigma, _mu, _lambda, P);
                P.makePD();
                P *= h * h * std::get<1>(elasticityAttr.Get_Unchecked(id));
                REP(d, dim) P(d, d)+= W[id](0, 0) * W[id](0, 0);
            };
            for (int iter = 0; iter < 100; ++iter) {
                computeGradient_zUpdate_SV(sigma, g);
                MATRIX<T, dim> P;
                computeHessianProxy_zUpdate_SV(sigma, P);
                VECTOR<T, dim> p(P.to_eigen().ldlt().solve(-g.to_eigen()));
                double alpha = 1.0;
                const double c1m = 0.0;
                VECTOR<T, dim> sigma0 = sigma;
                double E0;
                computeEnergyVal_zUpdate_SV(sigma0, E0);
                sigma = sigma0 + alpha * p;
                double E;
                computeEnergyVal_zUpdate_SV(sigma, E);
                if (p.abs().max() < 1e-6)
                    break;
                while (E > E0 + alpha * c1m) {
                    alpha /= 2.0;
                    sigma = sigma0 + alpha * p;
                    computeEnergyVal_zUpdate_SV(sigma, E);
                }
                if (iter == 99) {
                    puts("Local reaches max iterations");
                    getchar();
                }
            }
            z[id] = U * MATRIX<T, dim>(sigma) * V.transpose();
        });
        // residual
        double residual1 = 0, residual2 = 0;
        std::vector<T> tmp(X.size * dim, 0);
        Elem.Join(elemAttr).Each([&](int id, auto data) {
            auto &[elemVInd, A, P] = data;
            MATRIX<T, dim> F;
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
                int vI = elemVInd(p);
                F(i, j) += X2F(p, q, i, j, A) * std::get<0>(X.Get_Unchecked(vI))(q);
            }
            residual1 += (F - z[id]).length2() * W[id](0, 0) * W[id](0, 0);
            MATRIX<T, dim> delta = z[id] - old_z[id];
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
                int vI = elemVInd(p);
                tmp[vI * dim + q] += F2X(i, j, p, q, A) * delta(i, j) * W[id](0, 0) * W[id](0, 0);
            }
        });
        for (int i = 0; i < tmp.size(); ++i)
            residual2 += tmp[i] * tmp[i];
        printf("Iter : %d, Prime Residual : %.20f, Dual residual : %.20f, Tolerance : \n", iter, residual1, residual2);
        // duel step
        Elem.Join(elemAttr).Par_Each([&](int id, auto data) {
            auto &[elemVInd, A, P] = data;
            MATRIX<T, dim> F;
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
                int vI = elemVInd(p);
                F(i, j) += X2F(p, q, i, j, A) * std::get<0>(X.Get_Unchecked(vI))(q);
            }
            u[id] += F - z[id];
        });
    }
    X.Join(nodeAttr).Par_Each([&](int id, auto data) {
        auto &[x, x0, v, g, m] = data;
        v = (x - std::get<0>(Xn.Get_Unchecked(id))) / h;
    });
    return 20;
}

}
