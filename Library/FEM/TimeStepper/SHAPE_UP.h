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

template <class T, int dim>
double X2F(double p, double q, double i, double j, const MATRIX<T, dim>& A)
{
    return (p > 0 ? (i == q ? A(p - 1, j) : 0) : -(i == q ? A(0, j) + A(1, j) : 0));
}

template <class T, int dim>
double F2X(double i, double j, double p, double q, const MATRIX<T, dim>& A)
{
    return X2F(p, q, i, j, A);
}

template <class T, int dim>
int Advance_One_Step_SU(MESH_ELEM<dim>& Elem,
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
    TIMER_ANALYZE("shapeUp");

    RESTART:

    T constraint_weight = PARAMETER::Get("Constraint_Weight", (T)1);
    SHAPE_MATCHING_ENERGY<T,dim> sme;
    sme.Precompute(Elem, X, X0, elasticityAttr);

    std::vector<bool> isFixedVert(X.size, false);
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        isFixedVert[dbcI(0)] = true;
        REP(d, dim) std::get<0>(X.Get_Unchecked(dbcI(0)))(d) = dbcI(d + 1);
    });

    // compute matrix
    CSR_MATRIX<T> sysMtr;
    std::vector<Eigen::Triplet<T>> triplets;
    std::vector<Eigen::Triplet<T>> rhs_corrector;
    Elem.Join(elemAttr).Each([&](int id, auto data) {
        auto &[elemVInd, A, P] = data;
        REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) REP(pp, dim + 1) REP(qq, dim) {
        int vI = elemVInd(p) * dim + q;
        int vJ = elemVInd(pp) * dim + qq;
        double value = X2F(p, q, i, j, A) * F2X(i, j, pp, qq, A);
        if (!isFixedVert[vI / dim] && !isFixedVert[vJ / dim]) triplets.emplace_back(vI, vJ, value);
        if (!isFixedVert[vI / dim] && isFixedVert[vJ / dim]) rhs_corrector.emplace_back(vI, vJ, value);
    }
    });
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        REP(d, dim) triplets.emplace_back(dbcI(0) * dim + d, dbcI(0) * dim + d, 1);
    });
    for (int i = 0; i < sme.index_groups.size(); ++i) {
        std::vector<int>& indices = sme.index_groups[i];
        T n = (T)indices.size();
        for (auto i : indices) REP(d, dim)
        for (int j : indices) REP(e, dim) if (d == e) {
            T value = 0;
            if (i == j) value = ((n - 1) / n) * ((n - 1) / n) + (n - 1) * (-1 / n) * (-1 / n);
            else value = 2 * ((n - 1) / n) * (-1 / n) + (n - 2) * (-1 / n) * (-1 / n);
            triplets.emplace_back(i * dim + d, j * dim + e, constraint_weight * value);
        }
    }
    sysMtr.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
    Solver_Direct_Helper<T> helper(sysMtr);

    T res1 = -1, res2 = -1;
    for (int iter = 0; ; ++iter) {
        // compute rhs
        T res = 0;
        std::vector<T> rhs(X.size * dim), sol(X.size * dim);
        Elem.Join(elemAttr).Each([&](int id, auto data) {
            auto &[elemVInd, A, P] = data;
            MATRIX<T, dim> F;
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
            int vI = elemVInd(p);
            F(i, j) += X2F(p, q, i, j, A) * std::get<0>(X.Get_Unchecked(vI))(q);
        }
            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V);
            MATRIX<T, dim> R = U * V.transpose();
            REP(p, dim + 1) REP(q, dim) REP(i, dim) REP(j, dim) {
            int vI = elemVInd(p);
            rhs[vI * dim + q] += F2X(i, j, p, q, A) * R(i, j);
        }
            res += (F - R).length2();
        });
        for (auto tri : rhs_corrector)
            rhs[tri.row()] -= std::get<0>(X.Get_Unchecked(tri.col() / dim))(tri.col() % dim) * tri.value();
        DBC.Each([&](int id, auto data) {
            auto &[dbcI] = data;
            REP(d, dim) rhs[dbcI(0) * dim + d] = dbcI(d + 1);
        });
        for (int i = 0; i < sme.index_groups.size(); ++i) {
            std::vector<int>& indices = sme.index_groups[i];
            std::vector<VECTOR<T, dim>>& X_group = sme.X_groups[i];
            std::vector<VECTOR<T, dim>> x_group;
            for (auto ig : indices) x_group.push_back(std::get<0>(X.Get_Unchecked(ig)));
            SHAPE_MATCHING_DERIVATION<T, dim> dr(x_group, X_group);
            REP(i, dr.n) {
                VECTOR<T, dim> Px = dr.R * (X_group[i] - dr.X_com);
                REP(d, dim) rhs[indices[i] * dim + d] += Px(d) * constraint_weight;
                res += (Px - x_group[i]).length2() * constraint_weight;
            }
        }
        helper.Solve(rhs, sol);
        X.Par_Each([&](int id, auto data) {
            auto &[x] = data;
            REP(d, dim) x(d) = sol[id * dim + d];
        });

        T a = 0;
        T b = 0;
        for (int i = 0; i < sme.index_groups.size(); ++i) {
            std::vector<int>& indices = sme.index_groups[i];
            std::vector<VECTOR<T, dim>>& X_group = sme.X_groups[i];
            std::vector<VECTOR<T, dim>> x_group;
            for (auto ig : indices) x_group.push_back(std::get<0>(X.Get_Unchecked(ig)));
            SHAPE_MATCHING_DERIVATION<T, dim> dr(x_group, X_group);
            REP(i, dr.n) a += 0.5 * (x_group[i] - dr.x_com - dr.R * (X_group[i] - dr.X_com)).length2();
            b += X_group.size();
        }
        a /= b;
        printf("!!!!%.20f %.20f\n", a, constraint_weight);
        if (a < PARAMETER::Get("Shape_Matching_Convergence_Criteria", 1e-6)) {
            sme.Print(Elem, X, output_folder, current_frame);
            JGSL_FILE("iteration", iter + 1);
            return iter + 1;
        }

        if (res2 >= 0 && (res1 - res) < res1 * 1e-6) {
            sme.Print(Elem, X, output_folder, current_frame);
            JGSL_FILE("iteration", iter + 1);
            return iter + 1;
            constraint_weight *= 10;
            PARAMETER::Set("Constraint_Weight", constraint_weight);
            res1 = res2 = -1;
            goto RESTART;
        } else {
            res2 = res1;
            res1 = res;
        }
    }
}

}
