#pragma once
#include <Utils/PARAMETER.h>
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

#define REP(i, n) for (int i = 0; i < n; ++i)

template <class T, int dim>
class ABSTRACT_ENERGY;

template <class T, int dim>
class SHAPE_MATCHING_DERIVATION {
public:
    int n;
    VECTOR<T, dim> x_com, X_com;
    MATRIX<T, dim> A, R, S, K;
    std::vector<T> _dRdA;
    std::vector<T> _dAdx;
    std::vector<T> _dRdx;
    SHAPE_MATCHING_DERIVATION(std::vector<VECTOR<T, dim>>& x, std::vector<VECTOR<T, dim>>& X) {
        n = x.size();
        x_com = X_com = VECTOR<T, dim>(0);
        for (int i = 0; i < n; ++i) x_com += x[i] / (T)n;
        for (int i = 0; i < n; ++i) X_com += X[i] / (T)n;
        A = R = S = K = MATRIX<T, dim>(0);
        for (int i = 0; i < n; ++i) A += outer_product(x[i] - x_com, X[i] - X_com) / (T)n;
        polarDecomposition(A, R, S);
        K(0, 0) = 0; K(0, 1) = -1;
        K(1, 0) = 1; K(1, 1) = 0;
        // compute _dRdA
        _dRdA.assign(16, 0);
        REP(i, dim) REP(j, dim) REP(c, dim) REP(b, dim) {
            int idx = (i << 3) + (j << 2) + (c << 1) + b;
            REP(a, dim) REP(w, dim) {
                T den = S(0, 0) + S(1, 1);
                _dRdA[idx] += K(a, b) * R(c, a) / den * R(i, w) * K(w, j);
            }
        }
        // compute _dAdx
        _dAdx.assign(8 * n, 0);
        REP(c, dim) REP(b, dim) REP(k, n) REP(l, dim) {
            int idx = ((c << 2) + (b << 1) + l) * n + k;
            if (c == l) {
                _dAdx[idx] += (X[k] - X_com)(b) / (T)n;
                REP(i, n) _dAdx[idx] -= (X[i] - X_com)(b) / (T)n / (T)n;
            }
        }
        // compute _dRdx
        _dRdx.assign(8 * n, 0);
        REP(i, dim) REP(j, dim) REP(k, n) REP(l, dim) {
            int idx = ((i << 2) + (j << 1) + l) * n + k;
            REP(c, dim) REP(b, dim) {
                int idx1 = (i << 3) + (j << 2) + (c << 1) + b;
                int idx2 = ((c << 2) + (b << 1) + l) * n + k;
                _dRdx[idx] += _dRdA[idx1] * _dAdx[idx2];
            }
        }
    }

    T dRdA(int i, int j, int c, int b) { return _dRdA[(i << 3) + (j << 2) + (c << 1) + b]; }
    T dAdx(int c, int b, int k, int l) { return _dAdx[((c << 2) + (b << 1) + l) * n + k]; }
    T dRdx(int i, int j, int k, int l) { return _dRdx[((i << 2) + (j << 1) + l) * n + k]; }
    T d2Rdx2(int i, int j, int k, int l, int u, int v) {
        T ret = 0;
        REP(c, dim) REP(b, dim) {
            REP(a, dim) REP(w, dim) {
                T den = S(0, 0) + S(1, 1);
                T tmp = 0;
                tmp += dRdx(c, a, u, v) * R(i, w) / den;
                tmp += R(c, a) * dRdx(i, w, u, v) / den;
                REP(r, dim) {
                    T dSdx_qnuv = 0;
                    dSdx_qnuv += dRdx(r, 0, u, v) * A(r, 0) + R(r, 0) * dAdx(r, 0, u, v);
                    dSdx_qnuv += dRdx(r, 1, u, v) * A(r, 1) + R(r, 1) * dAdx(r, 1, u, v);
                    tmp += R(c, a) * R(i, w) * (-(T)1 / den / den * dSdx_qnuv);
                }
                ret += K(a, b) * K(w, j) * tmp * dAdx(c, b, k, l);
            }
        }
        return ret;
    }
};

double old_mu = -1;
std::vector<std::vector<VECTOR<double, 2>>> old_lambda2d;

template <class T, int dim>
class SHAPE_MATCHING_ENERGY : public ABSTRACT_ENERGY<T, dim> {
public:
    T mu = 1;
    std::vector<std::vector<VECTOR<T, dim>>> lambda_groups;
    bool projectPD;
    std::vector<std::vector<int>> index_groups;
    std::vector<std::vector<VECTOR<T, dim>>> X_groups;
    void Precompute(MESH_ELEM<dim>& Elem, MESH_NODE<T, dim>& nodes, MESH_NODE<T, dim>& X0, FIXED_COROTATED<T, dim>& fcr, bool setProjectPD = true) {
        // disjoint square
        printf("Enable shape matching : %s\n", PARAMETER::Get("Shape_Matching", std::string("")).c_str());
        if (PARAMETER::Get("Shape_Matching", std::string("")) == "disjoint_square") {
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.44, 0.4), VECTOR<T, dim>(0.56, 0.6)));
        }
        else if (PARAMETER::Get("Shape_Matching", std::string("")) == "disjoint_sharkey") {
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.2, 0.6), VECTOR<T, dim>(0.3, 0.7)));
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.42, 0.3), VECTOR<T, dim>(0.58, 0.4)));
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.7, 0.5), VECTOR<T, dim>(0.85, 0.7)));
        }
        else if (PARAMETER::Get("Shape_Matching", std::string("")) == "three") {
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.25, 0.000), VECTOR<T, dim>(0.2566, 0.666)));
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.45, 0.333), VECTOR<T, dim>(0.4566, 1.000)));
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.65, 0.000), VECTOR<T, dim>(0.6566, 1.000)));
        }
        else if (PARAMETER::Get("Shape_Matching", std::string("")) == "one_point") {
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.25, 0.33), VECTOR<T, dim>(0.35, 0.75)));
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.33, 0.15), VECTOR<T, dim>(0.65, 0.4)));
        }
        else if (PARAMETER::Get("Shape_Matching", std::string("")) == "two_points") {
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.325, 0.30), VECTOR<T, dim>(0.35, 0.75)));
            index_groups.push_back(select(X0, VECTOR<T, dim>(0.3, 0.34), VECTOR<T, dim>(0.65, 0.4)));
        } else {
            puts("Invalid Shape_Matching value");
            exit(0);
        }
        // do it for each group
        for (int idx = 0; idx < index_groups.size(); ++idx) {
            auto& indices = index_groups[idx];
            printf("Size of %d rigid region : %d\n", idx + 1, (int)indices.size());
            X_groups.emplace_back();
            for (auto i : indices) X_groups.back().push_back(std::get<0>(X0.Get_Unchecked_Const(i)));
            lambda_groups.emplace_back();
            for (auto i : indices) lambda_groups.back().push_back(VECTOR<T, dim>(0));
        }
        if constexpr (dim == 2) if (old_mu >= 0) {
            puts("Old Value Used.");
            mu = old_mu;
            lambda_groups = old_lambda2d;
        }
        projectPD = setProjectPD;
    }
    bool Keep_Going(MESH_NODE<T, dim>& nodes) {
        T a = 0;
        T b = 0;
        for (int i = 0; i < index_groups.size(); ++i) {
            std::vector<VECTOR<T, dim>>& X = X_groups[i];
            std::vector<VECTOR<T, dim>> x;
            for (auto ig : index_groups[i]) x.push_back(std::get<0>(nodes.Get_Unchecked_Const(ig)));
            SHAPE_MATCHING_DERIVATION<T, dim> dr(x, X);
            REP(i, dr.n) a += 0.5 * (x[i] - dr.x_com - dr.R * (X[i] - dr.X_com)).length2();
            b += X.size();
        }
        printf("=======================================================+ %.20f %.20f\n", a, b);
        a /= b;
        printf("=======================================================> %.20f %.4f %d %d\n", a, mu, a < 1e-8, a > 1e-6);
        if (a < PARAMETER::Get("Shape_Matching_Convergence_Criteria", 1e-6)) return false;
        if (a > PARAMETER::Get("Shape_Matching_Convergence_Criteria", 1e-6) * 10 && mu < 100) {
            mu = mu * 2;
        } else {
            for (int idx = 0; idx < index_groups.size(); ++idx) {
                std::vector<VECTOR<T, dim>>& X = X_groups[idx];
                std::vector<VECTOR<T, dim>> x;
                std::vector<VECTOR<T, dim>>& lambda = lambda_groups[idx];
                for (auto i : index_groups[idx]) x.push_back(std::get<0>(nodes.Get_Unchecked_Const(i)));
                SHAPE_MATCHING_DERIVATION<T, dim> dr(x, X);
                REP(i, dr.n) lambda[i] += mu * (x[i] - dr.x_com - dr.R * (X[i] - dr.X_com));
            }
        }
        return true;
    }
    void Print(MESH_ELEM<dim>& Elem, MESH_NODE<T, dim>& nodes, std::string output_folder, int current_frame) {
        if constexpr (dim == 2) {
            old_mu = mu;
            old_lambda2d = lambda_groups;
        }
        T a = 0;
        T b = 0;
        for (int i = 0; i < index_groups.size(); ++i) {
            std::string output_filename = output_folder + "rigid_" + std::to_string(i) + "_" + std::to_string(current_frame) + ".obj";
            if constexpr (dim == 2)
                Write_TriMesh_Region_Obj(nodes, Elem, output_filename, index_groups[i]);


            std::vector<VECTOR<T, dim>>& X = X_groups[i];
            std::vector<VECTOR<T, dim>> x;
            for (auto ig : index_groups[i]) x.push_back(std::get<0>(nodes.Get_Unchecked_Const(ig)));
            SHAPE_MATCHING_DERIVATION<T, dim> dr(x, X);
            REP(i, dr.n) a += 0.5 * (x[i] - dr.x_com - dr.R * (X[i] - dr.X_com)).length2();
            b += X.size();
        }
        a /= b;
        JGSL_ANALYZE("Shape Matching Satisfactory", a);
    }

    void Compute_IncPotential(
        MESH_ELEM<dim>& Elem,
        const VECTOR<T, dim>& gravity,
        T h, MESH_NODE<T, dim>& nodes,
        MESH_NODE<T, dim>& Xtilde,
        MESH_NODE_ATTR<T, dim>& nodeAttr,
        MESH_ELEM_ATTR<T, dim>& elemAttr,
        FIXED_COROTATED<T, dim>& elasticityAttr,
        std::vector<VECTOR<int, dim + 1>>& constraintSet,
        T dHat2, T kappa[],
        double& value
    ) {
        for (int idx = 0; idx < index_groups.size(); ++idx) {
            std::vector<VECTOR<T, dim>>& X = X_groups[idx];
            std::vector<VECTOR<T, dim>> x;
            std::vector<VECTOR<T, dim>>& lambda = lambda_groups[idx];
            for (auto i : index_groups[idx]) x.push_back(std::get<0>(nodes.Get_Unchecked_Const(i)));
            SHAPE_MATCHING_DERIVATION<T, dim> dr(x, X);
            REP(i, dr.n) value += 0.5 * mu * (x[i] - dr.x_com - dr.R * (X[i] - dr.X_com)).length2();
            REP(i, dr.n) value += lambda[i].dot(x[i] - dr.x_com - dr.R * (X[i] - dr.X_com));
        }
    }
    void Compute_IncPotential_Gradient(
        MESH_ELEM<dim>& Elem,
        const VECTOR<T, dim>& gravity,
        T h, MESH_NODE<T, dim>& nodes,
        MESH_NODE<T, dim>& Xtilde,
        MESH_NODE_ATTR<T, dim>& nodeAttr,
        MESH_ELEM_ATTR<T, dim>& elemAttr,
        FIXED_COROTATED<T, dim>& elasticityAttr,
        std::vector<VECTOR<int, dim + 1>>& constraintSet,
        T dHat2, T kappa[]
    ) {
        for (int idx = 0; idx < index_groups.size(); ++idx) {
            std::vector<VECTOR<T, dim>>& X = X_groups[idx];
            std::vector<VECTOR<T, dim>> x;
            std::vector<VECTOR<T, dim>> lambda = lambda_groups[idx];
            for (auto i : index_groups[idx]) x.push_back(std::get<0>(nodes.Get_Unchecked_Const(i)));
            SHAPE_MATCHING_DERIVATION<T, dim> dr(x, X);
            REP(i, dr.n) {
                VECTOR<T, dim> g;
                REP(j, dr.n) {
                    VECTOR<T, dim> v = x[j] - dr.x_com - dr.R * (X[j] - dr.X_com);
                    MATRIX<T, dim> m = ((i == j ? (T)1 : (T)0) - (T)1 / dr.n) * MATRIX<T, dim>(1);
                    REP(a, dim) REP(b, dim) REP(c, dim)
                       m(a, b) -= dr.dRdx(a, c, i, b) * (X[j] - dr.X_com)(c);
                    g += mu * m.transpose() * v;
                    g += m.transpose() * lambda[j];
                }
                std::get<2>(nodeAttr.Get_Unchecked(index_groups[idx][i])) += g;
            }
        }
    }
    void Compute_IncPotential_Hessian(
        MESH_ELEM<dim>& Elem,
        T h, MESH_NODE<T, dim>& nodes,
        MESH_NODE_ATTR<T, dim>& nodeAttr,
        MESH_ELEM_ATTR<T, dim>& elemAttr,
        FIXED_COROTATED<T, dim>& elasticityAttr,
        std::vector<VECTOR<int, dim + 1>>& constraintSet,
        T dHat2, T kappa[],
        std::vector<Eigen::Triplet<T>>& triplets
    ) {
        if (!PARAMETER::Get("CD.full_hessian", false)) {
            std::vector<std::vector<std::pair<int, int>>> sparse_pairs(index_groups.size());
            std::vector<int> idx2group(nodes.size, -1);
            std::vector<int> idx2idx(nodes.size, -1);
            for (int i = 0; i < index_groups.size(); ++i)
                for (int j = 0; j < index_groups[i].size(); ++j) {
                    idx2group[index_groups[i][j]] = i;
                    idx2idx[index_groups[i][j]] = j;
                    sparse_pairs[i].emplace_back(j, j);
                }
            std::map<std::pair<int, int>, bool> mp;
            Elem.Each([&](int id, auto data) {
                auto &[elemVInd] = data;
                REP(i, dim + 1) REP(j, i) {
                    int x = elemVInd[i];
                    int y = elemVInd[j];
                    if (idx2group[x] == idx2group[y] && idx2group[x] >= 0 && !mp[std::make_pair(x, y)]) {
                        mp[std::make_pair(x, y)] = true;
                        mp[std::make_pair(y, x)] = true;
                        sparse_pairs[idx2group[x]].emplace_back(idx2idx[x], idx2idx[y]);
                        sparse_pairs[idx2group[x]].emplace_back(idx2idx[y], idx2idx[x]);
                    }
                }
            });
            for (int idx = 0; idx < index_groups.size(); ++idx) {
                std::vector<VECTOR<T, dim>>& X = X_groups[idx];
                std::vector<VECTOR<T, dim>> x;
                std::vector<VECTOR<T, dim>> lambda = lambda_groups[idx];
                for (auto i : index_groups[idx]) x.push_back(std::get<0>(nodes.Get_Unchecked_Const(i)));
                SHAPE_MATCHING_DERIVATION<T, dim> dr(x, X);
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> raw(dr.n * dim, dr.n * dim);
                raw.setZero();
                for (auto p : sparse_pairs[idx]) {
                    int i = p.first, j = p.second;
                    MATRIX<T, dim> h;
                    REP(k, dr.n) {
                        MATRIX<T, dim> di = ((i == k ? (T)1 : (T)0) - (T)1 / dr.n) * MATRIX<T, dim>(1);
                        MATRIX<T, dim> dj = ((j == k ? (T)1 : (T)0) - (T)1 / dr.n) * MATRIX<T, dim>(1);
                        if (PARAMETER::Get("With_dRdx", true))
                        REP(a, dim) REP(b, dim) REP(c, dim) {
                            di(a, b) -= dr.dRdx(a, c, i, b) * (X[k] - dr.X_com)(c);
                            dj(a, b) -= dr.dRdx(a, c, j, b) * (X[k] - dr.X_com)(c);
                        }
                        h += mu * di.transpose() * dj;
                        VECTOR<T, dim> v = x[k] - dr.x_com - dr.R * (X[k] - dr.X_com);
                        if (PARAMETER::Get("With_d2Rdx2", true)) {
                            REP(a, dim) REP(b, dim) REP(c, dim) REP(d, dim)
                                h(a, b) -= mu * dr.d2Rdx2(c, d, i, a, j, b) * (X[k] - dr.X_com)(d) * v(c);
                            REP(a, dim) REP(b, dim) REP(c, dim) REP(d, dim)
                                h(a, b) -= dr.d2Rdx2(c, d, i, a, j, b) * (X[k] - dr.X_com)(d) * lambda[k](c);
                        }
                    }
                    REP(a, dim) REP(b, dim)
                        raw(i * dim + a, j * dim + b) = h(a, b);
                }
                if (projectPD) makePD(raw);
                REP(i, dr.n) REP(j, dr.n) REP(a, dim) REP(b, dim)
                    triplets.emplace_back(index_groups[idx][i] * dim + a, index_groups[idx][j] * dim + b, raw(i * dim + a, j * dim + b));
            }
        } else {
            for (int idx = 0; idx < index_groups.size(); ++idx) {
                std::vector<VECTOR<T, dim>>& X = X_groups[idx];
                std::vector<VECTOR<T, dim>> x;
                std::vector<VECTOR<T, dim>> lambda = lambda_groups[idx];
                for (auto i : index_groups[idx]) x.push_back(std::get<0>(nodes.Get_Unchecked_Const(i)));
                SHAPE_MATCHING_DERIVATION<T, dim> dr(x, X);
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> raw(dr.n * dim, dr.n * dim);
                REP(i, dr.n) REP(j, dr.n) {
                    MATRIX<T, dim> h;
                    REP(k, dr.n) {
                        MATRIX<T, dim> di = ((i == k ? (T)1 : (T)0) - (T)1 / dr.n) * MATRIX<T, dim>(1);
                        MATRIX<T, dim> dj = ((j == k ? (T)1 : (T)0) - (T)1 / dr.n) * MATRIX<T, dim>(1);
                        if (PARAMETER::Get("With_dRdx", true))
                        REP(a, dim) REP(b, dim) REP(c, dim) {
                            di(a, b) -= dr.dRdx(a, c, i, b) * (X[k] - dr.X_com)(c);
                            dj(a, b) -= dr.dRdx(a, c, j, b) * (X[k] - dr.X_com)(c);
                        }
                        h += mu * di.transpose() * dj;
                        if (PARAMETER::Get("With_d2Rdx2", true)) {
                            VECTOR<T, dim> v = x[k] - dr.x_com - dr.R * (X[k] - dr.X_com);
                            REP(a, dim) REP(b, dim) REP(c, dim) REP(d, dim)
                                h(a, b) -= mu * dr.d2Rdx2(c, d, i, a, j, b) * (X[k] - dr.X_com)(d) * v(c);
                            REP(a, dim) REP(b, dim) REP(c, dim) REP(d, dim)
                                h(a, b) -= dr.d2Rdx2(c, d, i, a, j, b) * (X[k] - dr.X_com)(d) * lambda[k](c);
                        }
                    }
                    REP(a, dim) REP(b, dim)
                        raw(i * dim + a, j * dim + b) = h(a, b);
                }
                if (projectPD) makePD(raw);
                REP(i, dr.n) REP(j, dr.n) REP(a, dim) REP(b, dim)
                    triplets.emplace_back(index_groups[idx][i] * dim + a, index_groups[idx][j] * dim + b, raw(i * dim + a, j * dim + b));
            }
        }
    }
};

}
