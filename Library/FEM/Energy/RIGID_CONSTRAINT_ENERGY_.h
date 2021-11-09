#pragma once
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

template <class T, int dim>
class ABSTRACT_ENERGY;

template <typename T, int n, int m>
using Matrix = Eigen::Matrix<T, n, m, 0, n, m>;
template <typename T, int dim>
using Vector = Eigen::Matrix<T, dim, 1, 0, dim, 1>;

template <class T, int dim>
class RIGID_CONSTRAINT_ENERGY : public ABSTRACT_ENERGY<T, dim> {
public:
    using TVStack = Matrix<T, Eigen::Dynamic, dim>;
    using TStack = Matrix<T, Eigen::Dynamic, 1>;
    Matrix<T, dim, dim> ATA_inv;
    TVStack A;
    TStack b;
    TStack P;
    MESH_NODE<T, dim> projected_X;

VECTOR<T, dim> Get_Centroid(const MESH_NODE<T, dim>& X) {
    VECTOR<T, dim> centroid;
    for (int i = 0; i < X.size; ++i)
        centroid += X.Get_Unchecked_Const(i);
    centroid /= (T) X.size;
    return centroid;
}

void Precompute(const MESH_NODE<T, dim>& X) {
    A = TVStack::Zero(X.size * dim, dim);
    for (int i = 0; i < X.size; ++i) {
        Vector<T, dim> x0, xi;
        for (int d = 0; d < dim; ++d) {
            x0(d) = X.Get_Unchecked_Const(0)(d);
            xi(d) = X.Get_Unchecked_Const(i)(d);
        }
        auto centroid = Get_Centroid(X);
        x0 -= centroid;
        xi -= centroid;
        Matrix<T, dim, dim> Ri = Eigen::Quaternionf().setFromTwoVectors(x0,xi).toRotationMatrix();
        T si = xi.length() / x0.length();
        A.template block<dim, dim>(i * dim, 0) = Ri * si;
    }
    ATA_inv = (A.transpose() * A).inverse();
    Append_Attribute(X, projected_X);
}

void Project(const MESH_NODE<T, dim>& X) {
    static bool first_run = true;
    if (first_run) {
        first_run = false;
        Precompute(X);
    }
    b = TStack::Zero(X.size * dim);
    auto centroid = Get_Centroid(X);
    for (int i = 0; i < X.size; ++i) {
        VECTOR<T, dim> xi = X.Get_Unchecked_Const(i) - centroid;
        for (int d = 0; d < dim; ++d)
            b(i * dim + d) = xi(d);
    }
    P = A * ATA_inv * A.transpose() * b;
    projected_X.Par_Each([&](const int i, auto data) {
        auto& [prjected_xi] = data;
        prjected_xi = centroid;
        for (int d = 0; d < dim; ++d)
            prjected_xi += P(i * dim + d);
    });
}

void Compute_IncPotential(
    MESH_ELEM<dim>& Elem,
    const VECTOR<T, dim>& gravity,
    T h, MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xtilde,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    T dHat2, T kappa[],
    double& value
) {
    Project(X);
    X.Each([&](int id, auto data) {
        auto &[x] = data;
        value += (x - projected_X.Get_Unchecked_Const(id)).length2();
    });
}
void Compute_IncPotential_Gradient(
    MESH_ELEM<dim>& Elem,
    const VECTOR<T, dim>& gravity,
    T h, MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xtilde,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    T dHat2, T kappa[]
) {
    Project(X);
    X.Join(nodeAttr).Par_Each([&](int id, auto data) {
        auto &[x, v, g, m] = data;
        g += (T)2 * (x - projected_X.Get_Unchecked_Const(id));
    });
}
void Compute_IncPotential_Hessian(
    MESH_ELEM<dim>& Elem,
    T h, MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    T dHat2, T kappa[],
    std::vector<Eigen::Triplet<T>>& triplets
) {
    Project(X);
    nodeAttr.Each([&](int id, auto data) {
        auto &[v, g, m] = data;
        triplets.emplace_back(id * dim, id * dim, 2);
        triplets.emplace_back(id * dim + 1, id * dim + 1, 2);
        if constexpr (dim == 3) {
            triplets.emplace_back(id * dim + 2, id * dim + 2, 2);
        }
    });
}
};

}