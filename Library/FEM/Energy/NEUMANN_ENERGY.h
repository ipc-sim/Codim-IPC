#pragma once
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

template <class T, int dim>
class ABSTRACT_ENERGY;

template <class T, int dim>
class NEUMANN_ENERGY : public ABSTRACT_ENERGY<T, dim> {
public:
    std::vector<VECTOR<T, dim>> forces;
    void Precompute(MESH_NODE<T, dim>& X, T current_h) {
        T lower_y = std::numeric_limits<T>::max();
        T upper_y = std::numeric_limits<T>::min();
        X.Each([&](const int i, auto data) {
            auto& [x] = data;
            lower_y = std::min(lower_y, x(0));
            upper_y = std::max(upper_y, x(0));
        });
        T tmp_y = lower_y + (upper_y - lower_y) * 0.8;
        lower_y = lower_y + (upper_y - lower_y) * 0.2;
        upper_y = tmp_y;

        forces.resize(X.size, VECTOR<T, dim>());
        X.Par_Each([&](int i, auto data) {
            auto &[x] = data;
            if (x(0) > upper_y) forces[i](0) = -current_h;
            if (x(0) < lower_y) forces[i](0) = current_h;
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
        std::vector<T> inertia(X.size);
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, v, g, m] = data;
            inertia[id] = - h * h * m * x.dot(forces[id]);
        });
        value = std::accumulate(inertia.begin(), inertia.end(), value);
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
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, v, g, m] = data;
            g -= h * h * m * forces[id];
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
    }
};

}