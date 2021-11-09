#pragma once
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

template <class T, int dim>
class ABSTRACT_ENERGY;

template <class T, int dim>
class INERTIA_ENERGY : public ABSTRACT_ENERGY<T, dim> {
public:
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
        if (PARAMETER::Get("Terminate", false))
            return;
        std::vector<T> inertia(X.size);
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            inertia[id] = 0.5 * m * (x - std::get<0>(Xtilde.Get_Unchecked(id))).length2();
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
        if (PARAMETER::Get("Terminate", false))
            return;
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            g += m * (x - std::get<0>(Xtilde.Get_Unchecked(id)));
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
        TIMER_FLAG("Compute_IncPotential_Hessian_Inertia");
        nodeAttr.Each([&](int id, auto data) {
            auto &[x0, v, g, m] = data;
            triplets.emplace_back(id * dim, id * dim, m);
            triplets.emplace_back(id * dim + 1, id * dim + 1, m);
            if constexpr (dim == 3) {
                triplets.emplace_back(id * dim + 2, id * dim + 2, m);
            }
        });
    }
};

}