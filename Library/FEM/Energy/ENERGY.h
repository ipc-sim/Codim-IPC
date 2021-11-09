#pragma once

#include <FEM/Energy/ELASTICITY_ENERGY.h>
#include <FEM/Energy/INERTIA_ENERGY.h>
#include <FEM/Energy/NEUMANN_ENERGY.h>
#include <FEM/Energy/SHAPE_MATCHING_ENERGY.h>
#include <FEM/Energy/IPC_ENERGY.h>
#include <FEM/Energy/RIGID_CONSTRAINT_ENERGY.h>
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

template <class T, int dim>
class ABSTRACT_ENERGY {
public:
    virtual void Compute_IncPotential(
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
    ) = 0;
    virtual void Compute_IncPotential_Gradient(
        MESH_ELEM<dim>& Elem,
        const VECTOR<T, dim>& gravity,
        T h, MESH_NODE<T, dim>& X,
        MESH_NODE<T, dim>& Xtilde,
        MESH_NODE_ATTR<T, dim>& nodeAttr,
        MESH_ELEM_ATTR<T, dim>& elemAttr,
        FIXED_COROTATED<T, dim>& elasticityAttr,
        std::vector<VECTOR<int, dim + 1>>& constraintSet,
        T dHat2, T kappa[]
    ) = 0;
    virtual void Compute_IncPotential_Hessian(
        MESH_ELEM<dim>& Elem,
        T h, MESH_NODE<T, dim>& X,
        MESH_NODE_ATTR<T, dim>& nodeAttr,
        MESH_ELEM_ATTR<T, dim>& elemAttr,
        FIXED_COROTATED<T, dim>& elasticityAttr,
        std::vector<VECTOR<int, dim + 1>>& constraintSet,
        T dHat2, T kappa[],
        std::vector<Eigen::Triplet<T>>& triplets
    ) = 0;
};

template <class T, int dim>
class ENERGY {
public:
    std::vector<std::shared_ptr<ABSTRACT_ENERGY<T, dim>>> energies;
    void Add(std::shared_ptr<ABSTRACT_ENERGY<T, dim>> e) { energies.push_back(e); }
    void Clear() { energies.clear(); }
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
        TIMER_FLAG("Compute_IncPotential");
        value = 0;
        for (auto e : energies)
            e->Compute_IncPotential(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr, constraintSet, dHat2, kappa, value);
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
        TIMER_FLAG("Compute_IncPotential_Gradient");
        nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
        for (auto e : energies)
            e->Compute_IncPotential_Gradient(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr, constraintSet, dHat2, kappa);
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
        TIMER_FLAG("Compute_IncPotential_Hessian");
        triplets.clear();
        for (auto e : energies)
            e->Compute_IncPotential_Hessian(Elem, h, X, nodeAttr, elemAttr, elasticityAttr, constraintSet, dHat2, kappa, triplets);
    }
};

template <class T, int dim>
std::vector<int> select(MESH_NODE<T, dim>& X, const VECTOR<T, dim>& lower, const VECTOR<T, dim>& upper) {
    VECTOR<T, dim> bb_lower(std::numeric_limits<T>::max());
    VECTOR<T, dim> bb_upper(std::numeric_limits<T>::min());
    X.Each([&](const int i, auto data) {
        auto& [x] = data;
        for (int d = 0; d < dim; ++d) {
            bb_lower(d) = std::min(bb_lower(d), x(d));
            bb_upper(d) = std::max(bb_upper(d), x(d));
        }
    });
    std::vector<int> v;
    X.Each([&](const int i, auto data) {
        auto& [x] = data;
        bool inside = true;
        for (int d = 0; d < dim; ++d) {
            T ratio = (x(d) - bb_lower(d)) / (bb_upper(d) - bb_lower(d));
            if (ratio < lower(d) || ratio > upper(d))
                inside = false;
        }
        if (inside) v.push_back(i);
    });
    return v;
}

}