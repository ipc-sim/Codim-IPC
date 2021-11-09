#pragma once
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

template <class T, int dim>
class ABSTRACT_ENERGY;

template <class T, int dim, bool elasticIPC>
class IPC_ENERGY : public ABSTRACT_ENERGY<T, dim> {
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
        Compute_Barrier<T, dim, elasticIPC>(X, nodeAttr, constraintSet, 
            std::vector<VECTOR<T, 2>>(constraintSet.size(), VECTOR<T, 2>(1, dHat2)),
            dHat2, kappa, T(0), value);
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
        Compute_Barrier_Gradient<T, dim, elasticIPC>(X, constraintSet, 
            std::vector<VECTOR<T, 2>>(constraintSet.size(), VECTOR<T, 2>(1, dHat2)),
            dHat2, kappa, T(0), nodeAttr);
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
        Compute_Barrier_Hessian<T, dim, elasticIPC>(X, nodeAttr, constraintSet,
            std::vector<VECTOR<T, 2>>(constraintSet.size(), VECTOR<T, 2>(1, dHat2)),
            dHat2, kappa, T(0), true, triplets);
    }
};

}