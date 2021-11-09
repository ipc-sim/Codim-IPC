#pragma once
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

template <class T, int dim>
class ABSTRACT_ENERGY;

template <class T, int dim>
class RIGID_CONSTRAINT_ENERGY : public ABSTRACT_ENERGY<T, dim> {
public:
    T penalty_mu = 1000000;
    FIXED_COROTATED<T, dim> arapAttr;
    void Precompute(MESH_ELEM<dim>& Elem, MESH_NODE<T, dim>& X, FIXED_COROTATED<T, dim>& fcr) {
        T lower_y = std::numeric_limits<T>::max();
        T upper_y = std::numeric_limits<T>::min();
        X.Each([&](const int i, auto data) {
            auto& [x] = data;
            lower_y = std::min(lower_y, x(1));
            upper_y = std::max(upper_y, x(1));
        });
        T tmp_y = lower_y + (upper_y - lower_y) * 0.6;
        lower_y = lower_y + (upper_y - lower_y) * 0.4;
        upper_y = tmp_y;
        arapAttr.Reserve(fcr.size);
        fcr.Join(Elem).Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu, elemVInd] = data;
            bool all_inside = true;
            for (int d = 0; d <= dim; ++d) {
                T y = std::get<0>(X.Get_Unchecked_Const(elemVInd(d)))(1);
                if (y < lower_y || y > upper_y)
                    all_inside = false;
            }
            if (all_inside) arapAttr.Insert(i, F, vol, 0, mu);
            else arapAttr.Insert(i, F, vol, 0, 0);
        });
    }

    void Synchronize_F(FIXED_COROTATED<T, dim>& elasticityAttr) {
        arapAttr.Join(elasticityAttr).Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu, _F, _vol, _lambda, _mu] = data;
            F = _F;
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
        Synchronize_F(elasticityAttr);
        FIXED_COROTATED_FUNCTOR<T, dim>::Compute_Psi(arapAttr, h * h * penalty_mu, elemAttr, value);
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
        Synchronize_F(elasticityAttr);
        FIXED_COROTATED_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(arapAttr, h * h * penalty_mu, elemAttr);
        Elem_To_Node(Elem, elemAttr, nodeAttr);
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
        Synchronize_F(elasticityAttr);
        typename FIXED_COROTATED_FUNCTOR<T, dim>::DIFFERENTIAL dP_div_dF;
        for (int i = 0; i < elemAttr.size; ++i) {
            dP_div_dF.Insert(i, Eigen::Matrix<T, dim * dim, dim * dim>::Zero());
        }
        FIXED_COROTATED_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress_Derivative(arapAttr, h * h * penalty_mu, true, dP_div_dF);
        Elem_To_Node(Elem, elemAttr, dP_div_dF, triplets);
    }
};

}