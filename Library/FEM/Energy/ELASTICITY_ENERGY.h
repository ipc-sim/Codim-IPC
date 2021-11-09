#pragma once
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

template <class T, int dim>
class ABSTRACT_ENERGY;

template <class T, int dim>
class ELASTICITY_ENERGY : public ABSTRACT_ENERGY<T, dim> {
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
        if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("FCR"))
            FIXED_COROTATED_FUNCTOR<T, dim>::Compute_Psi(elasticityAttr, h * h, elemAttr, value);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("NH"))
            NEOHOOKEAN_FUNCTOR<T, dim>::Compute_Psi(elasticityAttr, h * h, elemAttr, value);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("SD"))
            SYMMETRIC_DIRICHLET_FUNCTOR<T, dim>::Compute_Psi(elasticityAttr, h * h, elemAttr, value);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("SH"))
            STVK_HENCKY_FUNCTOR<T, dim>::Compute_Psi(elasticityAttr, h * h, elemAttr, value);
        else {
            puts("Please set Elasticity_model");
            exit(0);
        }
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
        if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("FCR"))
            FIXED_COROTATED_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(elasticityAttr, h * h, elemAttr);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("NH"))
            NEOHOOKEAN_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(elasticityAttr, h * h, elemAttr);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("SD"))
            SYMMETRIC_DIRICHLET_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(elasticityAttr, h * h, elemAttr);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("SH"))
            STVK_HENCKY_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(elasticityAttr, h * h, elemAttr);
        else {
            puts("Please set Elasticity_model");
            exit(0);
        }
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
        TIMER_FLAG("Compute_IncPotential_Hessian_Elasticity");
        typename FIXED_COROTATED_FUNCTOR<T, dim>::DIFFERENTIAL dP_div_dF;
        dP_div_dF.Reserve(elemAttr.size);
        for (int i = 0; i < elemAttr.size; ++i) {
            dP_div_dF.Insert(i, Eigen::Matrix<T, dim * dim, dim * dim>::Zero());
        }
        if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("FCR"))
            FIXED_COROTATED_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress_Derivative(elasticityAttr, h * h, true, dP_div_dF);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("NH"))
            NEOHOOKEAN_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress_Derivative(elasticityAttr, h * h, true, dP_div_dF);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("SD"))
            SYMMETRIC_DIRICHLET_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress_Derivative(elasticityAttr, h * h, true, dP_div_dF);
        else if (PARAMETER::Get("Elasticity_model", std::string("")) == std::string("SH"))
            STVK_HENCKY_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress_Derivative(elasticityAttr, h * h, true, dP_div_dF);
        else {
            puts("Please set Elasticity_model");
            exit(0);
        }
        Elem_To_Node(Elem, elemAttr, dP_div_dF, triplets);
    }
};

}