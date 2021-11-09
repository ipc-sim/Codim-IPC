#pragma once

#include <Physics/FIXED_COROTATED.h>
#include <FEM/DEFORMATION_GRADIENT.h>
#include <FEM/ELEM_TO_NODE.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim>
void Advance_One_Step_SE(MESH_ELEM<dim>& Elem,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const VECTOR<T, dim>& gravity, T h,
    MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr)
{
    Compute_Deformation_Gradient(X, Elem, elemAttr, elasticityAttr);

    FIXED_COROTATED_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(elasticityAttr, 1.0, elemAttr);

    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Elem_To_Node(Elem, elemAttr, nodeAttr);

    TIMER_FLAG("update V and X");
    X.Join(nodeAttr).Par_Each([&](int id, auto data) {
        auto &[x, x0, v, g, m] = data;
        v += h * (gravity - g / m);
        x += h * v;
    });

    DBC.Par_Each([&](int id, auto data) {
        auto &[dbcI] = data;
        VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
        x(0) = dbcI(1);
        x(1) = dbcI(2);
        if constexpr (dim == 3) {
            x(2) = dbcI(3);
        }
    });
}

}
