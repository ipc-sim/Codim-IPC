#pragma once

#include <Math/VECTOR.h>
#include <Math/CSR_MATRIX.h>
#include <Utils/MESHIO.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim, bool allocNA = true>
void Compute_Mass_And_Init_Velocity(
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim>& Elem,
    const SCALAR_STORAGE<T>& vol,
    const T& density,
    const VECTOR<T, dim>& velocity,
    MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    if constexpr (allocNA) {
        nodeAttr.Reserve(X.size);
        for (int nodeI = 0; nodeI < X.size; ++nodeI) {
            nodeAttr.Append(std::get<0>(X.Get_Unchecked(nodeI)), velocity, VECTOR<T, dim>(), 0.0);
        }
    }

    Elem.Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        const T &volI = std::get<0>(vol.Get_Unchecked_Const(id));
        T massEach = volI * density / (dim + 1);
        nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(elemVInd(0)) += massEach;
        nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(elemVInd(1)) += massEach;
        nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(elemVInd(2)) += massEach;
        if constexpr (dim == 3) {
            nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(elemVInd(3)) += massEach;
        }

        if constexpr (!allocNA) {
            nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::v>(elemVInd(0)) = velocity;
            nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::v>(elemVInd(1)) = velocity;
            nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::v>(elemVInd(2)) = velocity;
            if (dim == 3) {
                nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::v>(elemVInd(3)) = velocity;
            }
        }
    });
}

template <class T, int dim>
void Augment_Mass_Matrix_And_Body_Force(
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim>& Elem,
    const SCALAR_STORAGE<T>& vol,
    const T& density,
    const VECTOR<T, dim>& bodyAcc,
    CSR_MATRIX<T>& M, // mass matrix
    std::vector<T>& b)
{
    if (b.size() != X.size * dim || M.Get_Matrix().rows() != X.size * dim ||
        M.Get_Matrix().cols() != X.size * dim) 
    {
        std::cout << "mass matrix or bodyforce vector uninitialized!" << std::endl;
        exit(-1);
    }

    CSR_MATRIX<T> addM;
    std::vector<Eigen::Triplet<T>> triplets;
    Elem.Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        const T &volI = std::get<0>(vol.Get_Unchecked_Const(id));
        T massEach = volI * density / (dim + 1);
        for (int d = 0; d < dim + 1; ++d) {
            b[elemVInd(d) * dim] += massEach * bodyAcc[0];
            b[elemVInd(d) * dim + 1] += massEach * bodyAcc[1];
            if constexpr (dim == 3) {
                b[elemVInd(d) * dim + 2] += massEach * bodyAcc[2];
            }

            triplets.emplace_back(elemVInd(d) * dim, elemVInd(d) * dim, massEach);
            triplets.emplace_back(elemVInd(d) * dim + 1, elemVInd(d) * dim + 1, massEach);
            if constexpr (dim == 3) {
                triplets.emplace_back(elemVInd(d) * dim + 2, elemVInd(d) * dim + 2, massEach);
            }
        }
    });
    addM.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
    M.Get_Matrix() += addM.Get_Matrix();
}

template <class T, int dim>
void DF_Div_DX_Mult(const MATRIX<T, dim>& right,
    const MATRIX<T, dim>& IB,
    VECTOR<T, dim> result[dim + 1])
{
    if constexpr (dim == 2) {
        result[1](0) = right(0, 0) * IB(0, 0) + right(0, 1) * IB(0, 1);
        result[1](1) = right(1, 0) * IB(0, 0) + right(1, 1) * IB(0, 1);
        result[2](0) = right(0, 0) * IB(1, 0) + right(0, 1) * IB(1, 1);
        result[2](1) = right(1, 0) * IB(1, 0) + right(1, 1) * IB(1, 1);
        result[0](0) = -result[1](0) - result[2](0);
        result[0](1) = -result[1](1) - result[2](1);
    }
    else {
        auto dotRows = [&](const MATRIX<T, 3>& a, int i, const MATRIX<T, 3>& b, int j) {
            return a(i, 0) * b(j, 0) + a(i, 1) * b(j, 1) + a(i, 2) * b(j, 2);
        };
        result[1](0) = dotRows(IB, 0, right, 0);
        result[1](1) = dotRows(IB, 0, right, 1);
        result[1](2) = dotRows(IB, 0, right, 2);
        result[2](0) = dotRows(IB, 1, right, 0);
        result[2](1) = dotRows(IB, 1, right, 1);
        result[2](2) = dotRows(IB, 1, right, 2);
        result[3](0) = dotRows(IB, 2, right, 0);
        result[3](1) = dotRows(IB, 2, right, 1);
        result[3](2) = dotRows(IB, 2, right, 2);
        result[0](0) = -result[1](0) - result[2](0) - result[3](0);
        result[0](1) = -result[1](1) - result[2](1) - result[3](1);
        result[0](2) = -result[1](2) - result[2](2) - result[3](2);
    }
}

template <class T, int dim>
void DF_Div_DX_Mult2(const Eigen::Matrix<T, dim * dim, dim * dim>& middle,
    const MATRIX<T, dim>& IB, const VECTOR<int, dim + 1>& elemVInd,
    Eigen::Triplet<T> result[(dim + 1) * dim * (dim + 1) * dim])
{
    if constexpr (dim == 2) {
        int indMap[(dim + 1) * dim] = {
            elemVInd[0] * dim,
            elemVInd[0] * dim + 1,
            elemVInd[1] * dim,
            elemVInd[1] * dim + 1,
            elemVInd[2] * dim,
            elemVInd[2] * dim + 1,
        };
        T intermediate[(dim + 1) * dim][dim * dim];
        for (int colI = 0; colI < dim * dim; colI++) {
            const T _000 = middle(0, colI) * IB(0, 0);
            const T _010 = middle(0, colI) * IB(1, 0);
            const T _101 = middle(2, colI) * IB(0, 1);
            const T _111 = middle(2, colI) * IB(1, 1);
            const T _200 = middle(1, colI) * IB(0, 0);
            const T _210 = middle(1, colI) * IB(1, 0);
            const T _301 = middle(3, colI) * IB(0, 1);
            const T _311 = middle(3, colI) * IB(1, 1);

            intermediate[2][colI] = _000 + _101;
            intermediate[3][colI] = _200 + _301;
            intermediate[4][colI] = _010 + _111;
            intermediate[5][colI] = _210 + _311;
            intermediate[0][colI] = -intermediate[2][colI] - intermediate[4][colI];
            intermediate[1][colI] = -intermediate[3][colI] - intermediate[5][colI];
        }
        for (int colI = 0; colI < (dim + 1) * dim; colI++) {
            const T _000 = intermediate[colI][0] * IB(0, 0);
            const T _010 = intermediate[colI][0] * IB(1, 0);
            const T _101 = intermediate[colI][2] * IB(0, 1);
            const T _111 = intermediate[colI][2] * IB(1, 1);
            const T _200 = intermediate[colI][1] * IB(0, 0);
            const T _210 = intermediate[colI][1] * IB(1, 0);
            const T _301 = intermediate[colI][3] * IB(0, 1);
            const T _311 = intermediate[colI][3] * IB(1, 1);

            int colIBegin = colI * (dim + 1) * dim;
            result[colIBegin + 2] = Eigen::Triplet<T>(indMap[2], indMap[colI], _000 + _101);
            result[colIBegin + 3] = Eigen::Triplet<T>(indMap[3], indMap[colI], _200 + _301);
            result[colIBegin + 4] = Eigen::Triplet<T>(indMap[4], indMap[colI], _010 + _111);
            result[colIBegin + 5] = Eigen::Triplet<T>(indMap[5], indMap[colI], _210 + _311);
            result[colIBegin + 0] = Eigen::Triplet<T>(indMap[0], indMap[colI], 
                -result[colIBegin + 2].value() - result[colIBegin + 4].value());
            result[colIBegin + 1] = Eigen::Triplet<T>(indMap[1], indMap[colI], 
                -result[colIBegin + 3].value() - result[colIBegin + 5].value());
        }
    }
    else {
        int indMap[(dim + 1) * dim] = {
            elemVInd[0] * dim,
            elemVInd[0] * dim + 1,
            elemVInd[0] * dim + 2,
            elemVInd[1] * dim,
            elemVInd[1] * dim + 1,
            elemVInd[1] * dim + 2,
            elemVInd[2] * dim,
            elemVInd[2] * dim + 1,
            elemVInd[2] * dim + 2,
            elemVInd[3] * dim,
            elemVInd[3] * dim + 1,
            elemVInd[3] * dim + 2,
        };
        T intermediate[(dim + 1) * dim][dim * dim];
        auto dotRows = [&](const MATRIX<T, 3>& a, int i, const T* b, int j) {
            return a(i, 0) * b[j] + a(i, 1) * b[3 + j] + a(i, 2) * b[6 + j];
        };
        for (int colI = 0; colI < dim * dim; colI++) {
            const T *right = middle.col(colI).data();
            intermediate[3][colI] = dotRows(IB, 0, right, 0);
            intermediate[4][colI] = dotRows(IB, 0, right, 1);
            intermediate[5][colI] = dotRows(IB, 0, right, 2);
            intermediate[6][colI] = dotRows(IB, 1, right, 0);
            intermediate[7][colI] = dotRows(IB, 1, right, 1);
            intermediate[8][colI] = dotRows(IB, 1, right, 2);
            intermediate[9][colI] = dotRows(IB, 2, right, 0);
            intermediate[10][colI] = dotRows(IB, 2, right, 1);
            intermediate[11][colI] = dotRows(IB, 2, right, 2);
            intermediate[0][colI] = -intermediate[3][colI] - intermediate[6][colI] - intermediate[9][colI];
            intermediate[1][colI] = -intermediate[4][colI] - intermediate[7][colI] - intermediate[10][colI];
            intermediate[2][colI] = -intermediate[5][colI] - intermediate[8][colI] - intermediate[11][colI];
        }
        for (int rowI = 0; rowI < (dim + 1) * dim; rowI++) {
            const T *right = intermediate[rowI];
            int rowIBegin = rowI * (dim + 1) * dim;
            result[rowIBegin + 3] = Eigen::Triplet<T>(indMap[rowI], indMap[3], dotRows(IB, 0, right, 0));
            result[rowIBegin + 4] = Eigen::Triplet<T>(indMap[rowI], indMap[4], dotRows(IB, 0, right, 1));
            result[rowIBegin + 5] = Eigen::Triplet<T>(indMap[rowI], indMap[5], dotRows(IB, 0, right, 2));
            result[rowIBegin + 6] = Eigen::Triplet<T>(indMap[rowI], indMap[6], dotRows(IB, 1, right, 0));
            result[rowIBegin + 7] = Eigen::Triplet<T>(indMap[rowI], indMap[7], dotRows(IB, 1, right, 1));
            result[rowIBegin + 8] = Eigen::Triplet<T>(indMap[rowI], indMap[8], dotRows(IB, 1, right, 2));
            result[rowIBegin + 9] = Eigen::Triplet<T>(indMap[rowI], indMap[9], dotRows(IB, 2, right, 0));
            result[rowIBegin + 10] = Eigen::Triplet<T>(indMap[rowI], indMap[10], dotRows(IB, 2, right, 1));
            result[rowIBegin + 11] = Eigen::Triplet<T>(indMap[rowI], indMap[11], dotRows(IB, 2, right, 2));
            result[rowIBegin] = Eigen::Triplet<T>(indMap[rowI], indMap[0], 
                -result[rowIBegin + 3].value() - result[rowIBegin + 6].value() - result[rowIBegin + 9].value());
            result[rowIBegin + 1] = Eigen::Triplet<T>(indMap[rowI], indMap[1], 
                -result[rowIBegin + 4].value() - result[rowIBegin + 7].value() - result[rowIBegin + 10].value());
            result[rowIBegin + 2] = Eigen::Triplet<T>(indMap[rowI], indMap[2], 
                -result[rowIBegin + 5].value() - result[rowIBegin + 8].value() - result[rowIBegin + 11].value());
        }
    }
}

template <class T, int dim>
void Elem_To_Node(
    MESH_ELEM<dim>& Elem,
    const MESH_ELEM_ATTR<T, dim>& elemAttr,
    MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    TIMER_FLAG("elem2node");
    std::vector<VECTOR<T, dim>> grad(Elem.size * (dim + 1));
    Elem.Par_Each([&](int id, auto data) {
        auto &[elemVInd] = data;

        DF_Div_DX_Mult(std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked_Const(id)), 
            std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked_Const(id)), 
            grad.data() + id * (dim + 1));
    });

    Elem.Each([&](int id, auto data) {
        auto &[elemVInd] = data;

        nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(elemVInd(0)) += grad[id * (dim + 1)];
        nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(elemVInd(1)) += grad[id * (dim + 1) + 1];
        nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(elemVInd(2)) += grad[id * (dim + 1) + 2];
        if constexpr (dim == 3) {
            nodeAttr.template Get_Component_Unchecked<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(elemVInd(3)) += grad[id * (dim + 1) + 3];
        }
    });
}

template <class T, int dim>
void Elem_To_Node(
    MESH_ELEM<dim>& Elem,
    const MESH_ELEM_ATTR<T, dim>& elemAttr,
    const BASE_STORAGE<Eigen::Matrix<T, dim * dim, dim * dim>>& dP_div_dF,
    std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("elem2node");
    const int localMtrEntryAmt = (dim + 1) * dim * (dim + 1) * dim;
    int old_size = triplets.size();
    triplets.resize(old_size + dP_div_dF.size * localMtrEntryAmt);
    Elem.Par_Each([&](int id, auto data) {
        auto &[elemVInd] = data;

        DF_Div_DX_Mult2(std::get<0>(dP_div_dF.Get_Unchecked_Const(id)), 
            std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked_Const(id)), 
            elemVInd, triplets.data() + old_size + id * localMtrEntryAmt);
    });
}

void Export_Elem_To_Node(py::module& m) {
    m.def("Compute_Mass_And_Init_Velocity", &Compute_Mass_And_Init_Velocity<double, 2>);
    m.def("Compute_Mass_And_Init_Velocity", &Compute_Mass_And_Init_Velocity<double, 3>);
    m.def("Compute_Mass_And_Init_Velocity_NoAlloc", &Compute_Mass_And_Init_Velocity<double, 3, false>);
    m.def("Augment_Mass_Matrix_And_Body_Force", &Augment_Mass_Matrix_And_Body_Force<double, 3>);
}

}

