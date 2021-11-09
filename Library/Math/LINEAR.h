#pragma once
//#####################################################################
// Class LINEAR WEIGHTS
// Store the linear weights into arrays
// Function Compute_Linear_Weights
//#####################################################################
#include <Math/MATH_TOOLS.h>
#include <Math/VECTOR.h>
#include <iostream>

namespace JGSL {

//#####################################################################
// Function Base_Node
//
// 1D
//#####################################################################
template <int interpolation_degree, class T>
inline int Base_Node_L(const T& x) {
    if constexpr (interpolation_degree <= 1) {
        return MATH_TOOLS::Int_Floor(x);
    }
}

//#####################################################################
// Function Compute_Linear_Weights
// Compute Linear Weights, x the point in a h=1 grid,
// w the weights
//#####################################################################
template <class T>
inline void Compute_Linear_Weights(const T x, int& base_node, T &w)
{
    base_node = Base_Node_L<1>(x);
    T dx = x - (T)base_node;
    w = (T)(1 - dx);
}

//#####################################################################
// Class LINEAR_WEIGHTS
//#####################################################################
template <class T, int dim>
class LINEAR_WEIGHTS {
    const T dx;
public:
    VECTOR<T, dim> w; // a pair of weights
    const T one_over_dx;
    VECTOR<int, dim> base_node;

    //#################################################################
    // Constructor
    //#################################################################
    LINEAR_WEIGHTS(const VECTOR<T, dim>& X, T dx)
            : dx(dx),
              one_over_dx(1 / dx),
              base_node(VECTOR<int, dim>())
    {
        Compute(X);
    }

    //#################################################################
    // Function Compute
    //#################################################################
    void Compute(const VECTOR<T, dim>& X)
    {
        VECTOR<T, dim> tmp = one_over_dx * X;
        for (int d = 0; d < dim; d++) {
            Compute_Linear_Weights(tmp(d), base_node(d), w(d));
        }
    }
};

}
