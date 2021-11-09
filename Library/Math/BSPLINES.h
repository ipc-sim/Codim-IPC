#pragma once
//#####################################################################
// Class BSPLINE_WEIGHTS
// Function Base_Node
// Function Compute_BSpline_Weights
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
inline int Base_Node(const T& x) {
    if constexpr (interpolation_degree <= 1) {
        return MATH_TOOLS::Int_Floor(x);
    } else {
        return MATH_TOOLS::Int_Floor(x - (T)0.5 * (interpolation_degree - 1));
    }
}

//#####################################################################
// Function Base_Node
//
// multi-dimensions
//#####################################################################
template <int interpolation_degree, class T, int dim>
inline VECTOR<int, dim> Base_Node(const VECTOR<T, dim>& x)
{
    VECTOR<int, dim> base;
    for (int d = 0; d < dim; d++)
        base(d) = Base_Node<interpolation_degree, T>(x(d));
    return base;
}

//#####################################################################
// Function Compute_BSpline_Weights
//
// Compute constant weights, x the point in a h=1 grid,
// w the weights, dw the weight gradients.
//#####################################################################
template <class T>
inline void Compute_BSpline_Weights(const T x, int& base_node, VECTOR<T, 1>& w, VECTOR<T, 1>* dw = 0, VECTOR<T, 1>* ddw = 0)
{
    base_node = Base_Node<0>(x);
    w(0) = 1;
    if (dw)
        (*dw)(0) = 0;
    if (ddw)
        (*ddw)(0) = 0;
}

//#####################################################################
// Function Compute_BSpline_Weights
//
// Compute linear B-spline weights, x the point in a h=1 grid,
// w the weights, dw the weight gradients.
//#####################################################################
template <class T>
inline void Compute_BSpline_Weights(const T x, int& base_node, VECTOR<T, 2>& w, VECTOR<T, 2>* dw = 0, VECTOR<T, 2>* ddw = 0)
{
    base_node = Base_Node<1>(x);
    T dx = x - base_node;
    w(0) = 1 - dx;
    w(1) = dx;
    if (dw) {
        (*dw)(0) = -1;
        (*dw)(1) = 1;
    }
    if (ddw) {
        (*ddw)(0) = 0;
        (*ddw)(1) = 0;
    }
}

//#####################################################################
// Function Compute_BSpline_Weights
//
// Compute quadratic B-spline weights, x the point in a h=1 grid,
// w the weights, dw the weight gradients.
//#####################################################################
template <class T>
inline void Compute_BSpline_Weights(const T x, int& base_node, VECTOR<T, 3>& w, VECTOR<T, 3>* dw = 0, VECTOR<T, 3>* ddw = 0)
{
    base_node = Base_Node<2>(x);
    T d0 = x - base_node;
    T z = ((T)1.5 - d0);
    T z2 = z * z;
    w(0) = (T)0.5 * z2;
    T d1 = d0 - 1;
    w(1) = (T)0.75 - d1 * d1;
    T d2 = 1 - d1;
    T zz = (T)1.5 - d2;
    T zz2 = zz * zz;
    w(2) = (T)0.5 * zz2;

    if (dw) {
        (*dw)(0) = -z;
        (*dw)(1) = -(T)2 * d1;
        (*dw)(2) = zz;
    }

    if (ddw) {
        (*ddw)(0) = 1;
        (*ddw)(1) = -2;
        (*ddw)(2) = 1;
    }
}

//#####################################################################
// Function Compute_BSpline_Weights
//
// Compute cubic B-spline weights, x the point in a h=1 grid,
// w the weights, dw the weight gradients.
//#####################################################################
template <class T>
inline void Compute_BSpline_Weights(const T x, int& base_node, VECTOR<T, 4>& w, VECTOR<T, 4>* dw = 0, VECTOR<T, 4>* ddw = 0)
{
    base_node = Base_Node<3>(x);
    T d0 = x - base_node;
    T z = 2 - d0;
    T z3 = z * z * z;
    w(0) = ((T)1 / (T)6) * z3;
    T d1 = d0 - 1;
    T zz2 = d1 * d1;
    w(1) = ((T)0.5 * d1 - 1) * zz2 + (T)2 / (T)3;
    T d2 = 1 - d1;
    T zzz2 = d2 * d2;
    w(2) = ((T)0.5 * d2 - 1) * zzz2 + (T)2 / (T)3;
    T d3 = 1 + d2;
    T zzzz = 2 - d3;
    T zzzz3 = zzzz * zzzz * zzzz;
    w(3) = ((T)1 / (T)6) * zzzz3;

    if (dw) {
        (*dw)(0) = -(T)0.5 * z * z;
        (*dw)(1) = ((T)1.5 * d1 - (T)2) * d1;
        (*dw)(2) = (-(T)1.5 * d2 + (T)2) * d2;
        (*dw)(3) = (T)0.5 * zzzz * zzzz;
    }

    if (ddw) {
        (*ddw)(0) = (T)2 + base_node - x;
        (*ddw)(1) = -(T)2 + (T)3 * (-(T)1 - base_node + x);
        (*ddw)(2) = -(T)2 + (T)3 * ((T)2 + base_node - x);
        (*ddw)(3) = -(T)1 - base_node + x;
    }
}

//#####################################################################
// Class BSPLINE_WEIGHTS
//#####################################################################
template <class T, int dim, int degree>
class BSPLINE_WEIGHTS {
    const T dx;

public:
    static constexpr int interpolation_degree = degree;
    VECTOR<T, interpolation_degree + 1> w[dim];
    VECTOR<T, interpolation_degree + 1> dw[dim];
    const T one_over_dx;
    VECTOR<int, dim> base_node;

    //#################################################################
    // Constructor
    //#################################################################
    BSPLINE_WEIGHTS(const VECTOR<T, dim>& X, T dx)
            : dx(dx), one_over_dx(1 / dx), base_node(VECTOR<int, dim>())
    {
        Compute(X);
    }

    //#################################################################
    // Function Compute
    //#################################################################
    void Compute(const VECTOR<T, dim>& X)
    {
        VECTOR<T, dim> X_index_space = one_over_dx * X;
        for (int d = 0; d < dim; d++)
            Compute_BSpline_Weights(X_index_space(d), base_node(d), w[d], &dw[d]);
    }
};

}
