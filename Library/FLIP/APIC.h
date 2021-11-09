#pragma once
#include <Math/VECTOR.h>
#include <Grid/SPARSE_GRID.h>
#include <FLIP/TRANSFER_FLIP.h>
#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <Math/LINEAR.h>

namespace py = pybind11;
namespace JGSL {

template<class T, int dim>
VECTOR<T,dim> Get_Affine_Matrix(
    SPARSE_GRID<T, dim>& g,
    const VECTOR<T,dim>& pos,
    const VECTOR<T,dim>& origin,
    const int axis,
    const T dx)
{
  VECTOR<T, dim> p = pos - origin;
  VECTOR<T, dim> c;
  if constexpr (dim == 2) {
    if (axis == 0) {
        LINEAR_WEIGHTS<T, 2> l_u(p - VECTOR<T,2>(0.0, 0.5*dx), dx);
        g.get_gradient_from_Kernel(l_u, c);
        c = c / dx;
    } else if (axis == 1) {
        LINEAR_WEIGHTS<T, 2> l_v(p - VECTOR<T,2>(0.5*dx, 0.0), dx);
        g.get_gradient_from_Kernel(l_v, c);
        c = c / dx;
    }
  }
  else if constexpr (dim == 3) {
    if (axis == 0) {
        LINEAR_WEIGHTS<T, 3> l_u(p - VECTOR<T,3>(0.0, 0.5*dx, 0.5*dx), dx);
        g.get_gradient_from_Kernel(l_u, c);
        c = c / dx;
    } else if (axis == 1) {
        LINEAR_WEIGHTS<T, 3> l_v(p - VECTOR<T,3>(0.5*dx, 0.0, 0.5*dx), dx);
        g.get_gradient_from_Kernel(l_v, c);
        c = c / dx;
    } else if (axis == 2) {
        LINEAR_WEIGHTS<T, 3> l_w(p - VECTOR<T,3>(0.5*dx, 0.5*dx, 0.0), dx);
        g.get_gradient_from_Kernel(l_w, c);
        c = c / dx;
    }
  }
  return c;
}
}