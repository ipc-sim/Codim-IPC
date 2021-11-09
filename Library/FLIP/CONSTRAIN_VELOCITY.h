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
VECTOR<T, dim> Get_Velocity(
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    const VECTOR<T, dim> p,
    const VECTOR<T, dim> origin,
    T dx)
{
    VECTOR<T, dim> vel;
    if constexpr (dim == 2) {
        LINEAR_WEIGHTS<T,2> l_u(p - VECTOR<T,2>(0.0, 0.5*dx) - origin, dx);
        LINEAR_WEIGHTS<T,2> l_v(p - VECTOR<T,2>(0.5*dx, 0.0) - origin, dx);
        u.get_from_Kernel(l_u, vel(0));
        v.get_from_Kernel(l_v, vel(1));
    } else if constexpr (dim == 3) {
        LINEAR_WEIGHTS<T,3> l_u(p - VECTOR<T,3>(0.0, 0.5*dx, 0.5*dx) - origin, dx);
        LINEAR_WEIGHTS<T,3> l_v(p - VECTOR<T,3>(0.5*dx, 0.0, 0.5*dx) - origin, dx);
        LINEAR_WEIGHTS<T,3> l_w(p - VECTOR<T,3>(0.5*dx, 0.5*dx, 0.0) - origin, dx);
        u.get_from_Kernel(l_u, vel(0));
        v.get_from_Kernel(l_v, vel(1));
        w.get_from_Kernel(l_w, vel(2));
    }
    return vel;
}

template <class T, int dim>
void Constrain_Velocity(
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    SPARSE_GRID<T, dim>& w_u,
    SPARSE_GRID<T, dim>& w_v,
    SPARSE_GRID<T, dim>& w_w,
    SPARSE_GRID<T, dim>& solidPhi,
    VECTOR<T, dim> origin,
    VECTOR<T, dim> upper,
    T dx)
{
    TIMER_FLAG("Constrain_Velocity");
    auto p1 = std::make_unique<SPARSE_GRID<T, dim>>();
    auto p2 = std::make_unique<SPARSE_GRID<T, dim>>();
    auto p3 = std::make_unique<SPARSE_GRID<T, dim>>();
    auto& tempU = *p1;
    auto& tempV = *p2;
    auto& tempW = *p3;
    Save_Velocity(tempU, u);
    Save_Velocity(tempV, v);
    Save_Velocity(tempW, w);

    VECTOR<int, dim> length;
	for (int d = 0; d < dim; d++) { length(d) = (int)std::round((upper(d) - origin(d)) / dx); }

    w_u.Iterate_Grid([&](const auto& node, auto& g) {
        if (node(0) <= length(0) && node(1) < length(1)) {
            if (w_u(node) == 0.0) {
                VECTOR<T, dim> p;
                if constexpr (dim == 2) {
                    p = VECTOR<T,2>(node(0)*dx,(node(1)+0.5)*dx) + origin;
                } else if constexpr (dim == 3) {
                    p = VECTOR<T,3>(node(0)*dx,(node(1)+0.5)*dx,(node(2)+0.5)*dx) + origin;
                }
                VECTOR<T, dim> vel = Get_Velocity(u, v, w, p, origin, dx);
                VECTOR<T, dim> normal;
                if constexpr (dim == 2) {
                    LINEAR_WEIGHTS<T,2> l_u(p,dx);
                    solidPhi.get_gradient_from_Kernel(l_u, normal);
                } else if constexpr (dim == 3) {
                    LINEAR_WEIGHTS<T,3> l_u(p,dx);
                    solidPhi.get_gradient_from_Kernel(l_u, normal);
                }
                normal = normal.Normalized();
                T perp_component = vel.dot(normal);
                vel -= perp_component * normal;
                tempU(node) = vel(0);
            }
        }
    });
    Save_Velocity(u, tempU);

    w_v.Iterate_Grid([&](const auto& node, auto& g) {
        if (node(0)<length(0) && node(1)<length(1) && node(2)<=length(2)) {
            if (w_v(node) == 0.0) {
                VECTOR<T, dim> p;
                if constexpr (dim == 2) {
                    p = VECTOR<T,2>((node(0)+0.5)*dx,node(1)*dx) + origin;
                } else if constexpr (dim == 3) {
                    p = VECTOR<T,3>((node(0)+0.5)*dx,node(1)*dx,(node(2)+0.5)*dx) + origin;
                }
                VECTOR<T, dim> vel = Get_Velocity(u, v, w, p, origin, dx);
                VECTOR<T, dim> normal;
                if constexpr (dim == 2) {
                    LINEAR_WEIGHTS<T,2> l_v(p, dx);
                    solidPhi.get_gradient_from_Kernel(l_v, normal);
                } else if constexpr (dim == 3) {
                    LINEAR_WEIGHTS<T,dim> l_v(p, dx);
                    solidPhi.get_gradient_from_Kernel(l_v, normal);
                }
                normal = normal.Normalized();
                T perp_component = vel.dot(normal);
                vel -= perp_component * normal;
                tempV(node) = vel(1);
            }
        }
    });
    Save_Velocity(v, tempV);

    if constexpr (dim == 3) {
         w_w.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0)<length(0) && node(1)<length(1) && node(2)<=length(2)) {
                if (w_w(node) == 0.0) {
                    VECTOR<T, dim> p = VECTOR<T, dim>((node(0)+0.5)*dx,(node(1)+0.5)*dx,node(2)*dx) + origin;
                    VECTOR<T, dim> vel = Get_Velocity(u, v, w, p, origin, dx);
                    VECTOR<T, dim> normal;
                    LINEAR_WEIGHTS<T,dim> l_w(p,dx);
                    solidPhi.get_gradient_from_Kernel(l_w, normal);
                    normal = normal.Normalized();
                    T perp_component = vel.dot(normal);
                    vel -= perp_component * normal;
                    tempW(node) = vel(2);
                }
            }
        });
        Save_Velocity(w, tempW);
    }
}


//#####################################################################
void Export_Flip_Constrain_Velocity(py::module& m) {
    m.def("Constrain_Velocity", &Constrain_Velocity<double, 2>);
    m.def("Constrain_Velocity", &Constrain_Velocity<double, 3>);
}
}