#pragma once
#include <Math/VECTOR.h>
#include <Grid/SPARSE_GRID.h>
#include <FLIP/TRANSFER_FLIP.h>
#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <Math/LINEAR.h>
#include <FLIP/FRACTION_INSIDE.h>
#include <FLIP/BOUNDARY_PHI.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim>
std::unique_ptr<SPARSE_GRID<T, dim>> Compute_Phi(
    FLIP_PARTICLES<T, dim>& particles,
    SPARSE_GRID<T, dim>& solid_phi,
    T dx,
    const VECTOR<T, dim> origin,
    const VECTOR<T, dim> upper,
    T particle_radius)
{
    auto l = std::make_unique<SPARSE_GRID<T, dim>>();
    auto& liquid_phi = *l;

    VECTOR<int, dim> length;
	for (int d = 0; d < dim; d++) { length(d) = (int)std::round((upper(d) - origin(d)) / dx); }

	solid_phi.Iterate_Grid([&](const auto& node, auto& g) {
          liquid_phi(node) = 3*dx;
    });

    if constexpr (dim == 2) {
         particles.Each([&](const int i, auto data) {
            auto& [X, V] = data;
            //printf("particle position: %f,%f\n", X(0), X(1));
            LINEAR_WEIGHTS<T, dim> w(X-origin, dx);
            const VECTOR<int, dim>& c = w.base_node;
            for (int i = c(0)-2; i <= c(0)+2; i++) {
                for (int j = c(1)-2; j <= c(1)+2; j++) {
                    if (i < 0 || j < 0 || i >= length(0) || j >= length(1) ) { continue; }
                    VECTOR<T,dim> n_center = VECTOR<T,dim>((i+0.5)*dx, (j+0.5)*dx) + origin;
                    T phi_t = (X-n_center).norm() - particle_radius;
                    if (liquid_phi(i,j) > phi_t) {
                        liquid_phi(i,j) = phi_t;
                    }
                }
            }
        });

        auto p1 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto& phi_temp = *p1;
        Save_Velocity(phi_temp, liquid_phi);
        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            if (liquid_phi(node) < 0.5 * dx) {
                T solid_phi_val = (T)0.25 * (solid_phi(node)+solid_phi(node(0)+1,node(1))+
                                             solid_phi(node(0),node(1)+1)+solid_phi(node(0)+1,node(1)+1));
                if (solid_phi_val < 0) {
                    phi_temp(node) = (T)-0.5*dx;
                }
            }
        });
        Save_Velocity(liquid_phi, phi_temp);

        /*liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            VECTOR<T, dim> pos = VECTOR<T, dim>((node(0)+0.5)*dx, (node(1)+0.5)*dx) + origin;
            T solid_phi_val = Boundary_Phi(pos);
            liquid_phi(node) = std::min(liquid_phi(node), solid_phi_val);
        });*/

        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            //printf("compute liquid phi node: %d,%d v: %f\n", node(0),node(1),g);
        });
    }
    else if constexpr (dim == 3) {
        particles.Each([&](const int i, auto data) {
            auto& [X, V] = data;
            LINEAR_WEIGHTS<T, dim> w(X-origin, dx);
            //printf("particle position: %f, %f, %f, node : %d, %d, %d\n",X(0),X(1),X(2),w.base_node(0),w.base_node(1),w.base_node(2));
            const VECTOR<int, dim>& c = w.base_node;
            for (int i = c(0)-2; i <= c(0)+2; i++) {
                for (int j = c(1)-2; j <= c(1)+2; j++) {
                    for (int k = c(2)-2; k <= c(2)+2; k++) {
                        if (i < 0 || j < 0 || k < 0 || i > length(0) || j > length(1) || k > length(2)) { continue; }
                        VECTOR<T,dim> n_center = VECTOR<T,dim>((i+0.5)*dx,(j+0.5)*dx,(k+0.5)*dx) + origin;
                        T phi_t = (X-n_center).norm() - particle_radius;
                        if (liquid_phi(i,j,k) > phi_t) {
                           liquid_phi(i,j,k) = phi_t;
                           // if (liquid_phi(i,j,k) < 0) {
                           // printf("idx：%d,%d,%d, X: %f,%f,%f, C: %f,%f,%f, l: %f, r: %f, v：%f\n", i,j,k,X(0),X(1),X(2),n_center(0),n_center(1),n_center(2),(X - n_center).norm(),particle_radius,phi_t);
                           // }
                        }
                    }
                }
            }
        });

        auto p1 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto& phi_temp = *p1;
        Save_Velocity(phi_temp, liquid_phi);
        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            if (liquid_phi(node) < 0.5 * dx) {
                T solid_phi_val = (T)0.125 * (solid_phi(node(0),node(1),node(2))+solid_phi(node(0)+1,node(1),node(2))+
                                              solid_phi(node(0),node(1)+1,node(2))+solid_phi(node(0)+1,node(1)+1,node(2))+
                                              solid_phi(node(0),node(1),node(2)+1)+solid_phi(node(0)+1,node(1),node(2)+1)+
                                              solid_phi(node(0),node(1)+1,node(2)+1)+solid_phi(node(0)+1,node(1)+1,node(2)+1));
                if (solid_phi_val < 0) {
                    phi_temp(node) = (T)-0.5*dx;
                }
            }
        });
        Save_Velocity(liquid_phi, phi_temp);
    }
    return l;
}

template <class T, int dim>
std::unique_ptr<SPARSE_GRID<T, dim>> Compute_Phi_A(
    APIC_PARTICLES<T, dim>& particles,
    SPARSE_GRID<T, dim>& solid_phi,
    T dx,
    const VECTOR<T, dim> origin,
    const VECTOR<T, dim> upper,
    T particle_radius)
{
    auto l = std::make_unique<SPARSE_GRID<T, dim>>();
    auto& liquid_phi = *l;

    VECTOR<int, dim> length;
	for (int d = 0; d < dim; d++) {
	    length(d) = (int)std::round((upper(d) - origin(d)) / dx);
	}

	solid_phi.Iterate_Grid([&](const auto& node, auto& g) {
          liquid_phi(node) = 3*dx;
    });

    if constexpr (dim == 2) {
         particles.Each([&](const int i, auto data) {
            auto& [X, V, _] = data;
            LINEAR_WEIGHTS<T, dim> w(X-origin, dx);
            //printf("compute liquid position: id: %d, %f,%f\n", i, X(0),X(1));
            const VECTOR<int, dim>& c = w.base_node;
            for (int i = c(0)-2; i <= c(0)+2; i++) {
                for (int j = c(1)-2; j <= c(1)+2; j++) {
                    if (i < 0 || j < 0 || i >= length(0) || j >= length(1) ) { continue; }
                    VECTOR<T,dim> n_center = VECTOR<T,dim>((i+0.5)*dx, (j+0.5)*dx) + origin;
                    T phi_t = (X-n_center).norm() - std::max((dx / std::sqrt(2)), particle_radius);
                    if (liquid_phi(i,j) > phi_t) {
                        liquid_phi(i,j) = phi_t;
                    }
                }
            }
        });

        /*auto p1 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto& phi_temp = *p1;
        Save_Velocity(phi_temp, liquid_phi);
        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            if (liquid_phi(node) < 0.5 * dx) {
                T solid_phi_val = (T)0.25 * (solid_phi(node)+solid_phi(node(0)+1,node(1))+
                                             solid_phi(node(0),node(1)+1)+solid_phi(node(0)+1,node(1)+1));
                if (solid_phi_val < 0) {
                    phi_temp(node) = (T)-0.5*dx;
                    printf("solid add node: %d,%d\n", node(0),node(1));
                }
            }
        });
        Save_Velocity(liquid_phi, phi_temp);*/

        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) < length(0) && node(0) >= 0 && node(1) < length(1) && node(1) >= 0) {
                VECTOR<T,2> pos = VECTOR<T,2>((node(0)+0.5)*dx,(node(1)+0.5)*dx) + origin;
                T solid_val = Boundary_Phi(pos);
                if (liquid_phi(node) > solid_val) {
                   liquid_phi(node) = solid_val;
                   //printf("solid add node: %d, %d\n", node(0), node(1));
                }
            }
        });
    }
    else if constexpr (dim == 3) {
        particles.Each([&](const int i, auto data) {
            auto& [X, V, _] = data;
            LINEAR_WEIGHTS<T, dim> w(X-origin, dx);
            //printf("particle position: %f, %f, %f, node : %d, %d, %d\n",X(0),X(1),X(2),w.base_node(0),w.base_node(1),w.base_node(2));
            const VECTOR<int, dim>& c = w.base_node;
            for (int i = c(0)-2; i <= c(0)+2; i++) {
                for (int j = c(1)-2; j <= c(1)+2; j++) {
                    for (int k = c(2)-2; k <= c(2)+2; k++) {
                        if (i < 0 || j < 0 || k < 0 || i > length(0) || j > length(1) || k > length(2)) { continue; }
                        VECTOR<T,dim> n_center = VECTOR<T,dim>((i+0.5)*dx,(j+0.5)*dx,(k+0.5)*dx) + origin;
                        T phi_t = (X-n_center).norm() - std::max((dx / std::sqrt(2)), particle_radius);
                        if (liquid_phi(i,j,k) > phi_t) {
                           liquid_phi(i,j,k) = phi_t;
                        }
                    }
                }
            }
        });
        auto p1 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto& phi_temp = *p1;
        Save_Velocity(phi_temp, liquid_phi);
        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            if (liquid_phi(node) < 0.5 * dx) {
                T solid_phi_val = (T)0.125 * (solid_phi(node(0),node(1),node(2))+solid_phi(node(0)+1,node(1),node(2))+
                                              solid_phi(node(0),node(1)+1,node(2))+solid_phi(node(0)+1,node(1)+1,node(2))+
                                              solid_phi(node(0),node(1),node(2)+1)+solid_phi(node(0)+1,node(1),node(2)+1)+
                                              solid_phi(node(0),node(1)+1,node(2)+1)+solid_phi(node(0)+1,node(1)+1,node(2)+1));
                if (solid_phi_val < 0) {
                    phi_temp(node) = (T)-0.5*dx;
                }
            }
        });
        Save_Velocity(liquid_phi, phi_temp);
        /*
        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0)<length(0) && node(1)<length(1) && node(2)<length(2)) {
                VECTOR<T,3> pos = VECTOR<T,3>((node(0)+0.5)*dx,(node(1)+0.5)*dx,(node(2)+0.5)*dx) + origin;
                T solid_val = Boundary_Phi(pos);
                if (liquid_phi(node) > solid_val) {
                   liquid_phi(node) = solid_val;
                }
            }
        });*/
    }
    return l;
}

template <class T, int dim>
T Boundary_Phi(VECTOR<T,dim> p) {
    if constexpr (dim == 2) {
        VECTOR<T, 2> center(0.5,0.5);
        VECTOR<T, 2> b(0.4,0.4);
        T d1 = Box_Phi(p, center, b);
        return -d1;
    } else if constexpr (dim == 3) {
        VECTOR<T, 3> center(0.5,0.5,0.5);
        VECTOR<T, 3> b(0.25,0.25,0.25);
        T d1 = Box_Phi(p, center, b);
        return -d1;
    }
}

template <class T, int dim>
std::unique_ptr<SPARSE_GRID<T, dim>>Set_Boundary (
    T dx,
    const VECTOR<T, dim>& origin,
    const VECTOR<T, dim>& upper)
{
    TIMER_FLAG("Set_Boundary");
    auto grid_p = std::make_unique<SPARSE_GRID<T, dim>>();
    auto& solid_phi = *grid_p;

    VECTOR<int, dim> length;
	for (int d = 0; d < dim; d++) {
	    length(d) = (int)std::round((upper(d) - origin(d)) / dx);
	}

    if constexpr (dim == 2) {
        for (int i = 0; i <= length(0); i++) {
            for (int j = 0; j <= length(1); j++) {
                VECTOR<T, 2> p = VECTOR<T, 2>(i*dx,j*dx) + origin;
                 solid_phi(i, j) = Boundary_Phi(p);
            }
        }
    } else if constexpr (dim == 3) {
        for (int i = 0; i <= length(0); i++) {
            for (int j = 0; j <= length(1); j++) {
                for (int k = 0; k <= length(2); k++) {
                    VECTOR<T, 3> p = VECTOR<T, 3>(i*dx,j*dx,k*dx) + origin;
                    solid_phi(i, j, k) = Boundary_Phi(p);
                }
            }
        }
    }
    return grid_p;
}

template <class T, int dim>
std::unique_ptr<SPARSE_GRID<T, dim>>Set_Ob_Boundary (
    T dx,
    const VECTOR<T, dim>& lower,
    const VECTOR<T, dim>& upper)
{
    TIMER_FLAG("Set_obstacle");
    auto grid_p = std::make_unique<SPARSE_GRID<T, dim>>();
    auto& obstacle_phi = *grid_p;

    VECTOR<int, dim> length;
	for (int d = 0; d < dim; d++) {
	    length(d) = (int)std::round((upper(d) - lower(d)) / dx);
	}

    if constexpr (dim == 3) {
        T r = 0.2;
        VECTOR<T, dim> center(0.5, 0.5, 0.5);
        for (int i = 0; i <= length(0); i++) {
            for (int j = 0; j <= length(1); j++) {
                for (int k = 0; k <= length(2); k++) {
                    VECTOR<T, 3> p = VECTOR<T, 3>(i*dx,j*dx,k*dx) + lower;
                    obstacle_phi(i,j,k) = Sphere_Phi(p,center,r);
                }
            }
        }
    }
    return grid_p;
}


//#####################################################################
void Export_Flip_Marker(py::module& m) {
    m.def("Compute_Phi", &Compute_Phi<double, 2>);
    m.def("Compute_Phi", &Compute_Phi<double, 3>);

    m.def("Compute_Phi", &Compute_Phi_A<double, 2>);
    m.def("Compute_Phi", &Compute_Phi_A<double, 3>);

    m.def("Set_Boundary", &Set_Boundary<double, 2>);
    m.def("Set_Boundary", &Set_Boundary<double, 3>);

    m.def("Set_Ob_Boundary", &Set_Ob_Boundary<double, 2>);
    m.def("Set_Ob_Boundary", &Set_Ob_Boundary<double, 3>);
}
}