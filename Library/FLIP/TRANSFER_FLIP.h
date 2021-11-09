#pragma once
//#####################################################################
// Type BASE_STORAGE
// Type FLIP_PARTICLES
// Type APIC_PARTICLES
// Function FLIP_PARTICLES
// Function APIC_PARTICLES
// Function Particles_To_Grid
// Function Grid_To_Particles
// Class Multi_Object_Collision
// Function Kokkos_Initialize
// Function Kokkos_Finalize
//#####################################################################
#include <Grid/SPARSE_GRID.h>
#include <Math/LINEAR.h>
#include <Utils/LOGGING.h>
#include <Utils/PROFILER.h>
#include <Storage/prelude.hpp>
#include <FLIP/APIC.h>

namespace JGSL {

template <class T, int dim>
using FLIP_PARTICLES = BASE_STORAGE<VECTOR<T, dim>, VECTOR<T, dim>>; // X V

template <class T, int dim>
using APIC_PARTICLES = BASE_STORAGE<VECTOR<T, dim>, VECTOR<T, dim>, MATRIX<T, dim>>; // X V C

template <class T, int dim>
std::unique_ptr<FLIP_PARTICLES<T, dim>> Create_FLIP_Particle(
    const VECTOR<T, dim>& val)
{
    return std::make_unique<FLIP_PARTICLES<T, dim>>();
}

template <class T, int dim>
std::unique_ptr<APIC_PARTICLES<T, dim>> Create_APIC_Particle(
    const VECTOR<T, dim>& val)
{
    return std::make_unique<APIC_PARTICLES<T, dim>>();
}

template <class T, int dim>
std::unique_ptr<SPARSE_GRID<T, dim>> Create_Vel_Grid(
    const VECTOR<T, dim>& val)
{
    return std::make_unique<SPARSE_GRID<T, dim>>();
}

template <class T, int dim>
void Particles_To_Grid_Flip_(
    FLIP_PARTICLES<T, dim>& particles,
    T dx, T dt,
    SPARSE_GRID<T, dim>& vel,
    SPARSE_GRID<T, dim>& solid_phi,
    VECTOR<T, dim> lower,
    VECTOR<T, dim> upper,
    int axis)
{
    TIMER_FLAG("Particles_To_Grid_Flip");
    auto p = std::make_unique<SPARSE_GRID<T, dim>>();
    auto& w_grid = *p;

    VECTOR<int, dim> length;
	for (int d = 0; d < dim; d++) { length(d) = (int)std::round((upper(d) - lower(d)) / dx); }

	solid_phi.Iterate_Grid([&](const auto& node, auto& g) {
	    vel(node) = (T)0;
	});

    particles.Each([&](const int i, auto data) {
        auto& [X, V] = data;
        VECTOR<T, dim> X_g;
        if constexpr (dim == 2) {
            if(axis == 0) {        // x
                X_g = X - lower - VECTOR<T, dim>(0.0, 0.5*dx);
            } else if (axis == 1) { // y
                X_g = X - lower - VECTOR<T, dim>(0.5*dx, 0,0);
            }
        } else if constexpr (dim == 3) {
            if (axis == 0) {            // x
                X_g = X - lower - VECTOR<T, dim>(0.0, 0.5*dx, 0.5*dx);
            } else if (axis == 1) {     // y
                X_g = X - lower - VECTOR<T, dim>(0.5*dx, 0.0, 0.5*dx);
            } else if (axis == 2) {     // z
                X_g = X - lower - VECTOR<T, dim>(0.5*dx, 0.5*dx, 0.0);
            }
        }
        LINEAR_WEIGHTS<T, dim> linear_w(X_g, dx);
        //if constexpr (dim == 2) {
            //printf("X: %f %f, base node: %i, %i, w: %f, %f\n", X_g(0), X_g(1), linear_w.base_node(0), linear_w.base_node(1), linear_w.w(0), linear_w.w(1));
        //}
        vel.Iterate_Kernel(linear_w, V(axis));
        w_grid.Iterate_Kernel(linear_w, 1.0);
       // printf("after iterate w base node have : %f, %f w: %f, %f\n", vel_x(linear_w_y.base_node), vel_y(linear_w_y.base_node), w_grid_x(linear_w_x.base_node), w_grid_y(linear_w_y.base_node));
    });

    vel.Iterate_Grid([&](const auto& node, auto& g) {
        if (w_grid(node) > 0.00000001 || w_grid(node) < -0.000000001) {
            //printf("P2G_x divide %i, %i, data: %f, w: %f\n", node(0),node(1),g, w_grid(node));
            g /= w_grid(node);
        }
    });
}

template <class T, int dim>
void Particles_To_Grid_Apic_(
    APIC_PARTICLES<T, dim>& particles,
    T dx, T dt,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& liquid_phi,
    VECTOR<T, dim> lower,
    VECTOR<T, dim> upper,
    int axis)
{
    TIMER_FLAG("Particles_To_Grid_Apic");
    auto p = std::make_unique<SPARSE_GRID<T, dim>>();
    auto& w_grid = *p;

    VECTOR<int, dim> length;
	for (int d = 0; d < dim; d++) { length(d) = (int)std::round((upper(d) - lower(d)) / dx); }

	liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
	    v(node) = (T)0;
	});

    particles.Each([&](const int i, auto data) {
        auto& [X, V, C] = data;
        VECTOR<T, dim> X_g;
        VECTOR<T, dim> pos;
        if constexpr (dim == 2) {
            if (axis == 0) {
                X_g = X - lower - VECTOR<T, dim>(0.0, 0.5*dx);
            } else if (axis == 1) {
                X_g = X - lower - VECTOR<T, dim>(0.5*dx, 0,0);
            }
        } else if constexpr (dim == 3) {
            if (axis == 0) {
                X_g = X - lower - VECTOR<T, dim>(0.0, 0.5*dx, 0.5*dx);
            } else if (axis == 1) {
                X_g = X - lower - VECTOR<T, dim>(0.5*dx, 0.0, 0.5*dx);
            } else if (axis == 2) {
                X_g = X - lower - VECTOR<T, dim>(0.5*dx, 0.5*dx, 0.0);
            }
        }
        LINEAR_WEIGHTS<T, dim> linear_w(X_g, dx);
        v.Iterate_Kernel_APIC(linear_w, C(axis), X-lower, axis, dx, V(axis));
        w_grid.Iterate_Kernel(linear_w, 1.0);
    });
    if (axis == 1) {
        printf("particle count: %d\n", particles.size);
    }
    v.Iterate_Grid([&](const auto& node, auto& g) {
        if (std::fabs(w_grid(node)) > 0.0) {
            g /= w_grid(node);
        }
    });
}

template <class T, int dim>
void Grid_To_Particles_Flip_(
    SPARSE_GRID<T, dim>& vel_grid,
    SPARSE_GRID<T, dim>& diff,
    FLIP_PARTICLES<T, dim>& particles,
    T dx,
    T dt,
    T pic_ratio,
    int axis
) {
    TIMER_FLAG("Grid_To_Particles_Flip");

    particles.Par_Each([&](const int i, auto data) {
        auto& [X, V] = data;
        VECTOR<T, dim> X_g;
        if (dim == 2) {
            if (axis == 0) {
                X_g = X - VECTOR<T, dim>(0, 0.5*dx);
            } else if (axis == 1) {
                X_g = X - VECTOR<T, dim>(0.5*dx, 0);
            }
        } else if (dim == 3) {
            if (axis == 0) {
                X_g = X - VECTOR<T, dim>(0, 0.5*dx, 0.5*dx);
            } else if (axis == 1) {
                X_g = X - VECTOR<T, dim>(0.5*dx, 0.0, 0.5*dx);
            } else if (axis == 2) {
                X_g = X - VECTOR<T, dim>(0.5*dx, 0.5*dx, 0);
            }
        }
        LINEAR_WEIGHTS<T, dim> linear_w(X_g, dx);
        T picV;
        vel_grid.get_from_Kernel(linear_w, picV);
        T diffV;
        diff.get_from_Kernel(linear_w, diffV);
        T flipV = diffV + V(axis);
        V(axis) = pic_ratio * picV + (1 - pic_ratio) * flipV;
        //if (dim == 3 && axis == 1) {
            //printf("result: %f %f %f \n", V(0), V(1), V(2));
        //}
    });
}

template <class T, int dim>
void Grid_To_Particles_Apic_(
    SPARSE_GRID<T, dim>& vel_grid,
    SPARSE_GRID<T, dim>& diff,
    APIC_PARTICLES<T, dim>& particles,
    T dx, T dt, T pic_ratio,
    int axis
) {
    TIMER_FLAG("Grid_To_Particles_Apic");
    particles.Par_Each([&](const int i, auto data) {
        auto& [X, V, C] = data;
        VECTOR<T, dim> X_g;
        if (dim == 2) {
            if (axis == 0) {
                X_g = X - VECTOR<T, dim>(0.0, 0.5*dx);
            } else if (axis == 1) {
                X_g = X - VECTOR<T, dim>(0.5*dx, 0.0);
            }
        } else if (dim == 3) {
            if (axis == 0) {
                X_g = X - VECTOR<T, dim>(0, 0.5*dx, 0.5*dx);
            } else if (axis == 1) {
                X_g = X - VECTOR<T, dim>(0.5*dx, 0.0, 0.5*dx);
            } else if (axis == 2) {
                X_g = X - VECTOR<T, dim>(0.5*dx, 0.5*dx, 0);
            }
        }
        LINEAR_WEIGHTS<T, dim> linear_w(X_g, dx);
        T picV;
        vel_grid.get_from_Kernel(linear_w, picV);
        T diffV;
        diff.get_from_Kernel(linear_w, diffV);
        T flipV = diffV + V(axis);
        V(axis) = pic_ratio * picV + (1 - pic_ratio) * flipV;
        C(axis) = Get_Affine_Matrix(vel_grid, X, VECTOR<T,dim>(0), axis, dx);
//      if (dim == 2 && axis==1)
//          printf("c(%d) : %f,%f\n", axis, C(axis)(0), C(axis)(1));
    });
}

void Kokkos_Initialize_F() {
    int argc = 0;
    char **argv = 0;
    Kokkos::initialize(argc, argv);
}

void Kokkos_Finalize_F() {
    Kokkos::finalize();
}

//#####################################################################
template <class T, int dim>
void Export_Transfer_Impl_FLIP(py::module& m) {
    auto suffix = std::string("_") + (dim == 2 ? "2" : "3") + (std::is_same<T, float>::value ? "F" : "D");
    py::class_<FLIP_PARTICLES<T, dim>>(m, ("FLIP_PARTICLES" + suffix).c_str()).def(py::init<>());
    py::class_<APIC_PARTICLES<T, dim>>(m, ("APIC_PARTICLES" + suffix).c_str()).def(py::init<>());

    m.def("Create_FLIP_Particles", &Create_FLIP_Particle<T, dim>);
    m.def("Create_APIC_Particles", &Create_APIC_Particle<T, dim>);

    m.def("Particles_To_Grid_Flip_S", &Particles_To_Grid_Flip_<T, dim>);
    m.def("Grid_To_Particles_Flip_S", &Grid_To_Particles_Flip_<T, dim>);

    m.def("Grid_To_Particles_Apic_S", &Grid_To_Particles_Apic_<T, dim>);
    m.def("Particles_To_Grid_Apic_S", &Particles_To_Grid_Apic_<T, dim>);

    m.def("Create_Vel_Grid", &Create_Vel_Grid<T, dim>);
}

void Export_Transfer_FLIP(py::module& m) {
    Export_Transfer_Impl_FLIP<double, 2>(m);
    Export_Transfer_Impl_FLIP<double, 3>(m);

    m.def("Create_Vel_Grid", &Create_Vel_Grid<int, 2>);
    m.def("Create_Vel_Grid", &Create_Vel_Grid<int, 3>);
}

}