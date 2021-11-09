#pragma once
//#####################################################################
// Type BASE_STORAGE
// Type MPM_PARTICLES
// Type MPM_STRESS
// Function Create_MPM_Particle
// Function Create_MPM_Stress
// Function Particles_To_Grid
// Function Grid_To_Particles
// Class Multi_Object_Collision
// Function Kokkos_Initialize
// Function Kokkos_Finalize
//#####################################################################
#include <Grid/SPARSE_GRID.h>
#include <Math/BSPLINES.h>
#include <MPM/DATA_TYPE.h>
#include <MPM/COLOR.h>
#include <Utils/LOGGING.h>
#include <Utils/PROFILER.h>
#include <Utils/COLLIDER.h>
#ifdef ENABLE_MPM_IMPLICIT
#include <MPM/IMPLICIT_EULER.h>
#endif

namespace JGSL {

template <class T, int dim>
std::unique_ptr<MPM_PARTICLES<T, dim>> Create_MPM_Particle(
    const VECTOR<T, dim>& val
) {
    return std::make_unique<MPM_PARTICLES<T, dim>>();
}

template <class T, int dim>
std::unique_ptr<MPM_STRESS<T, dim>> Create_MPM_Stress(MPM_PARTICLES<T, dim>& particles) {
    TIMER_FLAG("Create Stress");
    auto stress_p = std::make_unique<MPM_STRESS<T, dim>>(particles.size);
    auto& stress = *stress_p;
    particles.Each([&](int i, auto) {
        stress.Insert(i, MATRIX<T, dim>(), VECTOR<T, dim>());
    });
    return stress_p;
}

template <class T, int dim>
void Clear_MPM_Stress(MPM_STRESS<T, dim>& stress) {
    TIMER_FLAG("Clear Stress");
    stress.Fill(MATRIX<T, dim>(), VECTOR<T, dim>());
}

template <class T, int dim>
std::unique_ptr<SPARSE_GRID<VECTOR<T, dim + 1>, dim>> Particles_To_Grid(
    MPM_PARTICLES<T, dim>& particles,
    MPM_STRESS<T, dim>& stress,
    const VECTOR<T, dim>& gravity,
    T dx,
    T dt,
    bool symplectic = true
) {
    TIMER_FLAG("Particles_To_Grid");
    auto grid_p = std::make_unique<SPARSE_GRID<VECTOR<T, dim + 1>, dim>>();
    auto& grid = *grid_p;
    {
    TIMER_FLAG("do full");
    Colored_Par_Each(particles, grid, dx, [&](const int i) {
        auto [X, V, _, C, m] = particles.Get_Unchecked(i);
        auto [ks, __] = stress.Get_Unchecked(i);
        BSPLINE_WEIGHTS<T, dim, 2> spline(X, dx);
        for (auto [node, w, dw, g] : grid.Iterate_Kernel(spline)) {
            VECTOR<T, dim> xi_minus_xp = node.template cast<T>() * dx - X;
            if (symplectic)
                g += VECTOR<T, dim + 1>(ks * dw * dt + V * m * w + C * m * xi_minus_xp * w, m * w);
            else
                g += VECTOR<T, dim + 1>(V * m * w + C * m * xi_minus_xp * w, m * w);
        }
    });
    }
    /*
    particles.Join(stress).Each([&](const int i, auto data) {
        auto& [X, V, _, C, m, ks, __] = data;
        BSPLINE_WEIGHTS<T, dim, 2> spline(X, dx);
        for (auto [node, w, dw, g] : grid.Iterate_Kernel(spline)) {
            VECTOR<T, dim> xi_minus_xp = node.template cast<T>() * dx - X;
            g += VECTOR<T, dim + 1>(ks * dw * dt + V * m * w + C * m * xi_minus_xp * w, m * w);
        }
    });
     */
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        if (g(dim)) {
            if (symplectic)
                g = VECTOR<T, dim + 1>(VECTOR<T, dim>(g) / g(dim) + dt * gravity, g(dim));
            else
                g = VECTOR<T, dim + 1>(VECTOR<T, dim>(g) / g(dim), g(dim));
        }
    });
    return grid_p;
}

template <class T, int dim>
void Multi_Object_Collision(
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    const COLLIDER<T, dim>& collider,
    T dx
) {
    TIMER_FLAG("Multi_Object_Collision");
    grid.Iterate_Grid([&](const VECTOR<int, dim>& node, VECTOR<T, dim + 1>& g) {
        VECTOR<T, dim> v(g);
        collider.Resolve(node.template cast<T>() * dx, v);
        g = VECTOR<T, dim + 1>(v, g(dim));
    });
}

template <class T, int dim>
void Multi_Object_Collision_With_Pressure(
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    const COLLIDER<T, dim>& collider,
    VECTOR<T, dim>& gravity,
    T center,
    T dt,
    T dx
) {
    TIMER_FLAG("Multi_Object_Collision_With_Pressure");
    SPARSE_GRID<VECTOR<T, dim + 1>, dim> grid_tmp = grid;
    grid.Iterate_Grid([&](const VECTOR<int, dim>& node, VECTOR<T, dim + 1>& g) {
        VECTOR<T, dim> v(g);
        bool near_air = false;
        Iterate_Region<T, dim>(-1, 1, [&](const VECTOR<int, dim>& delta) {
            VECTOR<int, dim> neighbor = node + delta;
            VECTOR<T, dim> position = neighbor.template cast<T>() * dx + VECTOR<T, dim>::Ones_Vector() * 0.5 * dx, tmp;
            if (grid_tmp(neighbor)(dim) == 0 && !collider.Resolve(position, tmp)) near_air = true;
        });
        if (near_air && std::abs((T)node(0) * dx - center) < dx)
            v += gravity * dt;
        collider.Resolve(node.template cast<T>() * dx, v);
        g = VECTOR<T, dim + 1>(v, g(dim));
    });
}

template <class T, int dim>
void Grid_To_Particles(
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    MPM_PARTICLES<T, dim>& particles,
    T dx,
    T dt
) {
    TIMER_FLAG("Grid_To_Particles");
    T D_inverse = (T)4 / dx / dx;
#ifdef STORAGE_ENABLED_OPENMP
    puts("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
#endif
    particles.Par_Each([&](const int i, auto data) {
        auto& [X, V, gradV, C, _] = data;
        BSPLINE_WEIGHTS<T, dim, 2> spline(X, dx);
        VECTOR<T, dim> picV;
        gradV = MATRIX<T, dim>();
        C = MATRIX<T, dim>();
        for (auto [node, w, dw, g] : grid.Iterate_Kernel(spline)) {
            VECTOR<T, dim> xi_minus_xp = node.template cast<T>() * dx - X;
            picV += VECTOR<T, dim>(g) * w;
            gradV += outer_product(VECTOR<T, dim>(g), dw);
            C += w * outer_product(VECTOR<T, dim>(g), xi_minus_xp);
        }
        V = picV;
        X += picV * dt;
        C *= D_inverse;
    });
}

void Kokkos_Initialize() {
    int argc = 0;
    char **argv = 0;
    Kokkos::initialize(argc, argv);
}

void Kokkos_Finalize() {
    Kokkos::finalize();
}

//#####################################################################
template <class T, int dim>
void Export_Transfer_Impl(py::module& m) {
    auto suffix = std::string("_") + (dim == 2 ? "2" : "3") + (std::is_same<T, float>::value ? "F" : "D");
    py::class_<MPM_PARTICLES<T, dim>>(m, ("MPM_PARTICLES" + suffix).c_str()).def(py::init<>());
    py::class_<MPM_STRESS<T, dim>>(m, ("MPM_STRESS" + suffix).c_str()).def(py::init<>());
    m.def("Create_MPM_Particles", &Create_MPM_Particle<T, dim>);
    m.def("Create_MPM_Stress", &Create_MPM_Stress<T, dim>);
    m.def("Clear_MPM_Stress", &Clear_MPM_Stress<T, dim>);
    m.def("Particles_To_Grid", &Particles_To_Grid<T, dim>);
    m.def("Update_Grid", &Particles_To_Grid<T, dim>);
    m.def("Multi_Object_Collision", &Multi_Object_Collision<T, dim>);
    m.def("Multi_Object_Collision_With_Pressure", &Multi_Object_Collision_With_Pressure<T, dim>);
    m.def("Grid_To_Particles", &Grid_To_Particles<T, dim>);
}

void Export_Transfer(py::module& m) {
    Export_Transfer_Impl<double, 2>(m);
    Export_Transfer_Impl<double, 3>(m);
#ifdef ENABLE_MPM_IMPLICIT
    Export_Implicit_Euler(m);
#endif
}

}