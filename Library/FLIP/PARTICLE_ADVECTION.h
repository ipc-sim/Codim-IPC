#pragma once
#include <Math/VECTOR.h>
#include <Grid/SPARSE_GRID.h>
#include <FLIP/TRANSFER_FLIP.h>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <Math/LINEAR.h>
#include <FLIP/CONSTRAIN_VELOCITY.h>

namespace py = pybind11;
namespace JGSL {

const int particle_correction_step = 1;

template<class T>
T smooth_kernel(const T& r2, const T& h) {
    return std::max(std::pow(1.0 - (r2 / (h * h)), 3.0), 0.0);
}

template<class T, int dim>
void    Correct(
    APIC_PARTICLES<T, dim>& particles,
    SPARSE_GRID<T, dim>& solid_phi,
    VECTOR<T, dim>& origin,
    T dx, T dt)
{
    TIMER_FLAG("Correct");
    int np = particles.data.size();
    T re = dx / std::sqrt(2.0);
    int offset = rand() % particle_correction_step;
    auto p = std::make_unique<SPARSE_GRID<Eigen::VectorXi, dim>>();
    auto& cell = *p;

    solid_phi.Iterate_Grid([&](const auto& node, auto& g) {
        cell(node) = Eigen::VectorXi(100);
        cell(node)(0) = 0;
    });

    particles.Each([&](const int i, auto data) {
        auto& [X, V, C] = data;
        VECTOR<T, dim> X_g = X - origin;
        LINEAR_WEIGHTS<T, dim> l(X_g, dx);
        int size = cell(l.base_node)(0);
        cell(l.base_node)(size+1) = i;
        cell(l.base_node)(0) = size+1;
    });

    particles.Each([&](const int i, auto data) {
        auto& [X, V, _] = data;
        if (i % particle_correction_step == offset) {
            VECTOR<T, dim> X_g;
            VECTOR<T, dim> spring;
            X_g = X - origin;
            LINEAR_WEIGHTS<T, dim> l(X_g, dx);
            int size = cell(l.base_node)(0);
            for (int a = 0; a < size; a++) {
                if (cell(l.base_node)(a+1) == i) { continue; }
                auto [Xp, V, _] = particles.Get_Unchecked(cell(l.base_node)(a+1));
                T dist = (X - Xp).norm();
                T w = 50.0 * smooth_kernel(dist*dist, re);
                if (dist > 0.01*re) {
                    spring += w * (X - Xp) / dist * re;
                } else {
                    spring(0) += 0.01 * re / dt * (rand() & 0xFF) / 255.0;
                    spring(1) += 0.01 * re / dt * (rand() & 0xFF) / 255.0;
                }
                VECTOR<T,dim> p = X + dt * spring;
                VECTOR<T,dim> pp = p - origin;
                T phi_value;
                solid_phi.get_from_Kernel(LINEAR_WEIGHTS<T, dim>(pp, dx), phi_value);
                if (phi_value < 0) {
                  VECTOR<T, dim> normal;
                  solid_phi.get_gradient_from_Kernel(LINEAR_WEIGHTS<T, dim>(pp, dx), normal);
                  normal = normal.Normalized();
                  p -= phi_value*normal;
                }
                X = p;
            }
        }
    });
}

template<class T, int dim>
T   CFL(
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    T dx)
{
    T max_v = 1e-6;
    u.Iterate_Grid([&](const auto& node, auto& g) {
        if (max_v < std::fabs(u(node))) {
            max_v = std::fabs(u(node));
        }
    });
    //printf("u max_v: %f\n", max_v);
    v.Iterate_Grid([&](const auto& node, auto& g) {
        if (max_v < std::fabs(v(node))) {
            max_v = std::fabs(v(node));
        }
    });
    //printf("v max_v: %f\n", max_v);
    if constexpr (dim == 3) {
        w.Iterate_Grid([&](const auto& node, auto& g) {
            if (max_v < std::fabs(w(node))) {
                max_v = std::fabs(w(node));
            }
        });
    }
    printf("cfl max_v: %f ", max_v);
    T dt = dx / max_v;
    if (dt < 0.0005) {
        dt = 0.0005;
    }
    printf("dt: %f\n", dt);
    return dt;
}

template <class T, int dim>
void Advect_P(
    FLIP_PARTICLES<T, dim>& particles,
    SPARSE_GRID<T, dim>& liquid_phi,
    SPARSE_GRID<T, dim>& solid_phi,
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    VECTOR<T, dim> origin,
    T dt, T dx)
{
    particles.Each([&](const int i, auto data) {
        auto& [X, V] = data;
        X += V * dt;
        LINEAR_WEIGHTS<T, dim> tmp(X-origin, dx);
        T phi_val;
        solid_phi.get_from_Kernel(tmp, phi_val);
        if (phi_val < 0) {//outside
           VECTOR<T, dim> grad;
           solid_phi.get_gradient_from_Kernel(tmp, grad);
           grad = grad.Normalized();
           //printf("advect inverse pos: %f,%f,%f, phi_val:%f, grad: %f,%f,%f ",X(0),X(1),X(2),phi_val,grad(0),grad(1),grad(2));
           X -= phi_val * grad;
           //printf("after pos: %f,%f,%f\n", X(0),X(1),X(2));
           //if (dim == 2) {
           //     printf("inverse after pos: %f,%f\n", X(0),X(1));
           //}
        } else {
           //printf("advect pos: %f,%f,%f, phi_val:%f\n",X(0),X(1),X(2),phi_val);
        }
    });
}

template <class T, int dim>
void Advect_Particles_Phi(
    FLIP_PARTICLES<T, dim>& particles,
    SPARSE_GRID<T, dim>& liquid_phi,
    SPARSE_GRID<T, dim>& solid_phi,
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    VECTOR<T, dim> origin,
    T dt, T dx)
{
    TIMER_FLAG("Advect_Particles");
    T t = (T)0.0;
    T substep = CFL(u, v, w, dx);
    while (t < dt) {
        if (t + substep > dt) {
            substep = dt - t;
        }
        Advect_P(particles, liquid_phi, solid_phi, u, v, w, origin, substep, dx);
        t += substep;
    }
    /*particles.Each([&](const int i, auto data) {
        auto& [X, V] = data;
        LINEAR_WEIGHTS<T, dim> tmp(X-origin, dx);
        if (dim == 2) {
           //printf("advect n:%d,%d, velocity: %f,%f\n",tmp.base_node(0),tmp.base_node(1),V(0),V(1));
        } else if (dim == 3) {
           //printf("advect n:%d,%d,%d, velocity: %f,%f,%f\n",tmp.base_node(0),tmp.base_node(1),tmp.base_node(2),V(0),V(1),V(2));
        }
    });*/
}

template <class T, int dim>
void Advect_P_A(
    APIC_PARTICLES<T, dim>& particles,
    SPARSE_GRID<T, dim>& liquid_phi,
    SPARSE_GRID<T, dim>& solid_phi,
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    VECTOR<T, dim> origin,
    T dt, T dx)
{
    particles.Each([&](const int i, auto data) {
        auto& [X, V, _] = data;
        X += V * dt;
        LINEAR_WEIGHTS<T, dim> tmp(X-origin, dx);
        T phi_val;
        solid_phi.get_from_Kernel(tmp, phi_val);
        if (phi_val < 0) {  //outside
           VECTOR<T, dim> grad;
           solid_phi.get_gradient_from_Kernel(tmp, grad);
           grad = grad.Normalized();
           X -= phi_val * grad;
           if (dim == 2 && X(0) < 0.12 && X(1) > 0.88) {
                printf("inverse after pos: %f,%f velocity: %f,%f\n", X(0),X(1),V(0),V(1));
           }
        } else {
           if (dim == 2 && X(0) < 0.12 && X(1) > 0.88) {
              printf("advect pos: %f,%f, phi_val:%f velocity: %f,%f\n",X(0),X(1),phi_val,V(0),V(1));
           } else if (dim == 3) {
              //printf("advect pos: %f,%f,%f, phi_val:%f\n",X(0),X(1),X(2),phi_val);
           }
        }
    });
}

template <class T, int dim>
void Advect_Particles_Phi_A(
    APIC_PARTICLES<T, dim>& particles,
    SPARSE_GRID<T, dim>& liquid_phi,
    SPARSE_GRID<T, dim>& solid_phi,
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    VECTOR<T, dim> origin,
    T dt, T dx)
{
    TIMER_FLAG("Advect_Particles");
    T t = (T)0.0;
    T substep = CFL(u, v, w, dx);
    while (t < dt) {
        if (t + substep > dt) {
            substep = dt - t;
        }
        Advect_P_A(particles, liquid_phi, solid_phi, u, v, w, origin, substep, dx);
        t += substep;
    }
}
//#####################################################################
void Export_particle_advect(py::module& m) {
    m.def("Advect_Particles_Phi", &Advect_Particles_Phi<double, 2>);
    m.def("Advect_Particles_Phi", &Advect_Particles_Phi<double, 3>);

    m.def("Advect_Particles_Phi", &Advect_Particles_Phi_A<double, 2>);
    m.def("Advect_Particles_Phi", &Advect_Particles_Phi_A<double, 3>);

    m.def("Correct", &Correct<double, 2>);
    m.def("Correct", &Correct<double, 3>);
}
}