#pragma once
#include <Math/VECTOR.h>
#include <Grid/SPARSE_GRID.h>
#include <FLIP/TRANSFER_FLIP.h>
#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <Math/LINEAR.h>
#include <FLIP/FRACTION_INSIDE.h>
#include <Math/CSR_MATRIX.h>
#include <Math/AMGCL_SOLVER.h>
#include <Eigen/SparseCore>
#include <math.h>

namespace py = pybind11;
namespace JGSL {

enum GRID_TYPE { SOLID, LIQUID, AIR };

template <class T, int dim>
void Create_Diff_Vel_Grid (
    SPARSE_GRID<T, dim>& prev,
    SPARSE_GRID<T, dim>& vel,
    SPARSE_GRID<T, dim>& diff)
{
    vel.Iterate_Grid([&](const auto& node, auto& g) {
        diff(node) = g - prev(node);
    });
}

template <class T, int dim>
void Add_External_Acc_to_P (
	FLIP_PARTICLES<T, dim>& particles,
	VECTOR<T, dim> &a, T dt)
{
    particles.Each([&](const int i, auto data) {
        auto& [X, V] = data;
        V += a * dt;
    });
}

template <class T, int dim>
void Add_External_Force_to_P (
	FLIP_PARTICLES<T, dim>& particles,
	VECTOR<T, dim> &F,
	T dt,
	T density)
{
    particles.Each([&](const int i, auto data) {
        auto& [X, V] = data;
        V += F / density  * dt;
    });
}

template <class T, int dim>
void Add_External_Force (
	SPARSE_GRID<int, dim>& type,
	SPARSE_GRID<T, dim>& vel_grid_X,
	SPARSE_GRID<T, dim>& vel_grid_Y,
	VECTOR<T, dim> &F,
	T dt,
	T density)
{
    type.Iterate_Grid([&](const auto& node, auto& g) {
        if (g == GRID_TYPE::LIQUID) {
            vel_grid_X(node) += F(0) / density * dt;
            vel_grid_Y(node) += F(1) / density * dt;
        }
    });
}

template <class T, int dim>
void Save_Velocity (
    SPARSE_GRID<T, dim>& besaved,
    SPARSE_GRID<T, dim>& save)
{
    save.Iterate_Grid([&](const auto& node, auto& g) {
        besaved(node) = g;
    });
}

template <class T, int dim>
void Save_Prev_Vel_Grid (
    SPARSE_GRID<T, dim>& prev,
    SPARSE_GRID<T, dim>& vel)
{
    vel.Iterate_Grid([&](const auto& node, auto& g) {
        prev(node) = g;
    });
}

template <class T, int dim>
void Add_External_Acc_Phi_(
	SPARSE_GRID<T, dim>& u,
	SPARSE_GRID<T, dim>& v,
	SPARSE_GRID<T, dim>& w,
	VECTOR<T, dim> lower,
	VECTOR<T, dim> upper,
    VECTOR<T, dim> a, T dt, T dx)
{
    TIMER_FLAG("Add_External_Acc_Phi_");

     VECTOR<int, dim> length;
     for (int d = 0; d < dim; d++) {
        length(d) = (int)std::round((upper(d) - lower(d)) / dx);
     }

    if constexpr (dim == 2) {
        u.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) <= length(0) && node(1) < length(1)) {
                u(node) += a(0) * dt;
            }
        });
        v.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) < length(0) && node(1) <= length(1)) {
                v(node) += a(1) * dt;
	        }
        });
    } else if constexpr (dim == 3) {
        u.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) <= length(0) && node(1) < length(1) && node(2) < length(2)) {
                u(node) += a(0) * dt;
            }
        });
        v.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) < length(0) && node(1) <= length(1) && node(2) < length(2)) {
                v(node) += a(1) * dt;
            }
        });
        w.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) < length(0) && node(1) < length(1) && node(2) <= length(2)) {
                w(node) += a(2) * dt;
            }
        });
    }
}

template <class T, int dim>
void Compute_Weight(
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    SPARSE_GRID<T, dim>& w_u,
    SPARSE_GRID<T, dim>& w_v,
    SPARSE_GRID<T, dim>& w_w,
    VECTOR<int, dim> &length,
    SPARSE_GRID<T, dim>& solid_phi)
{
    TIMER_FLAG("compute weights");
    if constexpr (dim == 2) {
        u.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) <= length(0) && node(1) < length(1)) {
                T w = (T)1 - Fraction_Inside(solid_phi(node(0),node(1)+1), solid_phi(node));
                w = w < 0 ? 0 : w;
                w = w > 1 ? 1 : w;
                w_u(node) = w;
            }
        });
        v.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) < length(0) && node(1) <= length(1)) {
                T w = (T)1 - Fraction_Inside(solid_phi(node(0)+1,node(1)), solid_phi(node));
                w = w < 0 ? 0 : w;
                w = w > 1 ? 1 : w;
                w_v(node) = w;
            }
        });
    }
    else if constexpr (dim == 3) {
        u.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) <= length(0) && node(1) < length(1) && node(2) < length(2)) {
                T w =  (T)1 - Fraction_Inside(solid_phi(node(0),node(1),node(2)),
                                              solid_phi(node(0),node(1)+1,node(2)),
                                              solid_phi(node(0),node(1),node(2)+1),
                                              solid_phi(node(0),node(1)+1,node(2)+1));
                w = w < 0 ? 0 : w;
                w = w > 1 ? 1 : w;
                w_u(node) = w;
            }
        });

        v.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) < length(0) && node(1) <= length(1) && node(2) < length(2)) {
                T w =  (T)1 - Fraction_Inside(solid_phi(node(0),node(1),node(2)),
                                              solid_phi(node(0),node(1),node(2)+1),
                                              solid_phi(node(0)+1,node(1),node(2)),
                                              solid_phi(node(0)+1,node(1),node(2)+1));
                w = w < 0 ? 0 : w;
                w = w > 1 ? 1 : w;
                w_v(node) = w;
            }
        });

        w.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) < length(0) && node(1) < length(1) && node(2) <= length(2)) {
                T w =  (T)1 - Fraction_Inside(solid_phi(node(0),node(1),node(2)),
                                              solid_phi(node(0),node(1)+1,node(2)),
                                              solid_phi(node(0)+1,node(1),node(2)),
                                              solid_phi(node(0)+1,node(1)+1,node(2)));
                w = w < 0 ? 0 : w;
                w = w > 1 ? 1 : w;
                w_w(node) = w;
            }
        });
    }
}

template <class T, int dim>
void Solve_Pressure_Phi (
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    SPARSE_GRID<int, dim>& u_valid,
    SPARSE_GRID<int, dim>& v_valid,
    SPARSE_GRID<int, dim>& w_valid,
    SPARSE_GRID<T, dim>& u_w,
    SPARSE_GRID<T, dim>& v_w,
    SPARSE_GRID<T, dim>& w_w,
    SPARSE_GRID<T, dim>& liquid_phi,
    T rho, T dx, T dt,
    VECTOR<int, dim> &length)
{
    TIMER_FLAG("Solve_Pressure");
    if constexpr (dim == 2) {
        auto p1 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto p2 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto& u_back = *p1;
        auto& v_back = *p2;
        Save_Velocity(u_back, u);
        Save_Velocity(v_back, v);
        auto p3 = std::make_unique<SPARSE_GRID<int, dim>>();
        auto& fluid_ind = *p3;

        int fluid_grids_num = 0;
        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0)<length(0)-1 && node(1)<length(1)-1) {
                if (liquid_phi(node)<0.0&&(u_w(node)>0.0||u_w(node(0)+1,node(1))>0.0||v_w(node)>0.0||v_w(node(0),node(1)+1)>0.0)) {
                    fluid_ind(node) = fluid_grids_num;
                    fluid_grids_num++;
                } else {
                    fluid_ind(node) = -1;
                }
		    } else {
                fluid_ind(node) = -1;
            }
        });
        printf("fluid grid num: %d\n", fluid_grids_num);

        if (fluid_grids_num == 0) { printf("error! there is no fluid grid\n"); return; }

        Eigen::SparseMatrix<T> A = Eigen::SparseMatrix<T>(fluid_grids_num, fluid_grids_num);
        A.setZero();
        Eigen::VectorXd b(fluid_grids_num);
        Eigen::VectorXd p(fluid_grids_num);
        Eigen::ConjugateGradient<Eigen::SparseMatrix<T>> my_solver;
        my_solver.setMaxIterations(150);

        T scale = dt;
        fluid_ind.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0)>=1 && node(0)<=length(0)-1 && node(1)>=1 && node(1)<=length(1)-1) {
                T center_phi = liquid_phi(node);
                if (center_phi < 0 && g >= 0) {
                    int idx = g;
                    A.insert(idx, idx) = (T)0;
                    b[idx] = 0; p[idx] = 0;

                    T right_phi = liquid_phi(node(0)+1, node(1));
                    T scale_t = scale * u_w(node(0)+1, node(1)) * (1.0 / (dx * dx));
                    if (right_phi < 0 && u_w(node(0)+1,node(1))>0) {
                       A.coeffRef(idx, idx) += scale_t;
                       A.insert(idx, fluid_ind(node(0)+1,node(1))) = -scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, right_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) += scale_t/theta;
                    }
                    b[idx] -= u_w(node(0)+1,node(1)) * u(node(0)+1,node(1)) / dx;

                    T left_phi = liquid_phi(node(0)-1, node(1));
                    scale_t = scale * u_w(node(0), node(1)) * (1.0 / (dx * dx));
                    if (left_phi < 0 && u_w(node)>0) {
                       A.coeffRef(idx, idx) += scale_t;
                       A.insert(idx, fluid_ind(node(0)-1,node(1))) = -scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, left_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) += scale_t/theta;
                    }
                    b[idx] += u_w(node(0),node(1)) * u(node(0),node(1)) / dx;

                    T top_phi = liquid_phi(node(0), node(1)+1);
                    scale_t = scale * v_w(node(0), node(1)+1) * (1.0 / (dx * dx));
                    if (top_phi < 0 && v_w(node(0),node(1)+1)>0) {
                       A.coeffRef(idx, idx) += scale_t;
                       A.insert(idx, fluid_ind(node(0),node(1)+1)) = -scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, top_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) += scale_t/theta;
                    }
                    b[idx] -= v_w(node(0),node(1)+1) * v(node(0),node(1)+1) / dx;

                    T bot_phi = liquid_phi(node(0), node(1)-1);
                    scale_t = scale * v_w(node(0), node(1)) * (1.0 / (dx * dx));
                    if (bot_phi < 0 && v_w(node)>0) {
                       A.coeffRef(idx, idx) += scale_t;
                       A.insert(idx, fluid_ind(node(0),node(1)-1)) = -scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, bot_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) += scale_t/theta;
                    }
                    b[idx] += v_w(node(0),node(1)) * v(node(0),node(1)) / dx;
                }
            }
        });

        my_solver.compute(A);
        p = my_solver.solve(b);

        u.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0)>=1 && node(1)>=0 && node(1)<length(1) && node(0)<=length(0)) {
                u_valid(node) = 0;
                if (u_w(node) > 0) {
                    if (liquid_phi(node) < 0 || liquid_phi(node(0)-1,node(1)) < 0) {
                        T theta = 1.0;
                        if (liquid_phi(node) >= 0 || liquid_phi(node(0)-1,node(1)) >= 0) {
                            theta = Fraction_Inside(liquid_phi(node(0)-1,node(1)),liquid_phi(node));
                        }
                        if (theta < 0.01) { theta = 0.01; }
                        T p_diff_u = p(fluid_ind(node)) - p(fluid_ind(node(0)-1,node(1)));
                        u(node) -= (dt) * p_diff_u / dx / theta;
                        u_valid(node) = 1;
                    }
                } else {
                    u(node) = 0;
                }
            }
        });

        v.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0)>=0 && node(1)>=1 && node(0)<length(0) && node(1)<=length(1)) {
                v_valid(node) = 0;
                if (v_w(node) > 0) {
                     if (liquid_phi(node) < 0 || liquid_phi(node(0),node(1)-1) < 0) {
                        T theta = 1.0;
                        if (liquid_phi(node) >= 0 || liquid_phi(node(0),node(1)-1) >= 0) {
                            theta = Fraction_Inside(liquid_phi(node(0),node(1)-1),liquid_phi(node));
                        }
                        if (theta < 0.01) { theta = 0.01; }
                        T p_diff_v = p(fluid_ind(node)) - p(fluid_ind(node(0),node(1)-1));
                        v(node) -=  (dt) * p_diff_v / dx / theta;
                        v_valid(node) = 1;
                    }
                } else {
                    v(node) = 0;
                }
            }
        });
	} else if constexpr (dim == 3) {
        auto p1 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto p2 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto p3 = std::make_unique<SPARSE_GRID<T, dim>>();
        auto& u_back = *p1;
        auto& v_back = *p2;
        auto& w_back = *p3;
        Save_Velocity(u_back, u);
        Save_Velocity(v_back, v);
        Save_Velocity(w_back, w);

        auto p4 = std::make_unique<SPARSE_GRID<int, dim>>();
        auto& fluid_ind = *p4;

        int fluid_grids_num = 0;
        liquid_phi.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0)<length(0)-1 && node(1)<length(1)-1 && node(2)<length(2)-1) {
                if (liquid_phi(node)<0.0&&(u_w(node)>0.0||u_w(node(0)+1,node(1),node(2))>0.0||v_w(node)>0.0||v_w(node(0),node(1)+1,node(2))>0.0||w_w(node)>0.0||w_w(node(0),node(1),node(2)+1)>0.0)) {
                    fluid_ind(node) = fluid_grids_num;
                    fluid_grids_num++;
                } else {
                    fluid_ind(node) = -1;
                }
		    } else {
                fluid_ind(node) = -1;
            }
        });
        printf("fluid grid num: %d\n", fluid_grids_num);

        if (fluid_grids_num == 0) { printf("error! there is no fluid grid\n"); return; }

        Eigen::SparseMatrix<T> A = Eigen::SparseMatrix<T>(fluid_grids_num, fluid_grids_num);
        Eigen::VectorXd b(fluid_grids_num);
        Eigen::VectorXd p(fluid_grids_num);
        Eigen::ConjugateGradient<Eigen::SparseMatrix<T>> my_solver;
        my_solver.setMaxIterations(150);
        T scale = dt;

        fluid_ind.Iterate_Grid([&](const auto& node, auto& g) {
            b[fluid_ind(node)] = 0;
            p[fluid_ind(node)] = 0;
            T center_phi = liquid_phi(node);
            if (node(0) >= 1 && node(0)<length(0) && node(1)>=1 && node(1)<length(1) && node(2)>=2 && node(2)<length(2)) {
                if (center_phi<0 && (u_w(node)>0.0 || u_w(node(0)+1,node(1),node(2))>0.0 || v_w(node)>0.0 || v_w(node(0),node(1)+1,node(2))>0.0|| w_w(node)>0.0 || w_w(node(0),node(1),node(2)+1)>0.0)) {
                    int idx = g;
                    A.insert(idx, idx) = 0;

                    T right_phi = liquid_phi(node(0)+1,node(1),node(2));
                    T scale_t = scale * u_w(node(0)+1,node(1),node(2)) * (1.0 / (dx * dx));
                    if (right_phi < 0 && u_w(node(0)+1,node(1),node(2)) > 0) {
                       A.coeffRef(idx, idx) -= scale_t;
                       A.insert(idx, fluid_ind(node(0)+1,node(1),node(2))) += scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, right_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) -= scale_t/theta;
                    }
                    b[idx] += u_w(node(0)+1,node(1),node(2)) * u(node(0)+1,node(1),node(2)) / dx;

                    T left_phi = liquid_phi(node(0)-1,node(1),node(2));
                    scale_t = scale * u_w(node(0),node(1),node(2)) * (1.0 / (dx * dx));
                    if (left_phi < 0 && u_w(node(0),node(1),node(2)) > 0) {
                       A.coeffRef(idx, idx) -= scale_t;
                       A.insert(idx, fluid_ind(node(0)-1,node(1),node(2))) += scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, left_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) -= scale_t/theta;
                    }
                    b[idx] -= u_w(node(0),node(1),node(2)) * u(node(0),node(1),node(2)) / dx;

                    T top_phi = liquid_phi(node(0),node(1)+1,node(2));
                    scale_t = scale * v_w(node(0),node(1)+1,node(2)) * (1.0 / (dx * dx));
                    if (top_phi < 0 && v_w(node(0),node(1)+1,node(2)) > 0) {
                       A.coeffRef(idx, idx) -= scale_t;
                       A.insert(idx, fluid_ind(node(0),node(1)+1,node(2))) += scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, top_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) -= scale_t/theta;
                    }
                    b[idx] += v_w(node(0),node(1)+1,node(2)) * v(node(0),node(1)+1,node(2)) / dx;

                    T bot_phi = liquid_phi(node(0),node(1)-1,node(2));
                    scale_t = scale * v_w(node(0),node(1),node(2)) * (1.0 / (dx * dx));
                    if (bot_phi < 0 && v_w(node(0),node(1),node(2)) > 0) {
                       A.coeffRef(idx, idx) -= scale_t;
                       A.insert(idx, fluid_ind(node(0),node(1)-1,node(2))) += scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, bot_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) -= scale_t/theta;
                    }
                    b[idx] -= v_w(node(0),node(1),node(2)) * v(node(0),node(1),node(2)) / dx;

                    T near_phi = liquid_phi(node(0),node(1),node(2)+1);
                    scale_t = scale * w_w(node(0),node(1),node(2)+1) * (1.0 / (dx * dx));
                    if (near_phi < 0 && w_w(node(0),node(1),node(2)+1) > 0) {
                       A.coeffRef(idx, idx) -= scale_t;
                       A.insert(idx, fluid_ind(node(0),node(1),node(2)+1)) += scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, near_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) -= scale_t/theta;
                    }
                    b[idx] += w_w(node(0),node(1),node(2)+1) * w(node(0),node(1),node(2)+1) / dx;

                    T far_phi = liquid_phi(node(0),node(1),node(2)-1);
                    scale_t = scale * w_w(node(0),node(1),node(2)) * (1.0 / (dx * dx));
                    if (far_phi < 0 && w_w(node(0),node(1),node(2)) > 0) {
                       A.coeffRef(idx, idx) -= scale_t;
                       A.insert(idx, fluid_ind(node(0),node(1),node(2)-1)) += scale_t;
                    } else {
                        T theta = Fraction_Inside(center_phi, far_phi);
                        if (theta < 0.01) { theta = 0.01; }
                        A.coeffRef(idx, idx) -= scale_t/theta;
                    }
                    b[idx] -= w_w(node(0),node(1),node(2)) * w(node(0),node(1),node(2)) / dx;
                }
            }
        });

        my_solver.compute(A);
        p = my_solver.solve(b);

        u.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(0) >= 1 && node(1) < length(1) && node(0) <= length(0) && node(2) < length(2)) {
                u_valid(node) = 0;
                if (u_w(node) > 0) {
                    if (liquid_phi(node) < 0 || liquid_phi(node(0)-1,node(1),node(2)) < 0) {
                        T theta = 1.0;
                        if (liquid_phi(node) >= 0 || liquid_phi(node(0)-1,node(1),node(2)) >= 0) {
                            theta = Fraction_Inside(liquid_phi(node),liquid_phi(node(0)-1,node(1),node(2)));
                        }
                        if (theta < 0.01) { theta = 0.01; }
                        T p_diff_u = p(fluid_ind(node)) - p(fluid_ind(node(0)-1,node(1),node(2)));
                        u(node) -= dt * p_diff_u / dx / theta;
                        u_valid(node) = 1;
                    }
                } else {
                    u(node) = 0;
                }
            }
        });

        v.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(1)>=1 && node(0)<length(0) && node(1)<=length(1) && node(2)<length(2)) {
                v_valid(node) = 0;
                if (v_w(node) > 0) {
                    if (liquid_phi(node) < 0 || liquid_phi(node(0),node(1)-1,node(2)) < 0) {
                        T theta = 1.0;
                        if (liquid_phi(node) >= 0 || liquid_phi(node(0),node(1)-1,node(2)) >= 0) {
                            theta = Fraction_Inside(liquid_phi(node(0),node(1)-1,node(2)),liquid_phi(node));
                        }
                        if (theta < 0.01) { theta = 0.01; }
                        T p_diff_v = p(fluid_ind(node)) - p(fluid_ind(node(0),node(1)-1,node(2)));
                        v(node) -=  dt * p_diff_v / dx / theta;
                        v_valid(node) = 1;
                    }
                } else {
                    v(node) = 0;
                }
            }
        });

        w.Iterate_Grid([&](const auto& node, auto& g) {
            if (node(2)>=1 && node(0)<length(0) && node(1)<length(1) && node(2)<=length(2)) {
                w_valid(node) = 0;
                if (w_w(node) > 0) {
                    if (liquid_phi(node) < 0 || liquid_phi(node(0),node(1),node(2)-1) < 0) {
                        T theta = 1.0;
                        if (liquid_phi(node) >= 0 || liquid_phi(node(0),node(1),node(2)-1) >= 0) {
                            theta = Fraction_Inside(liquid_phi(node(0),node(1),node(2)-1),liquid_phi(node));
                        }
                        if (theta < 0.01) { theta = 0.01; }
                        T p_diff_w = p(fluid_ind(node)) - p(fluid_ind(node(0),node(1),node(2)-1));
                        w(node) -=  dt * p_diff_w / dx / theta;
                        w_valid(node) = 1;
                    }
                } else {
                    w(node) = 0;
                }
            }
        });

        fluid_ind.Iterate_Grid([&](const auto& node, auto& g) {
             T center_phi = liquid_phi(node);
            if (center_phi<0 && (u_w(node)>0 || v_w(node)>0 || w_w(node)>0)) {
                int idx = g;
                T div = (T)0.0;
                div += (u_w(node(0)+1,node(1),node(2))*u(node(0)+1,node(1),node(2)) - u_w(node)*u(node)) / dx;
                div += (v_w(node(0),node(1)+1,node(2))*v(node(0),node(1)+1,node(2)) - v_w(node)*v(node)) / dx;
                div += (w_w(node(0),node(1),node(2)+1)*w(node(0),node(1),node(2)+1) - w_w(node)*w(node)) / dx;
                if (div > 0.00001 || div < -0.00001) {
                    //printf("fluid ind: %d %d %d, div: %f\n", node(0),node(1),node(2), div);
                }
            }
        });
	 }
}

template <class T, int dim>
void Projection (
    SPARSE_GRID<T, dim>& u,
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<T, dim>& w,
    SPARSE_GRID<int, dim>& u_valid,
    SPARSE_GRID<int, dim>& v_valid,
    SPARSE_GRID<int, dim>& w_valid,
    SPARSE_GRID<T, dim>& u_w,
    SPARSE_GRID<T, dim>& v_w,
    SPARSE_GRID<T, dim>& w_w,
    SPARSE_GRID<T, dim>& liquid_phi,
    SPARSE_GRID<T, dim>& solid_phi,
    T rho, T dx, T dt,
    VECTOR<T, dim> lower,
    VECTOR<T, dim> upper)
{
    VECTOR<int, dim> length;
    for (int d = 0; d < dim; d++) {
        length(d) = (int)std::round((upper(d) - lower(d)) / dx);
    }
    Compute_Weight(u, v, w, u_w, v_w, w_w, length, solid_phi);
    Solve_Pressure_Phi(u, v, w, u_valid, v_valid, w_valid, u_w, v_w, w_w, liquid_phi, rho, dx, dt, length);
}

template <class T, int dim>
void Extrapolate(
    SPARSE_GRID<T, dim>& v,
    SPARSE_GRID<int, dim>& valid,
    T dx,
    VECTOR<T, dim> lower,
    VECTOR<T, dim> upper,
    int inter_times)
{
    TIMER_FLAG("Extrapolate");
    auto p1 = std::make_unique<SPARSE_GRID<int, dim>>();
    auto& valid_back = *p1;
    Save_Velocity(valid_back, valid);

    VECTOR<int, dim> length;
	for (int d = 0; d < dim; d++) { length(d) = (int)std::round((upper(d) - lower(d)) / dx); }

    if constexpr (dim == 2) {
        for (int iter = 0; iter < inter_times; ++iter) {
            valid.Iterate_Grid([&](const VECTOR<int, dim>& node, auto& g) {
                if (node(0)>=1 && node(1)>=1 && node(0)<length(0)-1 && node(1)<length(1)-1) {
                    T sum = 0;
                    int count = 0;
                    if (valid_back(node) == 0) {
                        if (valid_back(node(0)+1,node(1))) { sum += v(node(0)+1,node(1)); count++; }
                        if (valid_back(node(0)-1,node(1))) { sum += v(node(0)-1,node(1)); count++; }
                        if (valid_back(node(0),node(1)+1)) { sum += v(node(0),node(1)+1); count++; }
                        if (valid_back(node(0),node(1)-1)) { sum += v(node(0),node(1)-1); count++; }
                        if (count > 0) { v(node) = sum / (T)count; valid(node) = 1; }
                    }
                }
            });
            Save_Velocity(valid_back, valid);
        }
    } else if constexpr (dim == 3) {
        for (int iter = 0; iter < inter_times; ++iter) {
            valid.Iterate_Grid([&](const VECTOR<int, dim>& node, auto& g) {
                if ((node(0)>=1)&&(node(1)>=1)&&(node(2)>=1)&&(node(0)<=length(0))&&(node(1)<=length(1))&&(node(2)<=length(2))) {
                    T sum = 0;
                    int count = 0;
                    if (valid_back(node) == 0) {
                        if (valid_back(node(0)+1,node(1),node(2))) { sum += v(node(0)+1,node(1),node(2)); count++; }
                        if (valid_back(node(0)-1,node(1),node(2))) { sum += v(node(0)-1,node(1),node(2)); count++; }
                        if (valid_back(node(0),node(1)+1,node(2))) { sum += v(node(0),node(1)+1,node(2)); count++; }
                        if (valid_back(node(0),node(1)-1,node(2))) { sum += v(node(0),node(1)-1,node(2)); count++; }
                        if (valid_back(node(0),node(1),node(2)+1)) { sum += v(node(0),node(1),node(2)+1); count++; }
                        if (valid_back(node(0),node(1),node(2)-1)) { sum += v(node(0),node(1),node(2)-1); count++; }
                        if (count > 0) { v(node) = sum / (T)count; valid(node) = 1; }
                    }
                }
            });
            Save_Velocity(valid_back, valid);
        }
    }
}


//#####################################################################
void Export_Flip_Pressure_Solve(py::module& m) {
    m.def("CreateDiffVelGrid",  &Create_Diff_Vel_Grid<double, 2>);
    m.def("CreateDiffVelGrid",  &Create_Diff_Vel_Grid<double, 3>);

    m.def("SavePrevVelGrid", &Save_Prev_Vel_Grid<double, 2>);
    m.def("SavePrevVelGrid", &Save_Prev_Vel_Grid<double, 3>);

    m.def("Extrapolate", &Extrapolate<double, 2>);
    m.def("Extrapolate", &Extrapolate<double, 3>);

    m.def("Projection", &Projection<double, 2>);
    m.def("Projection", &Projection<double, 3>);

    m.def("Add_Force", &Add_External_Acc_Phi_<double, 2>);
    m.def("Add_Force", &Add_External_Acc_Phi_<double, 3>);

    m.def("SolvePressure_phi", &Solve_Pressure_Phi<double, 2>);
    m.def("SolvePressure_phi", &Solve_Pressure_Phi<double, 3>);
}
}

