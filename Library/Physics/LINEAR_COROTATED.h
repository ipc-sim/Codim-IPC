#pragma once

#include <Particles/PARTICLES.h>
#include <Math/VECTOR.h>
#include <Math/SINGULAR_VALUE_DECOMPOSITION.h>
#include <Physics/FIXED_COROTATED.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim>
class LINEAR_COROTATED_FUNCTOR {
public:
    using STORAGE = typename FIXED_COROTATED_FUNCTOR<T, dim>::STORAGE;
    using DIFFERENTIAL = BASE_STORAGE<Eigen::Matrix<T, dim*dim, dim*dim>>;
    static const bool useJ = false;
    static const bool projectable = false;

    static std::unique_ptr<STORAGE> Create() {
        return std::make_unique<STORAGE>();
    }

    static void Append(STORAGE& lcr, const VECTOR<int, 2>& handle, T vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        for (int i = handle(0); i < handle(1); ++i)
            lcr.Insert(i, MATRIX<T, dim>(1), vol, lambda, mu, T());
    }

    static void Compute_Kirchoff_Stress(STORAGE& lcr, MPM_STRESS<T, dim>& stress) {
        lcr.Join(stress).Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu, _, ks, __] = data;
            MATRIX<T, dim> U(1), V(1);
            VECTOR<T, dim> sigma;
            Singular_Value_Decomposition(F, U, sigma, V);
            MATRIX<T, dim> R = U * V.transpose();
            MATRIX<T, dim> RT_mulF_minusI = (R.transpose() * F) - MATRIX<T, dim>(1);
            MATRIX<T, dim> first_piola = (T)2 * mu * (F - R) + lambda * RT_mulF_minusI * R;
            ks += -vol * first_piola * F.transpose();
        });
    }

    static void Compute_Psi(STORAGE& lcr, T w, T& Psi) {
        //TODO
        Psi = 0;
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& lcr, T w, BASE_STORAGE<MATRIX<T, dim>>& P) {
        //TODO
    }

    static void Compute_First_PiolaKirchoff_Stress_Derivative(STORAGE& lcr, T w, DIFFERENTIAL& dP_div_dF) {
        //TOD
    }
};

}
