#pragma once
//#####################################################################
// Function Compute_Kirchoff_Stress_NHB
//#####################################################################
#include <Math/VECTOR.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim>
using NEOHOOKEAN_BORDEN = BASE_STORAGE<MATRIX<T, dim>, T, T, T, T, T, T>; // F, logJp, vol, lambda, mu, kappa, g

template <std::size_t OFFSET, class T, int dim>
struct FIELDS_WITH_OFFSET<OFFSET, NEOHOOKEAN_BORDEN<T, dim>> {
    enum INDICES { F = OFFSET, LOG_JP, VOL, LAMBDA, MU, KAPPA, G };
};

template <class T, int dim>
class NEOHOOKEAN_BORDEN_FUNCTOR {
public:
    using STORAGE = NEOHOOKEAN_BORDEN<T, dim>;
    using DIFFERENTIAL = BASE_STORAGE<Eigen::Matrix<T, dim*dim, dim*dim>>;
    static const bool useJ = false;
    static const bool projectable = true;

    static std::unique_ptr<STORAGE> Create() {
        return std::make_unique<STORAGE>();
    }

    static void Append(STORAGE& nhb, const VECTOR<int, 2>& handle, T vol, T E, T nu) {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        T kappa = (T).666666 * mu + lambda;
        T g = (T)1; //this should eventually be replaced with the degradation function based on damage
        for (int i = handle(0); i < handle(1); ++i)
            nhb.Insert(i, MATRIX<T, dim>(1), (T)0, vol, lambda, mu, kappa, g);
    }

    static void Compute_Kirchoff_Stress(STORAGE& nhb, MPM_STRESS<T, dim>& stress) {
        TIMER_FLAG("Compute Kirchoff Stress for Neohookean Borden");
        nhb.Join(stress).Each([&](const int i, auto data) {
            auto& [F, _, vol, lambda, mu, kappa, g, ks, __] = data;
            T J = F.determinant();
            MATRIX<T,dim> B = F * F.transpose();
            MATRIX<T,dim> devB = B - (MATRIX<T,dim>(1.0) * ((T)1 / dim * B.trace()));
            MATRIX<T,dim> tau_dev = mu * std::pow(J, -(T)2 / (T)dim) * devB;
            T prime = kappa / 2 * (J - 1 / J);
            MATRIX<T,dim> tau_vol = J * prime * MATRIX<T,dim>(1.0);
            MATRIX<T,dim> tau;
            if(J >= 1)
                tau = g * (tau_dev + tau_vol);
            else
                tau = (g * tau_dev) + tau_vol;
            ks += -vol * tau; //tau = P * F.transpose
        });
    }
    
    static void Compute_Psi(STORAGE& nhb, T w, T& Psi) {
        //TODO
        Psi = 0;
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& nhb, T w, BASE_STORAGE<MATRIX<T, dim>>& P) {
        //TODO
    }

    static void Compute_First_PiolaKirchoff_Stress_Derivative(STORAGE& nhb, T w, bool projectSPD, DIFFERENTIAL& dP_div_dF) {
        //TOD
    }
};

}
