#pragma once

#include <Math/VECTOR.h>
#include <Storage/prelude.hpp>
#include <MPM/TRANSFER.h>

namespace py = pybind11;
namespace JGSL {

template <class T>
using EQUATION_OF_STATE = BASE_STORAGE<T, T, T, T>; // J, vol, bulk, gamma

template <std::size_t OFFSET, class T>
struct FIELDS_WITH_OFFSET<OFFSET, EQUATION_OF_STATE<T>> {
    enum INDICES { J = OFFSET, VOL, BULK, GAMMA };
};

template <class T, int dim>
class EQUATION_OF_STATE_FUNCTOR {
public:
    using STORAGE = EQUATION_OF_STATE<T>;
    using DIFFERENTIAL = BASE_STORAGE<T>;
    static const bool useJ = true;
    static const bool projectable = false;

    static std::unique_ptr<STORAGE> Create() {
        return std::make_unique<STORAGE>();
    }

    static void Append(STORAGE& eos, const VECTOR<int, 2>& handle, T vol, T bulk, T gamma) {
        for (int i = handle(0); i < handle(1); ++i)
            eos.Insert(i, (T)1, vol, bulk, gamma);
    }

    static void Compute_Kirchoff_Stress(STORAGE& eos, MPM_STRESS<T, dim>& stress) {
        TIMER_FLAG("Compute Kirchoff Stress for Equation of State");
        eos.Join(stress).Each([&](const int i, auto data) {
            auto& [J, vol, bulk, gamma, ks, _] = data;
            T J2 = J * J;
            T J4 = J2 * J2;
            T J7 = J * J2 * J4;
            T dpsi_dJ = -bulk * ((T)1 / J7 - (T)1);
            ks += -vol * dpsi_dJ * J * MATRIX<T, dim>(1);
        });
    }

    static void Compute_Psi(STORAGE& fcr, T w, T& Psi) {
        // TODO
        Psi = 0.0;
    }

    static void Compute_First_PiolaKirchoff_Stress(STORAGE& eos, T scale, BASE_STORAGE<T> dJ) {
        //TODO
    }

    static void Compute_First_PiolaKirchoff_Stress_Derivative(STORAGE& eos, T scale, DIFFERENTIAL& ddJ) {
        //TODO
    } 
};

}
