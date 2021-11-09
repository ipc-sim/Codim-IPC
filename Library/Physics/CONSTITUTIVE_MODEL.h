#pragma once

#include <Storage/prelude.hpp>
#include <Utils/PROFILER.h>
#include <MPM/DATA_TYPE.h>
#include <Physics/EQUATION_OF_STATE.h>
#include <Physics/FIXED_COROTATED.h>
#include <Physics/NEOHOOKEAN.h>
#include <Physics/LINEAR_COROTATED.h>
#include <Physics/NEOHOOKEAN_BORDEN.h>
#include <Physics/SYMMETRIC_DIRICHLET.h>
#include <Physics/STVK_HENCKY.h>
#include <Physics/NACC_PLASTICITY.h>

namespace py = pybind11;
namespace JGSL {

template <class MODEL, class T, int dim>
std::unique_ptr<MODEL> Create_Constitutive_Model(const VECTOR<T, dim>& val) {
    return std::make_unique<MODEL>();
}

template <class MODEL, class T, int dim>
void Evolve_F_Based_Strain(MODEL& model, MPM_PARTICLES<T, dim>& particles, T dt) {
    TIMER_FLAG("Evolve F");
    using JOINED = decltype(model.Join(particles));
    model.Join(particles).Par_Each([&](const int i, auto data) {
        auto& F = std::get<FIELDS<JOINED>::F>(data);
        auto& gradV = std::get<FIELDS<JOINED>::GRAD_V>(data);
        F = (MATRIX<T, dim>(1) + dt * gradV) * F;
    });
}

template <class MODEL, class T, int dim>
void Evolve_J_Based_Strain(MODEL& model, MPM_PARTICLES<T, dim>& particles, T dt) {
    TIMER_FLAG("Evolve J");
    using JOINED = decltype(model.Join(particles));
    model.Join(particles).Par_Each([&](const int i, auto data) {
        auto& J = std::get<FIELDS<JOINED>::J>(data);
        auto& gradV = std::get<FIELDS<JOINED>::GRAD_V>(data);
        J = (1 + dt * gradV.trace()) * J;
    });
}

//TEST FEM material switch
template <class ELASTICITY_TYPE, class T, int dim>
void Compute_First_PiolaKirchoff_Stress_Derivative(void) {
    // ELASTICITY_TYPE::template Compute_Psi<T, dim>();
}

template <class MODEL, class T, int dim>
py::module Export_Model_Submodule(py::module& m, const std::string &name, bool export_storage = true) {

    // The storage type might already been exported, so we add an option
    if (export_storage) {
        std::string storage_name = std::string(name).append("_STORAGE_" + std::to_string(dim));
        py::class_<typename MODEL::STORAGE>(m, storage_name.c_str(), py::module_local()).def(py::init<>());
    }

    // Export the module
    std::string module_name = std::string(name).append("_" + std::to_string(dim));
    py::module module = m.def_submodule(module_name.c_str());
    
    // Define member functions
    module.def("Create", &MODEL::Create);
    module.def("Append", &MODEL::Append);
    module.def("Compute_Kirchoff_Stress", &MODEL::Compute_Kirchoff_Stress);
    return module;
}

template <class T, int dim>
void Export_Constitutive_Model(py::module& m) {

    using FCR_MODEL = FIXED_COROTATED_FUNCTOR<T, dim>;
    auto fcr_mod = Export_Model_Submodule<FCR_MODEL, T, dim>(m, "FIXED_COROTATED");
    fcr_mod.def("Append_FEM", &FCR_MODEL::Append_FEM);
    fcr_mod.def("Append_All_FEM", &FCR_MODEL::Append_All_FEM);
    fcr_mod.def("All_Append_FEM", &FCR_MODEL::All_Append_FEM);
    fcr_mod.def("Evolve_Strain", &Evolve_F_Based_Strain<typename FCR_MODEL::STORAGE, T, dim>);

    using EOS_MODEL = EQUATION_OF_STATE_FUNCTOR<T, dim>;
    auto eos_mod = Export_Model_Submodule<EOS_MODEL, T, dim>(m, "EQUATION_OF_STATE" , dim == 2);
    eos_mod.def("Evolve_Strain", &Evolve_J_Based_Strain<typename EOS_MODEL::STORAGE, T, dim>);

    using LCR_MODEL = FIXED_COROTATED_FUNCTOR<T, dim>;
    auto lcr_mod = Export_Model_Submodule<LCR_MODEL, T, dim>(m, "LINEAR_COROTATED", false);
    lcr_mod.def("Evolve_Strain", &Evolve_F_Based_Strain<typename LCR_MODEL::STORAGE, T, dim>);

    using NHB_MODEL = NEOHOOKEAN_BORDEN_FUNCTOR<T, dim>;
    auto nhb_mod = Export_Model_Submodule<NHB_MODEL, T, dim>(m, "NEOHOOKEAN_BORDEN");
    nhb_mod.def("Evolve_Strain", &Evolve_F_Based_Strain<typename NHB_MODEL::STORAGE, T, dim>);

    // TODO: Do Project Strain NACC
    // m.def("Project_Strain_NACC", &Project_Strain_NACC<double, 2>);
}

}