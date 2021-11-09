#pragma once

#include <FEM/TimeStepper/SYMPLECTIC_EULER.h>
#include <FEM/TimeStepper/IMPLICIT_EULER.h>
#include <FEM/TimeStepper/SHAPE_UP.h>
#include <FEM/TimeStepper/ADMM.h>

namespace py = pybind11;
namespace JGSL {

void Export_Time_Stepper(py::module& m) {
    py::module symplectic_Euler_m = m.def_submodule("SymplecticEuler", "A submodule of JGSL for symplectic Euler time stepping");
    symplectic_Euler_m.def("Advance_One_Step", &Advance_One_Step_SE<double, 2>);
    symplectic_Euler_m.def("Advance_One_Step", &Advance_One_Step_SE<double, 3>);

    py::module implicit_Euler_m = m.def_submodule("ImplicitEuler", "A submodule of JGSL for implicit Euler time stepping");
    implicit_Euler_m.def("Advance_One_Step_SU", &Advance_One_Step_SU<double, 2>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    implicit_Euler_m.def("Advance_One_Step_ADMM", &Advance_One_Step_ADMM<double, 2>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    implicit_Euler_m.def("Advance_One_Step", &Advance_One_Step_IE<double, 2>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    implicit_Euler_m.def("Advance_One_Step", &Advance_One_Step_IE<double, 3>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    implicit_Euler_m.def("Advance_One_Step_Shell", &Advance_One_Step_IE<double, 3, true>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    implicit_Euler_m.def("Check_Gradient_FEM", &Check_Gradient_FEM<double, 2>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    
    implicit_Euler_m.def("Initialize_Elastic_IPC", &Initialize_Elastic_IPC<double, 3>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    implicit_Euler_m.def("Advance_One_Step_EIPC", &Advance_One_Step_IE<double, 3, false, true>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    implicit_Euler_m.def("Write_COM", &Write_COM<double>, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
}

}