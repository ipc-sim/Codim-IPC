#pragma once

#include <FEM/DEFORMATION_GRADIENT.h>
#include <FEM/ELEM_TO_NODE.h>
#include <FEM/BOUNDARY_CONDITION.h>
#include <FEM/TimeStepper/TIME_STEPPER.h>
#include <FEM/SHELL.h>
#include <FEM/Shell/DISCRETE_SHELL.h>
#include <FEM/FRACTURE.h>
#include <FEM/IPC.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace JGSL {

template <class EX_SPACE, class MEM_SPACE, std::size_t BIN_SIZE> void Export_Storage(py::module &m) {
    // Basic Single Element, Single Precision Float Storages
    py::class_<BASE_STORAGE<float>>(m, "SfStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<float, 2>>>(m, "V2fStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<float, 3>>>(m, "V3fStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<float, 4>>>(m, "V4fStorage").def(py::init());
    py::class_<BASE_STORAGE<MATRIX<float, 2>>>(m, "M2fStorage").def(py::init());
    py::class_<BASE_STORAGE<MATRIX<float, 3>>>(m, "M3fStorage").def(py::init());
    py::class_<BASE_STORAGE<MATRIX<float, 4>>>(m, "M4fStorage").def(py::init());
    // Basic Single Element, Double Precision Float Storages
    py::class_<BASE_STORAGE<double>>(m, "SdStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<double, 2>>>(m, "V2dStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<double, 3>>>(m, "V3dStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<double, 4>>>(m, "V4dStorage").def(py::init());
    py::class_<BASE_STORAGE<MATRIX<double, 2>>>(m, "M2dStorage").def(py::init());
    py::class_<BASE_STORAGE<MATRIX<double, 3>>>(m, "M3dStorage").def(py::init());
    py::class_<BASE_STORAGE<MATRIX<double, 4>>>(m, "M4dStorage").def(py::init());
    py::class_<BASE_STORAGE<int>>(m, "SiStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<int, 2>>>(m, "V2iStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<int, 3>>>(m, "V3iStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<int, 4>>>(m, "V4iStorage").def(py::init());
    // advanced mesh storage
    py::class_<BASE_STORAGE<VECTOR<double, 2>, VECTOR<double, 2>, VECTOR<double, 2>, double>>(m, "V2dV2dV2dSdStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<double, 3>, VECTOR<double, 3>, VECTOR<double, 3>, double>>(m, "V3dV3dV3dSdStorage").def(py::init());
    py::class_<BASE_STORAGE<MATRIX<double, 2>, MATRIX<double, 2>>>(m, "M2dM2dSdStorage").def(py::init());
    py::class_<BASE_STORAGE<MATRIX<double, 3>, MATRIX<double, 3>>>(m, "M3dM3dSdStorage").def(py::init());
    // advanced moving boundary condition storage
    py::class_<BASE_STORAGE<VECTOR<int, 2>, VECTOR<double, 2>, VECTOR<double, 2>, VECTOR<double, 2>, double>>(m, "V2iV2dV2dV2dSdStorage").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<int, 2>, VECTOR<double, 3>, VECTOR<double, 3>, VECTOR<double, 3>, double>>(m, "V2iV3dV3dV3dSdStorage").def(py::init());
    // Advanced Particle Storages
    py::class_<BASE_STORAGE<VECTOR<float, 2>, VECTOR<float, 2>, float>>(m, "XVm2f").def(py::init());
    py::class_<BASE_STORAGE<VECTOR<float, 3>, VECTOR<float, 3>, float>>(m, "XVm3f").def(py::init());
}

void Export_FEM(py::module& m) {
    Export_Deformation_Gradient(m);
    Export_Elem_To_Node(m);
    Export_Boundary_Condition(m);

    py::module time_stepper_m = m.def_submodule("TimeStepper", "A submodule of JGSL for FEM time steppers");
    Export_Time_Stepper(time_stepper_m);

#ifdef ENABLE_FEM_SHELL
    Export_Shell(m);
    Export_Discrete_Shell(m);
#endif

#ifdef ENABLE_FEM_FRACTURE
    py::module fracture_m = m.def_submodule("Fracture", "A submodule of JGSL for FEM fracture");
    Export_Fracture(fracture_m);
#endif
}

}
