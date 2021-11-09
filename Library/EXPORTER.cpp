#include <Eigen/Eigen>
#include <Math/VECTOR.h>
#include <Physics/CONSTITUTIVE_MODEL.h>
#include <FEM/FEM_EXPORTER.h>
#include <Utils/STDCONTAINER.h>
#include <Utils/MESHIO.h>
#include <Utils/RANDOM.h>
#include <Utils/PROFILER.h>
#include <Utils/PARAMETER.h>
#include <Storage/prelude.hpp>
#include <Storage2/PRELUDE.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace JGSL {

PYBIND11_MODULE(JGSL, m) {
    m.def("Kokkos_Initialize", &Kokkos_Initialize);
    m.def("Kokkos_Finalize", &Kokkos_Finalize);
    Export_Vector(m);
    Export_Random(m);
    py::module meshIO_m = m.def_submodule("MeshIO", "A submodule of JGSL for mesh IO");
    Export_MeshIO(meshIO_m);
    Export_StdContainer(m);
    Export_Constitutive_Model<double, 2>(m);
    Export_Constitutive_Model<double, 3>(m);
    py::module storage_m = m.def_submodule("Storage", "A submodule of JGSL for Storage");
    Export_Storage<Kokkos::Serial, Kokkos::HostSpace, 4>(storage_m);
    py::module storage_2_m = m.def_submodule("Storage2", "The newer version of Storage");
    Export_Storage_2(storage_2_m);
    Export_Profiler(m);
    py::module FEM_m = m.def_submodule("FEM", "A submodule of JGSL for FEM utils");
    Export_FEM(FEM_m);
    Export_SPARSE_MATRIX(m);
    PARAMETER::Export(m);
}

}