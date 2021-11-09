#include <../Projects/Melting/NilsCorrector.h>
#include <Eigen/Eigen>
#include <Math/VECTOR.h>
#include <Physics/CONSTITUTIVE_MODEL.h>
#include <Utils/COLLIDER.h>
#include <Utils/SAMPLER.h>
#include <Utils/VISUALIZER.h>
#include <Utils/RANDOM.h>
#include <Utils/PROFILER.h>
#include <MPM/TRANSFER.h>
#include <Storage/prelude.hpp>
#include <Storage2/PRELUDE.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace JGSL {

PYBIND11_MODULE(JGSL_MPM, m) {
    // Export_Vector(m);
    // Export_Random(m);
    Export_Particles(m);
    Export_Sparse_Grid(m);
    py::module sampler_m = m.def_submodule("Sampler", "A submodule of JGSL for Sampler");
    Export_Sampler(sampler_m);
    py::module visualizer_m = m.def_submodule("Visualizer", "A submodule of JGSL for Visualizer");
    Export_Visualizer(visualizer_m);
    Export_Transfer(m);
    // Export_Constitutive_Model<double, 2>(m);
    // Export_Constitutive_Model<double, 3>(m);
    // Export_ID_Allocator(storage_m);
    Export_Collider(m);
    // Export_Profiler(m);
    Export_Melting(m);
}

}