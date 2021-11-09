#pragma once

#include "./FULL_STORAGE.hpp"
#include "./RANGED_STORAGE.hpp"

namespace py = pybind11;
namespace JGSL {

void Export_Storage_2(py::module &m) {
    // -------------
    // Full Storages
    // -------------

    // Basic Single Element, Single Precision Float Storages
    py::class_<storage::FullStorage<storage::Config, float>>(m, "SfFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<float, 2>>>(m, "V2fFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<float, 3>>>(m, "V3fFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<float, 4>>>(m, "V4fFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, MATRIX<float, 2>>>(m, "M2fFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, MATRIX<float, 3>>>(m, "M3fFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, MATRIX<float, 4>>>(m, "M4fFullStorage").def(py::init());
    // Basic Single Element, Double Precision Float Storages
    py::class_<storage::FullStorage<storage::Config, double>>(m, "SdFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<double, 2>>>(m, "V2dFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<double, 3>>>(m, "V3dFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<double, 4>>>(m, "V4dFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, MATRIX<double, 2>>>(m, "M2dFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, MATRIX<double, 3>>>(m, "M3dFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, MATRIX<double, 4>>>(m, "M4dFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, int>>(m, "SiFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<int, 2>>>(m, "V2iFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<int, 3>>>(m, "V3iFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<int, 4>>>(m, "V4iFullStorage").def(py::init());
    // advanced mesh storage
    py::class_<storage::FullStorage<storage::Config, VECTOR<double, 2>, VECTOR<double, 2>, VECTOR<double, 2>, double>>(m, "V2dV2dV2dSdFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<double, 3>, VECTOR<double, 3>, VECTOR<double, 3>, double>>(m, "V3dV3dV3dSdFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, MATRIX<double, 2>, MATRIX<double, 2>>>(m, "M2dM2dSdFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, MATRIX<double, 3>, MATRIX<double, 3>>>(m, "M3dM3dSdFullStorage").def(py::init());
    // advanced moving boundary condition storage
    py::class_<storage::FullStorage<storage::Config, VECTOR<int, 2>, VECTOR<double, 2>, VECTOR<double, 2>, VECTOR<double, 2>, double>>(m, "V2iV2dV2dV2dSdFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<int, 2>, VECTOR<double, 3>, VECTOR<double, 3>, VECTOR<double, 3>, double>>(m, "V2iV3dV3dV3dSdFullStorage").def(py::init());
    // Advanced Particle Storages
    py::class_<storage::FullStorage<storage::Config, VECTOR<float, 2>, VECTOR<float, 2>, float>>(m, "XVm2fFullStorage").def(py::init());
    py::class_<storage::FullStorage<storage::Config, VECTOR<float, 3>, VECTOR<float, 3>, float>>(m, "XVm3fFullStorage").def(py::init());

    // ---------------
    // Ranged Storages
    // ---------------

    // Basic Single Element, Single Precision Float Storages
    py::class_<storage::RangedStorage<storage::Config, float>>(m, "SfRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<float, 2>>>(m, "V2fRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<float, 3>>>(m, "V3fRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<float, 4>>>(m, "V4fRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, MATRIX<float, 2>>>(m, "M2fRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, MATRIX<float, 3>>>(m, "M3fRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, MATRIX<float, 4>>>(m, "M4fRangedStorage").def(py::init());
    // Basic Single Element, Double Precision Float Storages
    py::class_<storage::RangedStorage<storage::Config, double>>(m, "SdRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<double, 2>>>(m, "V2dRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<double, 3>>>(m, "V3dRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<double, 4>>>(m, "V4dRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, MATRIX<double, 2>>>(m, "M2dRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, MATRIX<double, 3>>>(m, "M3dRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, MATRIX<double, 4>>>(m, "M4dRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, int>>(m, "SiRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<int, 2>>>(m, "V2iRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<int, 3>>>(m, "V3iRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<int, 4>>>(m, "V4iRangedStorage").def(py::init());
    // advanced mesh storage
    py::class_<storage::RangedStorage<storage::Config, VECTOR<double, 2>, VECTOR<double, 2>, VECTOR<double, 2>, double>>(m, "V2dV2dV2dSdRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<double, 3>, VECTOR<double, 3>, VECTOR<double, 3>, double>>(m, "V3dV3dV3dSdRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, MATRIX<double, 2>, MATRIX<double, 2>>>(m, "M2dM2dSdRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, MATRIX<double, 3>, MATRIX<double, 3>>>(m, "M3dM3dSdRangedStorage").def(py::init());
    // advanced moving boundary condition storage
    py::class_<storage::RangedStorage<storage::Config, VECTOR<int, 2>, VECTOR<double, 2>, VECTOR<double, 2>, VECTOR<double, 2>, double>>(m, "V2iV2dV2dV2dSdRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<int, 2>, VECTOR<double, 3>, VECTOR<double, 3>, VECTOR<double, 3>, double>>(m, "V2iV3dV3dV3dSdRangedStorage").def(py::init());
    // Advanced Particle Storages
    py::class_<storage::RangedStorage<storage::Config, VECTOR<float, 2>, VECTOR<float, 2>, float>>(m, "XVm2fRangedStorage").def(py::init());
    py::class_<storage::RangedStorage<storage::Config, VECTOR<float, 3>, VECTOR<float, 3>, float>>(m, "XVm3fRangedStorage").def(py::init());
}

}