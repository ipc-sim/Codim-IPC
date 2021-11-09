#pragma once
//#####################################################################
// Function Export_StdContainer
//#####################################################################
#include <Math/VECTOR.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<JGSL::VECTOR<int, 2>>);
PYBIND11_MAKE_OPAQUE(std::vector<JGSL::VECTOR<int, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<JGSL::VECTOR<int, 4>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<int, 4>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<int, 6>>);
PYBIND11_MAKE_OPAQUE(std::vector<JGSL::VECTOR<double, 2>>);
PYBIND11_MAKE_OPAQUE(std::vector<JGSL::VECTOR<double, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::map<std::pair<int, int>, int>);

namespace py = pybind11;
namespace JGSL {

void Export_StdContainer(py::module& m) {
    py::bind_vector<std::vector<VECTOR<int, 2>>>(m, "StdVectorVector2i");
    py::bind_vector<std::vector<VECTOR<int, 3>>>(m, "StdVectorVector3i");
    py::bind_vector<std::vector<VECTOR<int, 4>>>(m, "StdVectorVector4i");
    py::bind_vector<std::vector<std::array<int, 4>>>(m, "StdVectorArray4i");
    py::bind_vector<std::vector<std::array<int, 6>>>(m, "StdVectorArray6i");
    py::bind_vector<std::vector<VECTOR<double, 2>>>(m, "StdVectorVector2d");
    py::bind_vector<std::vector<VECTOR<double, 3>>>(m, "StdVectorVector3d");
    py::bind_vector<std::vector<int>>(m, "StdVectorXi");
    py::bind_vector<std::vector<double>>(m, "StdVectorXd");
    py::bind_vector<std::vector<float>>(m, "StdVectorXf");
    py::bind_map<std::map<std::pair<int, int>, int>>(m, "StdMapPairiToi");
}

};