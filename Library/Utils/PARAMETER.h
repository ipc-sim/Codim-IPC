#pragma once

#include <type_traits>
#include <typeinfo>
#include <boost/property_tree/ptree.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace JGSL {
namespace PARAMETER {

boost::property_tree::ptree pt;

template <class T>
void Set(std::string key, T value) { pt.put(key, value); }

template <class T>
T Get(std::string key, T default_value) { return pt.get(key, default_value); }

void Export(py::module &m) {
    m.def("Set_Parameter", &Set<bool>);
    m.def("Set_Parameter", &Set<int>);
    m.def("Set_Parameter", &Set<float>);
    m.def("Set_Parameter", &Set<double>);
    m.def("Set_Parameter", &Set<std::string>);
    m.def("Get_Parameter", &Get<bool>);
    m.def("Get_Parameter", &Get<int>);
    m.def("Get_Parameter", &Get<float>);
    m.def("Get_Parameter", &Get<double>);
    m.def("Get_Parameter", &Get<std::string>);
}

}
}