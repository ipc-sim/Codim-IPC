#pragma once
//#####################################################################
// Class PARTICLES
//#####################################################################
#include <Math/VECTOR.h>
#include <vector>

namespace JGSL {

template <class ATTR>
class PARTICLES {
public:
    std::vector<ATTR> data;
    PARTICLES() {}
    PARTICLES(const PARTICLES& particles) { puts("Called copy paste (unnecessay used detected)"); exit(0); }

    ATTR& operator[] (int d) { return data[d]; }
    const ATTR& operator[] (int d) const { return data[d]; }

    //#################################################################
    // Function size
    //#################################################################
    size_t size() const {
        return data.size();
    }
};

//#####################################################################
void Export_Particles(py::module& m) {
    py::class_<PARTICLES<float>>(m, "ScalarfParticles").def(py::init<>());
    py::class_<PARTICLES<VECTOR<float, 2>>>(m, "Vector2fParticles").def(py::init<>());
    py::class_<PARTICLES<VECTOR<float, 3>>>(m, "Vector3fParticles").def(py::init<>());
    py::class_<PARTICLES<VECTOR<float, 4>>>(m, "Vector4fParticles").def(py::init<>());
    py::class_<PARTICLES<MATRIX<float, 2>>>(m, "Matrix2fParticles").def(py::init<>());
    py::class_<PARTICLES<MATRIX<float, 3>>>(m, "Matrix3fParticles").def(py::init<>());
    py::class_<PARTICLES<double>>(m, "ScalardParticles").def(py::init<>());
    py::class_<PARTICLES<VECTOR<double, 2>>>(m, "Vector2dParticles").def(py::init<>());
    py::class_<PARTICLES<VECTOR<double, 3>>>(m, "Vector3dParticles").def(py::init<>());
    py::class_<PARTICLES<VECTOR<double, 4>>>(m, "Vector4dParticles").def(py::init<>());
    py::class_<PARTICLES<MATRIX<double, 2>>>(m, "Matrix2dParticles").def(py::init<>());
    py::class_<PARTICLES<MATRIX<double, 3>>>(m, "Matrix3dParticles").def(py::init<>());
    py::class_<PARTICLES<int>>(m, "ScalariParticles").def(py::init<>());
    py::class_<PARTICLES<VECTOR<int, 3>>>(m, "Vector3iParticles").def(py::init<>());
    py::class_<PARTICLES<VECTOR<int, 4>>>(m, "Vector4iParticles").def(py::init<>());
}

}