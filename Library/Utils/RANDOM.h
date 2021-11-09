#pragma once
//#####################################################################
// Class RANDOM
//#####################################################################
#include <Math/VECTOR.h>
#include <random>

namespace JGSL {

template <class T>
class RANDOM {
public:
    std::mt19937 g;
    RANDOM(unsigned s = 123):g(s){}
    ~RANDOM(){}

    T Rand_Real(const T& a,const T& b) const{
        std::uniform_real_distribution<T> d(a,b);
        return d(g);
    }

    int Rand_Int(const int a,const int b) const{
        std::uniform_int_distribution<> d(a, b);
        return d(g);
    }
};

//#####################################################################
void Export_Random(py::module& m) {
    py::class_<RANDOM<float>>(m, "ScalarfRandom").def(py::init<>());
    py::class_<RANDOM<double>>(m, "ScalardRandom").def(py::init<>());
}

}