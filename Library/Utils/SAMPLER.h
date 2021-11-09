#pragma once

#include <Math/VECTOR.h>
#include <MPM/TRANSFER.h>

namespace py = pybind11;
namespace JGSL {

template<class T, int dim>
VECTOR<int, 2> Uniform_Sample_In_Box(
    MPM_PARTICLES<T, dim>& particles,
    const VECTOR<T, dim>& lower,
    const VECTOR<T, dim>& upper,
    const VECTOR<T, dim>& velocity,
    T mass,
    T dx,
    T ppc
) {
    puts("Called!!!");
    printf("Sampled in lower (%.8f %.8f)\n", lower(0), lower(1));
    printf("           upper (%.8f %.8f)\n", upper(0), upper(1));
    T effectiveDx = dx / pow(ppc, (T)1 / (T)dim); //this gives us the correct ppc count
    if (dim == 2) {
        int size0 = (int)std::round((upper(0) - lower(0)) / effectiveDx);
        int size1 = (int)std::round((upper(1) - lower(1)) / effectiveDx);
        int first_index = -1, last_index = -1;
        for (int i = 0; i <= size0; ++i) {
            for (int j = 0; j <= size1; ++j) {
                T dbx = effectiveDx * ((T)rand() / RAND_MAX - 0.5);
                T dby = effectiveDx * ((T)rand() / RAND_MAX - 0.5);
                auto index = particles.Append(
                    VECTOR<T, dim>(lower(0) + effectiveDx * i + dbx, lower(1) + effectiveDx * j + dby),
                    velocity,
                    MATRIX<T, dim>(),
                    MATRIX<T, dim>(),
                    mass
                );
                if (i == 0 && j == 0) {
                    first_index = index;
                }
                last_index = index;
            }
        }
        printf("Total samplers number : %lu\n", particles.data.size());
        printf("Total samplers number : %lu %d %d\n", particles.size, first_index, last_index + 1);
        assert(0 <= first_index && first_index <= last_index);
        return VECTOR<int, 2>(first_index, last_index + 1);
    } else {
        int size0 = (int)std::round((upper(0) - lower(0)) / effectiveDx);
        int size1 = (int)std::round((upper(1) - lower(1)) / effectiveDx);
        int size2 = (int)std::round((upper(2) - lower(2)) / effectiveDx);
        int first_index = -1, last_index = -1;
        for (int i = 0; i <= size0; ++i)
            for (int j = 0; j <= size1; ++j)
                for (int k = 0; k <= size2; ++k) {
                    T dbx = effectiveDx * ((T)rand() / RAND_MAX - 0.5) * 0;
                    T dby = effectiveDx * ((T)rand() / RAND_MAX - 0.5) * 0;
                    T dbz = effectiveDx * ((T)rand() / RAND_MAX - 0.5) * 0;
                    auto index = particles.Append(
                            VECTOR<T, dim>(lower(0) + effectiveDx * i + dbx, lower(1) + effectiveDx * j + dby, lower(2) + effectiveDx * k + dbz),
                            velocity,
                            MATRIX<T, dim>(),
                            MATRIX<T, dim>(),
                            mass
                    );
                    if (i == 0 && j == 0 && k == 0) {
                        first_index = index;
                    }
                    last_index = index;
                }
        printf("Total samplers number : %lu\n", particles.data.size());
        assert(0 <= first_index && first_index <= last_index);
        return VECTOR<int, 2>(first_index, last_index + 1);
    }
}

/* move to somewhere else
template<class T, int dim>
VECTOR<int, 2> Uniform_Sample_In_Box_FLIP(
    FLIP_PARTICLES<T, dim>& particles,
    const VECTOR<T, dim>& lower,
    const VECTOR<T, dim>& upper,
    const VECTOR<T, dim>& velocity, // initial velocity
    T dx,
    T ppc
) {
    puts("Called!!!");
    T effectiveDx = dx / pow(ppc, (T)1 / (T)dim); //this gives us the correct ppc count
    if (dim == 2) {
        printf("Sampled in lower (%.8f %.8f)\n", lower(0), lower(1));
        printf("           upper (%.8f %.8f)\n", upper(0), upper(1));
        int size0 = (int)std::round((upper(0) - lower(0)) / effectiveDx);
        int size1 = (int)std::round((upper(1) - lower(1)) / effectiveDx);
        int first_index = -1, last_index = -1;
        for (int i = 0; i < size0; ++i) {
            for (int j = 0; j < size1; ++j) {
                auto index = particles.Append(
                    VECTOR<T, dim>(lower(0) + effectiveDx * i, lower(1) + effectiveDx * j),
                    velocity
                );
                if (i == 0 && j == 0) {
                    first_index = index;
                }
                last_index = index;
            }
        }
        assert(0 <= first_index && first_index <= last_index);
        return VECTOR<int, 2>(first_index, last_index + 1);
    } else {
        printf("Sampled in lower (%.8f %.8f %.8f)\n", lower(0), lower(1), lower(2));
        printf("           upper (%.8f %.8f %.8f)\n", upper(0), upper(1), upper(2));
        int size0 = (int)std::round((upper(0) - lower(0)) / effectiveDx);
        int size1 = (int)std::round((upper(1) - lower(1)) / effectiveDx);
        int size2 = (int)std::round((upper(2) - lower(2)) / effectiveDx);
        int first_index = -1, last_index = -1;
        for (int i = 0; i <= size0; ++i)
            for (int j = 0; j <= size1; ++j)
                for (int k = 0; k <= size2; ++k) {
                    T dbx = effectiveDx * ((T)rand() / RAND_MAX - 0.5);
                    T dby = effectiveDx * ((T)rand() / RAND_MAX - 0.5);
                    T dbz = effectiveDx * ((T)rand() / RAND_MAX - 0.5);
//                    printf("sample : %d, %d, %d\n", i, j, k);
                    auto index = particles.Append(
                            VECTOR<T, dim>(lower(0) + effectiveDx * i + dbx, lower(1) + effectiveDx * j + dby, lower(2) + effectiveDx * k + dbz),
                            velocity
                    );
                    if (i == 0 && j == 0 && k == 0) {
                        first_index = index;
                    }
                    last_index = index;
                }
        printf("Total samplers number : %lu\n", particles.data.size());
        assert(0 <= first_index && first_index <= last_index);
        return VECTOR<int, 2>(first_index, last_index + 1);
    }
    printf("Total samplers number : %lu\n", particles.data.size());
}

template<class T, int dim>
VECTOR<int, 2> Uniform_Sample_In_Box_APIC(
    APIC_PARTICLES<T, dim>& particles,
    const VECTOR<T, dim>& lower,
    const VECTOR<T, dim>& upper,
    const VECTOR<T, dim>& velocity,
    T dx,
    T ppc
) {
    puts("Called!!!");
    T effectiveDx = dx / pow(ppc, (T)1 / (T)dim);
    if (dim == 2) {
        printf("Sampled in lower (%.8f %.8f)\n", lower(0), lower(1));
        printf("           upper (%.8f %.8f)\n", upper(0), upper(1));
        int size0 = (int)std::round((upper(0) - lower(0)) / effectiveDx);
        int size1 = (int)std::round((upper(1) - lower(1)) / effectiveDx);
        int first_index = -1, last_index = -1;
        for (int i = 0; i < size0; ++i) {
            for (int j = 0; j < size1; ++j) {
                auto index = particles.Append(
                    VECTOR<T, dim>(lower(0) + effectiveDx * i, lower(1) + effectiveDx * j),
                    velocity,
                    MATRIX<T,dim>(0.0)
                );
                if (i == 0 && j == 0) {
                    first_index = index;
                }
                last_index = index;
            }
        }
        assert(0 <= first_index && first_index <= last_index);
        return VECTOR<int, 2>(first_index, last_index + 1);
    } else {
        printf("effectivedx: %f \n", effectiveDx);
        printf("Sampled in lower (%.8f %.8f %.8f)\n", lower(0), lower(1), lower(2));
        printf("           upper (%.8f %.8f %.8f)\n", upper(0), upper(1), upper(2));
        int size0 = (int)std::round((upper(0) - lower(0)) / effectiveDx);
        int size1 = (int)std::round((upper(1) - lower(1)) / effectiveDx);
        int size2 = (int)std::round((upper(2) - lower(2)) / effectiveDx);
        int first_index = -1, last_index = -1;
        for (int i = 0; i <= size0; ++i)
            for (int j = 0; j <= size1; ++j)
                for (int k = 0; k <= size2; ++k) {
                    T dbx = effectiveDx * ((T)rand() / RAND_MAX - 0.5);
                    T dby = effectiveDx * ((T)rand() / RAND_MAX - 0.5);
                    T dbz = effectiveDx * ((T)rand() / RAND_MAX - 0.5);
                    auto index = particles.Append(
                            VECTOR<T, dim>(lower(0) + effectiveDx * i + dbx, lower(1) + effectiveDx * j + dby, lower(2) + effectiveDx * k + dbz),
                            velocity,
                            MATRIX<T,dim>(0.0)
                    );
                    if (i == 0 && j == 0 && k == 0) {
                        first_index = index;
                    }
                    last_index = index;
                }
        printf("Total samplers number : %lu\n", particles.data.size());
        assert(0 <= first_index && first_index <= last_index);
        return VECTOR<int, 2>(first_index, last_index + 1);
    }
}
*/

void Export_Sampler(py::module& m) {
    m.def("Uniform_Sample_In_Box", &Uniform_Sample_In_Box<double, 2>);
    m.def("Uniform_Sample_In_Box", &Uniform_Sample_In_Box<double, 3>);
/* move to somewhere else
        m.def("Uniform_Sample_In_Box_flip", &Uniform_Sample_In_Box_FLIP<double, 2>, "uniform sample in box");
    m.def("Uniform_Sample_In_Box_flip", &Uniform_Sample_In_Box_FLIP<double, 3>, "uniform sample in box");

    m.def("Uniform_Sample_In_Box_APIC", &Uniform_Sample_In_Box_APIC<double, 2>, "uniform sample in box");
    m.def("Uniform_Sample_In_Box_APIC", &Uniform_Sample_In_Box_APIC<double, 3>, "uniform sample in box");
*/
}
}