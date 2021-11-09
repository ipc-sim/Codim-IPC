#pragma once
#include <Storage/storage.hpp>

namespace JGSL{

template <class T, int dim>
using MPM_PARTICLES = BASE_STORAGE<VECTOR<T, dim>, VECTOR<T, dim>, MATRIX<T, dim>, MATRIX<T, dim>, T>; // X V gradV C m

template <std::size_t OFFSET, class T, int dim>
struct FIELDS_WITH_OFFSET<OFFSET, MPM_PARTICLES<T, dim>> {
    enum INDICES { X = OFFSET, V, GRAD_V, C, M };
};

template <class T, int dim>
using MPM_STRESS = BASE_STORAGE<MATRIX<T, dim>, VECTOR<T, dim>>; // ks ls

template <std::size_t OFFSET, class T, int dim>
struct FIELDS_WITH_OFFSET<OFFSET, MPM_STRESS<T, dim>> {
    enum INDICES { KS = OFFSET, LS };
};

}