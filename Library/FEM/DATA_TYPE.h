#pragma once
#include <Storage/storage.hpp>

namespace JGSL{

template <class T, int dim>
using MESH_NODE = BASE_STORAGE<VECTOR<T, dim>>; // X

template <int dim>
using MESH_ELEM = BASE_STORAGE<VECTOR<int, dim + 1>>; // Elem

template <class T, int dim>
using MESH_NODE_ATTR = BASE_STORAGE<VECTOR<T, dim>, VECTOR<T, dim>, VECTOR<T, dim>, T>; // x0, v, g, mass

template <std::size_t OFFSET, class T, int dim>
struct FIELDS_WITH_OFFSET<OFFSET, MESH_NODE_ATTR<T, dim>> {
    enum INDICES { x0 = OFFSET, v, g, m };
};

template <class T, int dim>
using MESH_ELEM_ATTR = BASE_STORAGE<MATRIX<T, dim>, MATRIX<T, dim>>; // IB, P

template <std::size_t OFFSET, class T, int dim>
struct FIELDS_WITH_OFFSET<OFFSET, MESH_ELEM_ATTR<T, dim>> {
    enum INDICES { IB = OFFSET, P };
};

template<class T>
using SCALAR_STORAGE = BASE_STORAGE<T>;
template<class T, int dim>
using VECTOR_STORAGE = BASE_STORAGE<VECTOR<T, dim>>;

}