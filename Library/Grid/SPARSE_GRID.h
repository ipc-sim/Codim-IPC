#pragma once
//#####################################################################
// Class SPARSE_GRID
//#####################################################################
#include <Math/BSPLINES.h>
#include <Math/VECTOR.h>
#include <Math/LINEAR.h>
#include <flat_hash_map.hpp>
#include <pybind11/operators.h>
#include <vector>

namespace JGSL {

template <class ATTR, int dim>
class SPARSE_GRID;

template <class ATTR, int dim, class T, int degree>
struct KERNEL_ITERATOR {
    std::tuple<VECTOR<int, dim>, T, VECTOR<T, dim>, ATTR&> operator*();
    void operator++();
    bool operator!=(const KERNEL_ITERATOR<ATTR, 2, T, degree> &other);
};

template <class ATTR, int dim, class T, int degree>
struct KERNEL_ITERATOR_HOLDER {
    KERNEL_ITERATOR_HOLDER(SPARSE_GRID<ATTR, dim> &grid, const BSPLINE_WEIGHTS<T, dim, degree>& spline);
    KERNEL_ITERATOR<ATTR, dim, T, degree> begin();
    KERNEL_ITERATOR<ATTR, dim, T, degree> end();
};

//#####################################################################
// Class SPARSE_GRID
//#####################################################################
template <class ATTR, int dim>
class SPARSE_GRID {
public:
    struct BLOCK {
        ATTR data[64];
        BLOCK() { for (auto& d : data) d = ATTR(); }
    };

    std::vector<BLOCK> data;
    std::vector<uint64_t> offsets;
    std::unordered_map<uint64_t, size_t> mapping;
    SPARSE_GRID() = default;
    JGSL_FORCE_INLINE std::unique_ptr<SPARSE_GRID<ATTR, dim>> copy() {
        auto grid_p = std::make_unique<SPARSE_GRID<ATTR, dim>>();
        (*grid_p) = (*this);
        return grid_p;
    }
    JGSL_FORCE_INLINE SPARSE_GRID<ATTR, dim>& operator-=(const SPARSE_GRID<ATTR, dim>& o) {
        Iterate_Grid([&](const auto& node, auto& g) { (*this)(node) -= o(node); });
        return *this;
    }

    JGSL_FORCE_INLINE uint64_t  Get_Zorder(int x, int y) {
        uint64_t xmask = 0x5555555555555555UL;
        uint64_t ymask = 0xaaaaaaaaaaaaaaaaUL;
        return _pdep_u64((uint64_t)x, xmask) | _pdep_u64((uint64_t)y, ymask);
    }
    JGSL_FORCE_INLINE uint64_t  Get_Zorder(int x, int y, int z) {
        uint64_t xmask = 0x9249249249249249UL;
        uint64_t ymask = 0x2492492492492492UL;
        uint64_t zmask = 0x4924924924924924UL;
        return _pdep_u64((uint64_t)x, xmask) | _pdep_u64((uint64_t)y, ymask) | _pdep_u64((uint64_t)z, zmask);
    }
    JGSL_FORCE_INLINE uint64_t  Get_Zorder(const VECTOR<int, dim>& node) {
        if constexpr (dim == 2) return Get_Zorder(node[0], node[1]);
        else return Get_Zorder(node[0], node[1], node[2]);
    }
    JGSL_FORCE_INLINE ATTR& operator() (int x, int y) {
        uint64_t xmask = 0x5555555555555555UL;
        uint64_t ymask = 0xaaaaaaaaaaaaaaaaUL;
        uint64_t zorder = _pdep_u64((uint64_t)x, xmask) | _pdep_u64((uint64_t)y, ymask);
        uint64_t high_bits = (zorder >> (unsigned)6);
        uint64_t low_bits = (zorder & (unsigned)63);
        if (mapping.find(high_bits) == mapping.end()) {
            data.emplace_back();
            offsets.emplace_back(high_bits);
            mapping[high_bits] = data.size() - 1;
        }
        return data[mapping[high_bits]].data[low_bits];
    }
    JGSL_FORCE_INLINE ATTR& operator() (int x, int y, int z) {
        uint64_t xmask = 0x9249249249249249UL;
        uint64_t ymask = 0x2492492492492492UL;
        uint64_t zmask = 0x4924924924924924UL;
        uint64_t zorder = _pdep_u64((uint64_t)x, xmask) | _pdep_u64((uint64_t)y, ymask) | _pdep_u64((uint64_t)z, zmask);
        uint64_t high_bits = (zorder >> (unsigned)6);
        uint64_t low_bits = (zorder & (unsigned)63);
        if (mapping.find(high_bits) == mapping.end()) {
            data.emplace_back();
            offsets.emplace_back(high_bits);
            mapping[high_bits] = data.size() - 1;
        }
        return data[mapping[high_bits]].data[low_bits];
    }
    JGSL_FORCE_INLINE ATTR& operator() (const VECTOR<int, dim>& node) {
        if constexpr (dim == 2) return (*this)(node(0), node(1));
        else return (*this)(node(0), node(1), node(2));
    }
    JGSL_FORCE_INLINE ATTR operator() (int x, int y) const {
        uint64_t xmask = 0x5555555555555555UL;
        uint64_t ymask = 0xaaaaaaaaaaaaaaaaUL;
        uint64_t zorder = _pdep_u64((uint64_t)x, xmask) | _pdep_u64((uint64_t)y, ymask);
        uint64_t high_bits = (zorder >> (unsigned)6);
        uint64_t low_bits = (zorder & (unsigned)63);
        auto it = mapping.find(high_bits);
        if (it == mapping.end()) return ATTR();
        return data[it->second].data[low_bits];
    }
    JGSL_FORCE_INLINE ATTR operator() (int x, int y, int z) const {
        uint64_t xmask = 0x9249249249249249UL;
        uint64_t ymask = 0x2492492492492492UL;
        uint64_t zmask = 0x4924924924924924UL;
        uint64_t zorder = _pdep_u64((uint64_t)x, xmask) | _pdep_u64((uint64_t)y, ymask) | _pdep_u64((uint64_t)z, zmask);
        uint64_t high_bits = (zorder >> (unsigned)6);
        uint64_t low_bits = (zorder & (unsigned)63);
        auto it = mapping.find(high_bits);
        if (it == mapping.end()) return ATTR();
        return data[it->second].data[low_bits];
    }
    JGSL_FORCE_INLINE ATTR operator() (const VECTOR<int, dim>& node) const {
        if constexpr (dim == 2) return (*this)(node(0), node(1));
        else return (*this)(node(0), node(1), node(2));
    }
    JGSL_FORCE_INLINE ATTR Get_Value(const VECTOR<int, dim>& node) const {
        return (*this)(node);
    }
    JGSL_FORCE_INLINE VECTOR<int, dim> Get_Node(uint64_t zorder) {
        if constexpr (dim == 2) {
            uint64_t xmask = 0x5555555555555555UL;
            uint64_t ymask = 0xaaaaaaaaaaaaaaaaUL;
            int x = (int)(_pext_u64(zorder, xmask) & 0x00000000FFFFFFFFUL);
            int y = (int)(_pext_u64(zorder, ymask) & 0x00000000FFFFFFFFUL);
            return VECTOR<int, dim>(x, y);
        } else {
            uint64_t xmask = 0x9249249249249249UL;
            uint64_t ymask = 0x2492492492492492UL;
            uint64_t zmask = 0x4924924924924924UL;
            int x = (int)(_pext_u64(zorder, xmask) | (0xFFFFFFFFFFC00000UL * ((zorder >> 63) & 1)));
            int y = (int)(_pext_u64(zorder, ymask) | (0xFFFFFFFFFFE00000UL * ((zorder >> 61) & 1)));
            int z = (int)(_pext_u64(zorder, zmask) | (0xFFFFFFFFFFE00000UL * ((zorder >> 62) & 1)));
            return VECTOR<int, dim>(x, y, z);
        }
    }

    JGSL_FORCE_INLINE void Iterate_Grid(std::function<void(const VECTOR<int, dim>&, ATTR&)> target) {
        for (int i = 0; i < data.size(); ++i) {
            uint64_t base_offset = (offsets[i] << (unsigned)6);
            BLOCK& b = data[i];
            for (uint64_t j = 0; j < 64; ++j)
                target(Get_Node(base_offset | j), b.data[j]);
        }
    }

    template <class T, int degree>
    JGSL_FORCE_INLINE void Iterate_Kernel(
        const BSPLINE_WEIGHTS<T, dim, degree>& spline,
        std::function<void(const VECTOR<int, dim>&, const T&, const VECTOR<T, dim>&, ATTR&)> target
    ) {
        T one_over_dx = spline.one_over_dx;
        auto& w = spline.w;
        auto& dw = spline.dw;
        const VECTOR<int, dim>& base_coord = spline.base_node;
        VECTOR<int, dim> coord;
        if constexpr (dim == 2) {
            for (int i = 0; i < degree + 1; ++i) {
                T wi = w[0](i);
                T dwidxi = one_over_dx * dw[0](i);
                coord(0) = base_coord(0) + i;
                for (int j = 0; j < degree + 1; ++j) {
                    T wj = w[1](j);
                    T wij = wi * wj;
                    T dwijdxi = dwidxi * wj;
                    T dwijdxj = wi * one_over_dx * dw[1](j);
                    coord(1) = base_coord(1) + j;
                    target(coord, wij, VECTOR<T, dim>{ dwijdxi, dwijdxj }, (*this)(coord(0), coord(1)));
                }
            }
        } else {
            for (int i = 0; i < degree + 1; ++i) {
                T wi = w[0](i);
                T dwidxi = one_over_dx * dw[0](i);
                coord(0) = base_coord(0) + i;
                for (int j = 0; j < degree + 1; ++j) {
                    T wj = w[1](j);
                    T wij = wi * wj;
                    T dwijdxi = dwidxi * wj;
                    T dwijdxj = wi * one_over_dx * dw[1](j);
                    coord(1) = base_coord(1) + j;
                    for (int k = 0; k < degree + 1; ++k) {
                        coord(2) = base_coord(2) + k;
                        T wk = w[2](k);
                        T wijk = wij * wk;
                        T wijkdxi = dwijdxi * wk;
                        T wijkdxj = dwijdxj * wk;
                        T wijkdxk = wij * one_over_dx * dw[2](k);
                        target(coord, wijk, VECTOR<T, dim>{ wijkdxi, wijkdxj, wijkdxk }, (*this)(coord(0), coord(1), coord(2)));
                    }
                }
            }
        }
    }

    template <class T>
    JGSL_FORCE_INLINE void Iterate_Kernel (
        const LINEAR_WEIGHTS<T, dim>& linear,
        const ATTR& val)
    {
        T one_over_dx = linear.one_over_dx;
        VECTOR<T, dim> w = linear.w;
        const VECTOR<int, dim>& c = linear.base_node;
        if constexpr (dim == 2) {
            (*this)(c(0),c(1))      +=  w(0) * w(1) * val;
            (*this)(c(0),c(1)+1)    +=  w(0) * (1 - w(1)) * val;
            (*this)(c(0)+1,c(1))    +=  (1 - w(0)) * w(1) * val;
            (*this)(c(0)+1,c(1)+1)  +=  (1 - w(0)) * (1 - w(1)) * val;
        }
        else if constexpr (dim == 3) {
            (*this)(c(0),c(1),c(2))      +=  w(0) * w(1) * w(2) * val;
            (*this)(c(0),c(1)+1,c(2))    +=  w(0) * (1 - w(1)) * w(2) * val;
            (*this)(c(0)+1,c(1),c(2))    +=  (1 - w(0)) * w(1) * w(2) * val;
            (*this)(c(0)+1,c(1)+1,c(2))  +=  (1 - w(0)) * (1 - w(1)) * w(2) * val;
            (*this)(c(0),c(1),c(2)+1)      +=  w(0) * w(1) * (1 - w(2)) * val;
            (*this)(c(0),c(1)+1,c(2)+1)    +=  w(0) * (1 - w(1)) * (1 - w(2)) * val;
            (*this)(c(0)+1,c(1),c(2)+1)    +=  (1 - w(0)) * w(1) * (1 - w(2)) * val;
            (*this)(c(0)+1,c(1)+1,c(2)+1)  +=  (1 - w(0)) * (1 - w(1)) * (1 - w(2)) * val;
        }
    }

    template <class T>
    JGSL_FORCE_INLINE void Iterate_Kernel_APIC (
        const LINEAR_WEIGHTS<T, dim>& linear,
        const VECTOR <T, dim>& c,
        const VECTOR <T, dim>& p,
        const int axis, const T dx,
        const ATTR& val)
    {
        VECTOR<T, dim> w = linear.w;
        const VECTOR<int, dim>& n = linear.base_node;
        if constexpr (dim == 2) {
            if (axis == 0) {
                (*this)(n(0),n(1))     +=  w(0)*w(1)*(val + c.dot(VECTOR<T,2>(n(0)*dx, (n(1)+0.5)*dx) - p));
                (*this)(n(0),n(1)+1)   +=  w(0)*(1-w(1))*(val + c.dot(VECTOR<T,2>(n(0)*dx, (n(1)+1.5)*dx) - p));
                (*this)(n(0)+1,n(1))   +=  (1-w(0))*w(1)*(val + c.dot(VECTOR<T,2>((n(0)+1)*dx, (n(1)+0.5)*dx) - p));
                (*this)(n(0)+1,n(1)+1) +=  (1-w(0))*(1-w(1))*(val + c.dot(VECTOR<T,2>((n(0)+1)*dx, (n(1)+1.5)*dx) - p));
            } else if (axis == 1) {
                (*this)(n(0),n(1))     +=  w(0)*w(1)*(val + c.dot(VECTOR<T,2>((n(0)+0.5)*dx, n(1)*dx) - p));
                (*this)(n(0)+1,n(1))   +=  (1-w(0))*w(1)*(val + c.dot(VECTOR<T,2>((n(0)+1.5)*dx, n(1)*dx) - p));
                (*this)(n(0),n(1)+1)   +=  w(0)*(1-w(1))*(val + c.dot(VECTOR<T,2>((n(0)+0.5)*dx, (n(1)+1)*dx) - p));
                (*this)(n(0)+1,n(1)+1) +=  (1-w(0))*(1-w(1))*(val + c.dot(VECTOR<T,2>((n(0)+1.5)*dx, (n(1)+1)*dx) - p));
            }
        }
        else if constexpr (dim == 3) {
            if (axis == 0) {
                (*this)(n(0),n(1),n(2))      +=  w(0)*w(1)*w(2)*(val+c.dot(VECTOR<T,3>(n(0)*dx,(n(1)+0.5)*dx,(n(2)+0.5)*dx) - p));
                (*this)(n(0),n(1)+1,n(2))    +=  w(0)*(1-w(1))*w(2)*(val+c.dot(VECTOR<T,3>(n(0)*dx,(n(1)+1.5)*dx,(n(2)+0.5)*dx) - p));
                (*this)(n(0)+1,n(1),n(2))    +=  (1-w(0))*w(1)*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+1)*dx,(n(1)+0.5)*dx,(n(2)+0.5)*dx) - p));
                (*this)(n(0)+1,n(1)+1,n(2))  +=  (1-w(0))*(1-w(1))*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+1)*dx,(n(1)+1.5)*dx,(n(2)+0.5)*dx) - p));
                (*this)(n(0),n(1),n(2)+1)      +=  w(0)*w(1)*(1-w(2))*(val+c.dot(VECTOR<T,3>(n(0)*dx,(n(1)+0.5)*dx,(n(2)+1.5)*dx) - p));
                (*this)(n(0),n(1)+1,n(2)+1)    +=  w(0)*(1-w(1))*(1-w(2))*(val+c.dot(VECTOR<T,3>(n(0)*dx,(n(1)+1.5)*dx,(n(2)+1.5)*dx) - p));
                (*this)(n(0)+1,n(1),n(2)+1)    +=  (1-w(0))*w(1)*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+1)*dx,(n(1)+0.5)*dx,(n(2)+1.5)*dx) - p));
                (*this)(n(0)+1,n(1)+1,n(2)+1)  +=  (1-w(0))*(1-w(1))*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+1)*dx,(n(1)+1.5)*dx,(n(2)+1.5)*dx) - p));
            }
            else if (axis == 1) {
                (*this)(n(0),n(1),n(2))      +=  w(0)*w(1)*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+0.5)*dx,(n(1)+0.5)*dx,n(2)*dx) - p));
                (*this)(n(0),n(1)+1,n(2))    +=  w(0)*(1-w(1))*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+0.5)*dx,(n(1)+1.5)*dx,n(2)*dx) - p));
                (*this)(n(0)+1,n(1),n(2))    +=  (1-w(0))*w(1)*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+1.5)*dx,(n(1)+0.5)*dx,n(2)*dx) - p));
                (*this)(n(0)+1,n(1)+1,n(2))  +=  (1-w(0))*(1-w(1))*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+1.5)*dx,(n(1)+1.5)*dx,n(2)*dx) - p));
                (*this)(n(0),n(1),n(2)+1)      +=  w(0)*w(1)*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+0.5)*dx,(n(1)+0.5)*dx,(n(2)+1)*dx) - p));
                (*this)(n(0),n(1)+1,n(2)+1)    +=  w(0)*(1-w(1))*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+0.5)*dx,(n(1)+1.5)*dx,(n(2)+1)*dx) - p));
                (*this)(n(0)+1,n(1),n(2)+1)    +=  (1-w(0))* w(1)*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+1.5)*dx,(n(1)+0.5)*dx,(n(2)+1)*dx) - p));
                (*this)(n(0)+1,n(1)+1,n(2)+1)  +=  (1-w(0))*(1-w(1))*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+1.5)*dx,(n(1)+1.5)*dx,(n(2)+1)*dx) - p));
            }
            else if (axis == 2) {
                (*this)(n(0),n(1),n(2))      +=  w(0)*w(1)*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+0.5)*dx,(n(1)+0.5)*dx,n(2)*dx) - p));
                (*this)(n(0),n(1)+1,n(2))    +=  w(0)*(1-w(1))*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+0.5)*dx,(n(1)+1.5)*dx,n(2)*dx) - p));
                (*this)(n(0)+1,n(1),n(2))    +=  (1-w(0))*w(1)*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+1.5)*dx,(n(1)+0.5)*dx,n(2)*dx) - p));
                (*this)(n(0)+1,n(1)+1,n(2))  +=  (1-w(0))*(1-w(1))*w(2)*(val+c.dot(VECTOR<T,3>((n(0)+1.5)*dx,(n(1)+1.5)*dx,n(2)*dx) - p));
                (*this)(n(0),n(1),n(2)+1)      +=  w(0)*w(1)*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+0.5)*dx,(n(1)+0.5)*dx,(n(2)+1)*dx) - p));
                (*this)(n(0),n(1)+1,n(2)+1)    +=  w(0)*(1-w(1))*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+0.5)*dx,(n(1)+1.5)*dx,(n(2)+1)*dx) - p));
                (*this)(n(0)+1,n(1),n(2)+1)    +=  (1-w(0))*w(1)*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+1.5)*dx,(n(1)+0.5)*dx,(n(2)+1)*dx) - p));
                (*this)(n(0)+1,n(1)+1,n(2)+1)  +=  (1-w(0))*(1-w(1))*(1-w(2))*(val+c.dot(VECTOR<T,3>((n(0)+1.5)*dx,(n(1)+0.5)*dx,(n(2)+1)*dx) - p));
            }
        }
    }

    template<class T>
    JGSL_FORCE_INLINE void get_from_Kernel(
        const LINEAR_WEIGHTS<T, dim>& linear, ATTR& val)
    {
        if constexpr (dim == 2) {
            auto& w = linear.w;
            const VECTOR<int, dim>& c = linear.base_node;
            ATTR y_0 = w(0) * (*this)(c(0),c(1)) + (1-w(0)) * (*this)(c(0)+1, c(1));
            ATTR y_1 = w(0) * (*this)(c(0),c(1)+1) + (1-w(0)) * (*this)(c(0)+1, c(1)+1);
            val =  w(1) * y_0 + (1 - w(1)) * y_1;
        }
        else if constexpr (dim == 3) {
            auto& w = linear.w;
            const VECTOR<int, dim>& c = linear.base_node;

            ATTR z_1 = w(2) * (*this)(c(0),c(1),c(2)) + (1-w(2)) * (*this)(c(0),c(1),c(2)+1);
            ATTR z_2 = w(2) * (*this)(c(0),c(1)+1,c(2)) + (1-w(2)) * (*this)(c(0),c(1)+1,c(2)+1);
            ATTR z_3 = w(2) * (*this)(c(0)+1,c(1),c(2)) + (1-w(2)) * (*this)(c(0)+1,c(1),c(2)+1);
            ATTR z_4 = w(2) * (*this)(c(0)+1,c(1)+1,c(2)) + (1-w(2)) * (*this)(c(0)+1,c(1)+1,c(2)+1);

            ATTR y_0 = w(1) * z_1 + (1 - w(1)) * z_2;
            ATTR y_1 = w(1) * z_3 + (1 - w(1)) * z_4;

            val =  w(0) * y_0 + (1 - w(0)) * y_1;
        }
    }

    template<class T>
    JGSL_FORCE_INLINE void get_gradient_from_Kernel(
        const LINEAR_WEIGHTS<T, dim>& linear,  VECTOR<ATTR, dim> &g)
    {
        if constexpr (dim == 2) {
            auto& w = linear.w;
            const VECTOR<int, dim>& c = linear.base_node;

            ATTR v00 = (*this)(c(0),c(1));
            ATTR v01 = (*this)(c(0),c(1)+1);
            ATTR v10 = (*this)(c(0)+1,c(1));
            ATTR v11 = (*this)(c(0)+1,c(1)+1);

            ATTR ddx0 = (*this)(c(0)+1,c(1)) - (*this)(c(0),c(1));
            ATTR ddx1 = (*this)(c(0)+1,c(1)+1) - (*this)(c(0),c(1)+1);

            ATTR ddy0 = (*this)(c(0),c(1)+1) - (*this)(c(0),c(1));
            ATTR ddy1 = (*this)(c(0)+1,c(1)+1) - (*this)(c(0)+1,c(1));

            g(0) = w(1) * ddx0 + (1-w(1)) * ddx1;
            g(1) = w(0) * ddy0 + (1-w(0)) * ddy1;
        }
        else if constexpr (dim == 3) {
            auto& w = linear.w;
            const VECTOR<int, dim>& c = linear.base_node;

            ATTR ddx0 = (*this)(c(0)+1,c(1),c(2)) - (*this)(c(0),c(1),c(2));
            ATTR ddx1 = (*this)(c(0)+1,c(1)+1,c(2)) - (*this)(c(0),c(1)+1,c(2));
            ATTR ddx2 = (*this)(c(0)+1,c(1),c(2)+1) - (*this)(c(0),c(1),c(2));
            ATTR ddx3 = (*this)(c(0)+1,c(1)+1,c(2)+1) - (*this)(c(0),c(1)+1,c(2)+1);

            ATTR ddy0 = (*this)(c(0),c(1)+1,c(2)) - (*this)(c(0),c(1),c(2));
            ATTR ddy1 = (*this)(c(0)+1,c(1)+1,c(2)) - (*this)(c(0)+1,c(1),c(2));
            ATTR ddy2 = (*this)(c(0),c(1)+1,c(2)+1) - (*this)(c(0),c(1),c(2)+1);
            ATTR ddy3 = (*this)(c(0)+1,c(1)+1,c(2)+1) - (*this)(c(0)+1,c(1),c(2)+1);

            ATTR ddz0 = (*this)(c(0),c(1),c(2)+1) - (*this)(c(0),c(1),c(2));
            ATTR ddz1 = (*this)(c(0)+1,c(1),c(2)+1) - (*this)(c(0)+1,c(1),c(2));
            ATTR ddz2 = (*this)(c(0),c(1)+1,c(2)+1) - (*this)(c(0),c(1+1),c(2));
            ATTR ddz3 = (*this)(c(0)+1,c(1)+1,c(2)+1) - (*this)(c(0)+1,c(1)+1,c(2));

            g(0) = w(2)*(w(1)*ddx0 + (1-w(1))*ddx1) + (1-w(2))*(w(1)*ddx2 + (1-w(1))*ddx3);
            g(1) = w(2)*(w(0)*ddy0 + (1-w(0))*ddy1) + (1-w(2))*(w(0)*ddy2 + (1-w(0))*ddy3);
            g(2) = w(1)*(w(0)*ddz0 + (1-w(0))*ddz1) + (1-w(1))*(w(0)*ddz2 + (1-w(0))*ddz3);
        }
    }

    template <class T, int degree> JGSL_FORCE_INLINE
    KERNEL_ITERATOR_HOLDER<ATTR, dim, T, degree> Iterate_Kernel(const BSPLINE_WEIGHTS<T, dim, degree>& spline) {
        return KERNEL_ITERATOR_HOLDER<ATTR, dim, T, degree>(*this, spline);
    }
};

template <class ATTR, class T, int degree>
struct KERNEL_ITERATOR<ATTR, 2, T, degree> {
    using Result = std::tuple<VECTOR<int, 2>, T, VECTOR<T, 2>, ATTR&>;

    SPARSE_GRID<ATTR, 2> &grid;
    const BSPLINE_WEIGHTS<T, 2, degree>& spline;
    int i, j;

    KERNEL_ITERATOR(SPARSE_GRID<ATTR, 2> &grid, const BSPLINE_WEIGHTS<T, 2, degree>& spline, int i, int j)
        : grid(grid), spline(spline), i(i), j(j) {}

    // Returns Coord, weight, weight_gradient, and Data at that Coord (mutable)
    Result operator*() {
        T wij = spline.w[0](i) * spline.w[1](j);
        T dwijdxi = spline.dw[0](i) * spline.w[1](j);
        T dwijdxj = spline.w[0](i) * spline.dw[1](j);
        VECTOR<T, 2> w_grad = VECTOR<T, 2>(dwijdxi, dwijdxj) * spline.one_over_dx;
        VECTOR<int, 2> coord(spline.base_node(0) + i, spline.base_node(1) + j);
        return Result(coord, wij, w_grad, grid(coord(0), coord(1)));
    }

    void operator++() {
        if (i < degree) {
            i++;
        } else {
            i = 0;
            j++;
        }
    }

    bool operator!=(const KERNEL_ITERATOR<ATTR, 2, T, degree> &other) {
        return i != other.i || j != other.j;
    }
};

template <class ATTR, class T, int degree>
struct KERNEL_ITERATOR<ATTR, 3, T, degree> {
    using Result = std::tuple<VECTOR<int, 3>, T, VECTOR<T, 3>, ATTR&>;

    SPARSE_GRID<ATTR, 3> &grid;
    const BSPLINE_WEIGHTS<T, 3, degree>& spline;
    int i, j, k;

    KERNEL_ITERATOR(SPARSE_GRID<ATTR, 3> &grid, const BSPLINE_WEIGHTS<T, 3, degree>& spline, int i, int j, int k)
        : grid(grid), spline(spline), i(i), j(j), k(k) {}

    Result operator*() {
        T wijk = spline.w[0](i) * spline.w[1](j) * spline.w[2](k);
        T wijkdxi = spline.dw[0](i) * spline.w[1](j) * spline.w[2](k);
        T wijkdxj = spline.w[0](i) * spline.dw[1](j) * spline.w[2](k);
        T wijkdxk = spline.w[0](i) * spline.w[1](j) * spline.dw[2](k);
        VECTOR<T, 3> w_grad = VECTOR<T, 3>(wijkdxi, wijkdxj, wijkdxk) * spline.one_over_dx;
        VECTOR<int, 3> coord = spline.base_node + VECTOR<int, 3>(i, j, k);
        return Result(coord, wijk, w_grad, grid(coord(0), coord(1), coord(2)));
    }

    void operator++() {
        if (i < degree) {
            i++;
        } else {
            i = 0;
            if (j < degree) {
                j++;
            } else {
                j = 0;
                k++;
            }
        }
    }

    bool operator!=(const KERNEL_ITERATOR<ATTR, 3, T, degree> &other) {
        return i != other.i || j != other.j || k != other.k;
    }
};

template <class ATTR, class T, int degree>
struct KERNEL_ITERATOR_HOLDER<ATTR, 2, T, degree> {
    SPARSE_GRID<ATTR, 2> &grid;
    const BSPLINE_WEIGHTS<T, 2, degree>& spline;

    KERNEL_ITERATOR_HOLDER(SPARSE_GRID<ATTR, 2> &grid, const BSPLINE_WEIGHTS<T, 2, degree>& spline)
        : grid(grid), spline(spline) {}

    KERNEL_ITERATOR<ATTR, 2, T, degree> begin() {
        return KERNEL_ITERATOR<ATTR, 2, T, degree>(grid, spline, 0, 0);
    }

    KERNEL_ITERATOR<ATTR, 2, T, degree> end() {
        return KERNEL_ITERATOR<ATTR, 2, T, degree>(grid, spline, 0, degree + 1);
    }
};

template <class ATTR, class T, int degree>
struct KERNEL_ITERATOR_HOLDER<ATTR, 3, T, degree> {
    SPARSE_GRID<ATTR, 3> &grid;
    const BSPLINE_WEIGHTS<T, 3, degree>& spline;

    KERNEL_ITERATOR_HOLDER(SPARSE_GRID<ATTR, 3> &grid, const BSPLINE_WEIGHTS<T, 3, degree>& spline)
        : grid(grid), spline(spline) {}

    KERNEL_ITERATOR<ATTR, 3, T, degree> begin() {
        return KERNEL_ITERATOR<ATTR, 3, T, degree>(grid, spline, 0, 0, 0);
    }

    KERNEL_ITERATOR<ATTR, 3, T, degree> end() {
        return KERNEL_ITERATOR<ATTR, 3, T, degree>(grid, spline, 0, 0, degree + 1);
    }
};

//#####################################################################
template <class T, int _dim, int dim>
void Export_Sparse_Grid_Impl(py::module& m) {
    auto suffix = std::string("_") + (_dim == 2 ? "2" : "3") + (dim == 2 ? "2" : "3") + (std::is_same<T, float>::value ? "F" : "D");
    py::class_<SPARSE_GRID<VECTOR<T, _dim>, dim>>(m, ("Grid" + suffix).c_str())
        .def(py::init<>())
        .def(py::self -= py::self)
        .def(py::init<>()).def("copy", &SPARSE_GRID<VECTOR<T, _dim>, dim>::copy);
}

void Export_Sparse_Grid(py::module& m) {
    Export_Sparse_Grid_Impl<double, 3, 2>(m);
    Export_Sparse_Grid_Impl<double, 4, 3>(m);

    //Export_Sparse_Grid_Impl<int, 4, 3>(m);
    //Export_Sparse_Grid_Impl<int, 3, 2>(m);
    py::class_<SPARSE_GRID<VECTOR<int, 2>, 2>>(m, "Vector2iGrid2");
    py::class_<SPARSE_GRID<VECTOR<int, 3>, 3>>(m, "Vector3iGrid3");

    py::class_<SPARSE_GRID<double, 2>>(m, "dGrid2");
    py::class_<SPARSE_GRID<double, 3>>(m, "dGrid3");

    py::class_<SPARSE_GRID<float, 2>>(m, "fGrid2");
    py::class_<SPARSE_GRID<float, 3>>(m, "fGrid3");

    py::class_<SPARSE_GRID<int, 2>>(m, "iGrid2");
    py::class_<SPARSE_GRID<int, 3>>(m, "iGrid3");
}

}