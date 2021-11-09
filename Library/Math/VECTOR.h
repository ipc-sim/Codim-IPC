#pragma once
//#####################################################################
// Class VECTOR
// Class MATRIX
// Class SCALAR
//#####################################################################
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <Eigen/Eigen>
#include <immintrin.h>
#include <functional>

namespace py = pybind11;
namespace JGSL {

#ifdef _WIN64
#define JGSL_FORCE_INLINE __forceinline
#else
#define JGSL_FORCE_INLINE inline __attribute__((always_inline))
#endif

template<class T> struct SIMD_SWITCHER {};
template<> struct SIMD_SWITCHER<int> { typedef __m128i SIMD_TYPE; };
template<> struct SIMD_SWITCHER<float> { typedef __m128 SIMD_TYPE; };
template<> struct SIMD_SWITCHER<double> { typedef __m256d SIMD_TYPE; };

//#####################################################################
// Class VECTOR
//
// TODO: check correctness of alignment by
// https://stackoverflow.com/questions/6786138/structure-alignment-in-gcc-should-alignment-be-specified-in-typedef/12014377
//#####################################################################
template <class T, int dim>
class __attribute__ ((aligned(sizeof(T) << 2))) VECTOR {
    using SIMD_TYPE = typename SIMD_SWITCHER<T>::SIMD_TYPE;

public:
    using SCALAR_TYPE = T;
    union {
        SIMD_TYPE v;
        T data[4];
        struct { T x, y, z, w; };
    };
    explicit JGSL_FORCE_INLINE VECTOR() : x(0), y(0), z(0), w(0) {}
    explicit JGSL_FORCE_INLINE VECTOR(T x) : x(x), y(x), z(x), w(x) {}
    explicit JGSL_FORCE_INLINE VECTOR(T x, T y) : x(x), y(y), z(0), w(0) {}
    explicit JGSL_FORCE_INLINE VECTOR(T x, T y, T z) : x(x), y(y), z(z), w(0) {}
    explicit JGSL_FORCE_INLINE VECTOR(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    explicit JGSL_FORCE_INLINE VECTOR(SIMD_TYPE v) : v(v) {}
    explicit JGSL_FORCE_INLINE VECTOR(const VECTOR<T, dim - 1> &o, T extra) {
        for (int d = 0; d < dim - 1; ++d) data[d] = o.data[d];
        data[dim - 1] = extra;
    }
    template <int dim_>
    explicit JGSL_FORCE_INLINE VECTOR(const VECTOR<T, dim_> &o) {
        for (int d = 0; d < std::min(dim, dim_); ++d) data[d] = o.data[d];
    }
    explicit JGSL_FORCE_INLINE VECTOR(const Eigen::Matrix<T, dim, 1> &o) {
        for (int d = 0; d < dim; ++d) data[d] = o(d);
    }
    template <class T_>
    JGSL_FORCE_INLINE VECTOR<T_, dim> cast() const {
        VECTOR<T_, dim> v;
        for (int d = 0; d < dim; ++d) v.data[d] = (T_)data[d];
        return v;
    }



    JGSL_FORCE_INLINE T& operator() (int d) { return data[d]; }
    JGSL_FORCE_INLINE const T& operator() (int d) const { return data[d]; }
    JGSL_FORCE_INLINE T& operator[] (int d) { return data[d]; }
    JGSL_FORCE_INLINE const T& operator[] (int d) const { return data[d]; }

    JGSL_FORCE_INLINE VECTOR<T, dim> operator-() const {
        return VECTOR<T, dim>() - (*this);
    }
    JGSL_FORCE_INLINE VECTOR<T, dim> operator+(const VECTOR<T, dim>& o) const {
        if constexpr (std::is_same<T, float>::value) return VECTOR<T, dim>(_mm_add_ps(v, o.v));
        if constexpr (std::is_same<T, double>::value) return VECTOR<T, dim>(_mm256_add_pd(v, o.v));
        if constexpr (std::is_same<T, int>::value)  return VECTOR<T, dim>(_mm_add_epi32(v, o.v));
    }
    JGSL_FORCE_INLINE VECTOR<T, dim> operator-(const VECTOR<T, dim>& o) const {
        if constexpr (std::is_same<T, float>::value) return VECTOR<T, dim>(_mm_sub_ps(v, o.v));
        if constexpr (std::is_same<T, double>::value) return VECTOR<T, dim>(_mm256_sub_pd(v, o.v));
        if constexpr (std::is_same<T, int>::value)  return VECTOR<T, dim>(_mm_sub_epi32(v, o.v));
    }
    JGSL_FORCE_INLINE VECTOR<T, dim> operator*(T a) const {
        if constexpr (std::is_same<T, float>::value)
            return VECTOR<T, dim>(_mm_mul_ps(v, _mm_set_ps1(a)));
        else
            return VECTOR<T, dim>(_mm256_mul_pd(v, _mm256_set1_pd(a)));
    }
    JGSL_FORCE_INLINE VECTOR<T, dim> operator/(T a) const {
        if constexpr (std::is_same<T, float>::value)
            return VECTOR<T, dim>(_mm_div_ps(v, _mm_set_ps1(a)));
        else
            return VECTOR<T, dim>(_mm256_div_pd(v, _mm256_set1_pd(a)));
    }

    JGSL_FORCE_INLINE VECTOR<T, dim> operator-(const T& a) const {
        VECTOR<T, dim> o;
        for(size_t d=0;d<dim;d++) o(d)=data[d]-a;
        return o;
    }

    JGSL_FORCE_INLINE VECTOR<T, dim>& operator+=(const VECTOR<T, dim>& o) { (*this) = (*this) + o; return *this; }
    JGSL_FORCE_INLINE VECTOR<T, dim>& operator-=(const VECTOR<T, dim>& o) { (*this) = (*this) - o; return *this; }
    JGSL_FORCE_INLINE VECTOR<T, dim>& operator*=(T o) { (*this) = (*this) * o; return *this; }
    JGSL_FORCE_INLINE VECTOR<T, dim>& operator/=(T o) { (*this) = (*this) / o; return *this; }

    JGSL_FORCE_INLINE VECTOR<T, dim>& operator+=(const T* o) { 
        for (int d = 0; d < dim; ++d) {
            data[d] += o[d];
        }
        return *this; 
    }

    JGSL_FORCE_INLINE bool operator==(const VECTOR<T, dim>& o) const{ 
        for (int d = 0; d < dim; ++d) if ((*this)(d) != o(d)) return false;
        return true;
    }

    JGSL_FORCE_INLINE bool operator<(const VECTOR<T, dim>& o) const{ 
        for (int d = 0; d < dim; ++d) {
            if ((*this)(d) < o(d)) {
                return true;
            }
            else if ((*this)(d) > o(d)) {
                return false;
            }
        }
        return false;
    }

    JGSL_FORCE_INLINE T dot(const VECTOR<T, dim>& o) const {
        if constexpr (std::is_same<T, float>::value) {
            if constexpr (dim == 1) return _mm_cvtss_f32(_mm_dp_ps(v, o.v, 0x11));
            if constexpr (dim == 2) return _mm_cvtss_f32(_mm_dp_ps(v, o.v, 0x31));
            if constexpr (dim == 3) return _mm_cvtss_f32(_mm_dp_ps(v, o.v, 0x71));
            if constexpr (dim == 4) return _mm_cvtss_f32(_mm_dp_ps(v, o.v, 0xF1));
        } else {
            VECTOR<T, dim> p(_mm256_mul_pd(v, o.v));
            if constexpr (dim == 1) return p(0);
            if constexpr (dim == 2) return p(0) + p(1);
            if constexpr (dim == 3) return p(0) + p(1) + p(2);
            if constexpr (dim == 4) return p(0) + p(1) + p(2) + p(3);
        }
    }
    JGSL_FORCE_INLINE T length2() const { return this->dot(*this); }
    JGSL_FORCE_INLINE T length() const { return std::sqrt(this->length2()); }
    JGSL_FORCE_INLINE T norm() const { return std::sqrt(this->length2()); }
    JGSL_FORCE_INLINE T prod() const {
        T result = this->data[0];
        for (int i = 1; i < dim; ++i) {
            result *= this->data[i];
        }
        return result;
    }

    JGSL_FORCE_INLINE VECTOR<T, dim> sqrt() {
        VECTOR<T, dim> result;
        for (int i = 0; i < dim; i++) {
            result(i) = std::sqrt(data[i]);
        }
        return result;
    }

    JGSL_FORCE_INLINE T max() {
        if constexpr (dim == 1) return data[0];
        if constexpr (dim == 2) return std::max(data[0], data[1]);
        if constexpr (dim == 3) return std::max(std::max(data[0], data[1]), data[2]);
        if constexpr (dim == 4) return std::max(std::max(data[0], data[1]), std::max(data[2], data[3]));
    }


    JGSL_FORCE_INLINE VECTOR<T, dim> log() const {
        VECTOR<T, dim> v;
        for (int d = 0; d < dim; ++d) v(d) = std::log(data[d]);
        return v;
    }

    JGSL_FORCE_INLINE VECTOR<T, dim> square() const {
        VECTOR<T, dim> v;
        for (int d = 0; d < dim; ++d) v(d) = data[d] * data[d];
        return v;
    }

    JGSL_FORCE_INLINE VECTOR<T, dim> inverse() const {
        VECTOR<T, dim> v;
        for (int d = 0; d < dim; ++d) v(d) = (T)1. / data[d];
        return v;
    }

    JGSL_FORCE_INLINE VECTOR<T, dim> abs() {
        VECTOR<T, dim> v;
        for (int d = 0; d < dim; ++d) v(d) = std::abs(data[d]);
        return v;
    }

    JGSL_FORCE_INLINE static VECTOR<T,dim> Unit_Vector(const int d) {
        VECTOR<T,dim> v;
        v(d)=1;
        return v;
    }

    JGSL_FORCE_INLINE static VECTOR<T,dim> Ones_Vector() {
        VECTOR<T,dim> v;
        for (int d = 0; d < dim; ++d) v(d)=1;
        return v;
    }

    JGSL_FORCE_INLINE VECTOR<T,dim> Normalized() {
        T len = length();
        if(len)return *this/len;
        else return Unit_Vector(0);
    }

    JGSL_FORCE_INLINE T Sum() const {
        if constexpr (dim == 1) return data[0];
        if constexpr (dim == 2) return data[0]+data[1];
        if constexpr (dim == 3) return data[0]+data[1]+data[2];
        if constexpr (dim == 4) return data[0]+data[1]+data[2]+data[3];
    }

    JGSL_FORCE_INLINE T Average() const {
        return Sum()/(T)dim;
    }

    JGSL_FORCE_INLINE void setZero() {
        if constexpr (std::is_same<T, float>::value)
            v = _mm_set_ps1(0);
        else
            v = _mm256_set1_pd(0);
    }

    JGSL_FORCE_INLINE Eigen::Matrix<T, dim, 1> to_eigen() {
        Eigen::Matrix<T, dim, 1> v;
        for (int i = 0; i < dim; ++i) v(i) = (*this)(i);
        return v;
    }
};
template <class T, int dim> JGSL_FORCE_INLINE VECTOR<T, dim> operator*(T a, const VECTOR<T, dim>& o) { return o * a; }
template <class T> JGSL_FORCE_INLINE T det(const VECTOR<T, 2>& col0, const VECTOR<T, 2>& col1) { 
    return col0[0] * col1[1] - col0[1] * col1[0];
}

//#####################################################################
// Class MATRIX
//#####################################################################
template <class T, int dim>
class MATRIX {
public:
    using SCALAR_TYPE = typename VECTOR<T, dim>::SCALAR_TYPE;
    VECTOR<T, dim> data[dim];
    explicit JGSL_FORCE_INLINE MATRIX() {
        for (int d = 0; d < dim; ++d)
            data[d] = VECTOR<T, dim>(0);
    }
    explicit JGSL_FORCE_INLINE MATRIX(T x) {
        for (int d = 0; d < dim; ++d) {
            data[d] = VECTOR<T, dim>(0);
            data[d](d) = x;
        }
    }
    explicit JGSL_FORCE_INLINE MATRIX(const VECTOR<T, dim>& v) {
        for (int d = 0; d < dim; ++d) {
            data[d] = VECTOR<T, dim>(0);
            data[d](d) = v(d);
        }
    }
    template <int dim_>
    explicit JGSL_FORCE_INLINE MATRIX(const MATRIX<T, dim_>& m) {
        for (int i = 0; i < std::min(dim, dim_); ++i)
            for (int j = 0; j < std::min(dim, dim_); ++j)
                data[i](j) = m.data[i](j);
    }
    explicit JGSL_FORCE_INLINE MATRIX(const Eigen::Matrix<T, dim, 1> &o) {
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) (*this)(i, j) = o(i, j);
    }

    JGSL_FORCE_INLINE VECTOR<T, dim>& operator() (int j) { return data[j]; }
    JGSL_FORCE_INLINE T& operator() (int i, int j) { return data[j](i); }
    JGSL_FORCE_INLINE const T& operator() (int i, int j) const { return data[j](i); }

    JGSL_FORCE_INLINE static MATRIX<T,dim> Ones_Matrix() {
        MATRIX<T, dim> m;
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) m(i, j) = (T)1;
        return m;
    }

    template <class T_>
    JGSL_FORCE_INLINE MATRIX<T_, dim> cast() const {
        MATRIX<T_, dim> M;
        for (int j = 0; j < dim; ++j) {
            for (int i = 0; i < dim; ++i) {
                M.data[j](i) = (T_)data[j](i);
            }
        }
        return M;
    }

    JGSL_FORCE_INLINE MATRIX<T, dim> operator+(const MATRIX<T, dim>& o) const { MATRIX<T, dim> m; for (int d = 0; d < dim; ++d) m.data[d] = data[d] + o.data[d]; return m; }
    JGSL_FORCE_INLINE MATRIX<T, dim> operator-(const MATRIX<T, dim>& o) const { MATRIX<T, dim> m; for (int d = 0; d < dim; ++d) m.data[d] = data[d] - o.data[d]; return m; }
    JGSL_FORCE_INLINE VECTOR<T, dim> operator*(const VECTOR<T, dim>& o) const {
        if constexpr (std::is_same<T, float>::value) {
            auto tmp = _mm_permute_ps(o.v, 0x55 * 0);
            auto ret = _mm_mul_ps(data[0].v, tmp);
            if constexpr (dim >= 2) { tmp = _mm_permute_ps(o.v, 0x55 * 1); ret = _mm_fmadd_ps(data[1].v, tmp, ret); }
            if constexpr (dim >= 3) { tmp = _mm_permute_ps(o.v, 0x55 * 2); ret = _mm_fmadd_ps(data[2].v, tmp, ret); }
            if constexpr (dim >= 4) { tmp = _mm_permute_ps(o.v, 0x55 * 3); ret = _mm_fmadd_ps(data[3].v, tmp, ret); }
            return VECTOR<T, dim>(ret);
        } else {
            auto tmp = _mm256_permute4x64_pd(o.v, 0x55 * 0);
            auto ret = _mm256_mul_pd(data[0].v, tmp);
            if constexpr (dim >= 2) { tmp = _mm256_permute4x64_pd(o.v, 0x55 * 1); ret = _mm256_fmadd_pd(data[1].v, tmp, ret); }
            if constexpr (dim >= 3) { tmp = _mm256_permute4x64_pd(o.v, 0x55 * 2); ret = _mm256_fmadd_pd(data[2].v, tmp, ret); }
            if constexpr (dim >= 4) { tmp = _mm256_permute4x64_pd(o.v, 0x55 * 3); ret = _mm256_fmadd_pd(data[3].v, tmp, ret); }
            return VECTOR<T, dim>(ret);
        }
    }
    JGSL_FORCE_INLINE MATRIX<T, dim> operator*(const MATRIX<T, dim>& o) const { MATRIX<T, dim> m; for (int d = 0; d < dim; ++d) m.data[d] = (*this) * o.data[d]; return m; }
    JGSL_FORCE_INLINE MATRIX<T, dim> operator*(T a) const { MATRIX<T, dim> m; for (int d = 0; d < dim; ++d) m.data[d] = data[d] * a; return m; }
    JGSL_FORCE_INLINE MATRIX<T, dim> operator/(T a) const { MATRIX<T, dim> m; for (int d = 0; d < dim; ++d) m.data[d] = data[d] / a; return m; }

    JGSL_FORCE_INLINE MATRIX<T, dim>& operator+=(const MATRIX<T, dim>& o) { for (int d = 0; d < dim; ++d) data[d] += o.data[d]; return *this; }
    template<class MAT>
    JGSL_FORCE_INLINE MATRIX<T, dim>& operator+=(const MAT& o) { for (int d1 = 0; d1 < dim; ++d1) { for (int d2 = 0; d2 < dim; ++d2) { (*this)(d1, d2) += o(d1, d2); }} return *this; }
    JGSL_FORCE_INLINE MATRIX<T, dim>& operator-=(const MATRIX<T, dim>& o) { for (int d = 0; d < dim; ++d) data[d] -= o.data[d]; return *this; }
    JGSL_FORCE_INLINE MATRIX<T, dim>& operator*=(T a) { for (int d = 0; d < dim; ++d) data[d] *= a; return *this; }
    JGSL_FORCE_INLINE MATRIX<T, dim>& operator/=(T a) { for (int d = 0; d < dim; ++d) data[d] /= a; return *this; }

    JGSL_FORCE_INLINE T length2() const {
        T result = 0;
        for (int i = 0; i < dim; ++i)
            result += data[i].length2();
        return result;
    }

    JGSL_FORCE_INLINE MATRIX<T, dim> transpose() const {
        MATRIX<T, dim> m;
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) m(i, j) = (*this)(j, i);
        return m;
    }

    JGSL_FORCE_INLINE T determinant() const {
        if constexpr (dim == 2) return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        else return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(2, 1) * (*this)(1, 2)) -
                    (*this)(1, 0) * ((*this)(0, 1) * (*this)(2, 2) - (*this)(2, 1) * (*this)(0, 2)) +
                    (*this)(2, 0) * ((*this)(0, 1) * (*this)(1, 2) - (*this)(1, 1) * (*this)(0, 2));
    }

    JGSL_FORCE_INLINE MATRIX<T, dim> cofactor() const {
        MATRIX<T, dim> m;
        if constexpr (dim == 2) { m(0, 0) = (*this)(1, 1); m(1, 0) = -(*this)(0, 1); m(0, 1) = -(*this)(1, 0); m(1, 1) = (*this)(0, 0);}
        else {
            m(0, 0) = (*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1);
            m(0, 1) = (*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2);
            m(0, 2) = (*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0);
            m(1, 0) = (*this)(0, 2) * (*this)(2, 1) - (*this)(0, 1) * (*this)(2, 2);
            m(1, 1) = (*this)(0, 0) * (*this)(2, 2) - (*this)(0, 2) * (*this)(2, 0);
            m(1, 2) = (*this)(0, 1) * (*this)(2, 0) - (*this)(0, 0) * (*this)(2, 1);
            m(2, 0) = (*this)(0, 1) * (*this)(1, 2) - (*this)(0, 2) * (*this)(1, 1);
            m(2, 1) = (*this)(0, 2) * (*this)(1, 0) - (*this)(0, 0) * (*this)(1, 2);
            m(2, 2) = (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        }
        return m;
    }

    JGSL_FORCE_INLINE MATRIX<T, dim> inverse() const {
        MATRIX<T, dim> m = (*this);
        m.invert();
        return m;
    }

    JGSL_FORCE_INLINE void invert() {
        T det = determinant();
        if (det == 0.0) {
            puts("matrix noninvertible!");
            exit(-1);
        }

        if constexpr (dim == 2) { 
            T temp = (*this)(0, 0);
            (*this)(0, 0) = (*this)(1, 1);
            (*this)(1, 1) = temp;
            (*this)(0, 1) = -(*this)(0, 1);
            (*this)(1, 0) = -(*this)(1, 0);
            (*this) /= det;
        }
        else {
            MATRIX<T, dim> m;
            m(0, 0) = (*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1);
            m(1, 0) = (*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2);
            m(2, 0) = (*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0);
            m(0, 1) = (*this)(0, 2) * (*this)(2, 1) - (*this)(0, 1) * (*this)(2, 2);
            m(1, 1) = (*this)(0, 0) * (*this)(2, 2) - (*this)(0, 2) * (*this)(2, 0);
            m(2, 1) = (*this)(0, 1) * (*this)(2, 0) - (*this)(0, 0) * (*this)(2, 1);
            m(0, 2) = (*this)(0, 1) * (*this)(1, 2) - (*this)(0, 2) * (*this)(1, 1);
            m(1, 2) = (*this)(0, 2) * (*this)(1, 0) - (*this)(0, 0) * (*this)(1, 2);
            m(2, 2) = (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
            (*this) = m / det;
        }
    }

    JGSL_FORCE_INLINE T trace() {
        if constexpr (dim == 1) return data[0](0);
        if constexpr (dim == 2) return data[0](0)+data[1](1);
        if constexpr (dim == 3) return data[0](0)+data[1](1)+data[2](2);
        if constexpr (dim == 4) return data[0](0)+data[1](1)+data[2](2)+data[3](3);
    }

    JGSL_FORCE_INLINE void setZero() {
        for (int i = 0; i < dim; ++i) {
            data[i].setZero();
        }
    }

    template<class T2>
    JGSL_FORCE_INLINE void copy(const MATRIX<T2, dim>& src) {
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                (*this)(i, j) = src(i, j);
            }
        }
    }

    JGSL_FORCE_INLINE Eigen::Matrix<T, dim, dim> to_eigen() {
        Eigen::Matrix<T, dim, dim> m;
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) m(i, j) = (*this)(i, j);
        return m;
    }

    JGSL_FORCE_INLINE void makePD(void) {
        if constexpr (dim == 2) {
            // based on http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/

            MATRIX<T, dim>& symMtr = *this;
            const T a = symMtr(0, 0);
            const T b = (symMtr(0, 1) + symMtr(1, 0)) / 2.0;
            const T d = symMtr(1, 1);

            T b2 = b * b;
            const T D = a * d - b2;
            const T T_div_2 = (a + d) / 2.0;
            const T sqrtTT4D = std::sqrt(T_div_2 * T_div_2 - D);
            const T L2 = T_div_2 - sqrtTT4D;
            if (L2 < 0.0) {
                const T L1 = T_div_2 + sqrtTT4D;
                if (L1 <= 0.0) {
                    symMtr.setZero();
                }
                else {
                    if (b2 == 0.0) {
                        symMtr.setZero();
                        symMtr(0, 0) = L1;
                    }
                    else {
                        const T L1md = L1 - d;
                        const T L1md_div_L1 = L1md / L1;
                        symMtr(0, 0) = L1md_div_L1 * L1md;
                        symMtr(0, 1) = symMtr(1, 0) = b * L1md_div_L1;
                        symMtr(1, 1) = b2 / L1;
                    }
                }
            }
        }
        else if (dim == 3) {
            MATRIX<double, dim> mtr;
            mtr.copy(*this);

            VECTOR<double, dim> eigVals;
            Get_Eigen_Values(mtr, eigVals);
            MATRIX<double, dim> eigVecs;
            Get_Eigen_Vectors(mtr, eigVals, eigVecs);

            if (eigVals[2] >= 0) {
                return;
            }

            for (int i = 2; i >= 0; --i) {
                if (eigVals[i] < 0.0) {
                    eigVals[i] = 0.0;
                }
                else {
                    break;
                }
            }
            MATRIX<double, dim> A = eigVecs;
            for (int i = 0; i < dim; ++i) {
                A(0, i) *= eigVals[i];
                A(1, i) *= eigVals[i];
                A(2, i) *= eigVals[i];
            }
            MATRIX<double, dim> B = A * eigVecs.transpose();
            this->copy(B);
        }
        else {
            puts("makePD dim > 3 to be implemented");
            exit(-1);
        }
    }
};
template <class T, int dim> JGSL_FORCE_INLINE MATRIX<T, dim> operator*(T a, const MATRIX<T, dim>& o) { return o * a; }
template <class T, int dim> JGSL_FORCE_INLINE MATRIX<T, dim> outer_product(const VECTOR<T, dim>& col, const VECTOR<T, dim>& row) { MATRIX<T, dim> m; for (int d = 0; d < dim; ++d) m.data[d] = col * row(d); return m; }
template <class T> JGSL_FORCE_INLINE VECTOR<T, 3> cross(const VECTOR<T, 3>& a, const VECTOR<T, 3>& b) { return VECTOR<T, 3>(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]); }

//#####################################################################
// Class SCALAR
//#####################################################################
template <class T>
class SCALAR {
    T data;
};

template <typename T>
constexpr std::string get_type_short_name();

template <>
std::string get_type_short_name<float>() {
    return "f";
}

template <>
std::string get_type_short_name<double>() {
    return "d";
}

template <>
std::string get_type_short_name<int>() {
    return "i";
}

template <class T, int dim>
void Iterate_Region(int l, int r, std::function<void(const VECTOR<int, dim>&)> target) {
    if constexpr (dim == 2) {
        for (int i = l; i <= r; ++i)
            for (int j = l; j <= r; ++j) target(VECTOR<int, dim>(i, j));
    } else {
        for (int i = l; i <= r; ++i)
            for (int j = l; j <= r; ++j)
                for (int k = l; k <= r; ++k) target(VECTOR<int, dim>(i, j, k));
    }
}

//#####################################################################
template <class T, int dim, typename Class>
void Register_Vector(py::module& m, Class& cls)
{
    cls.def(py::init<T>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * T())
        .def(py::self / T())
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= T())
        .def(py::self /= T())
        .def("__eq__", &VECTOR<T, dim>::operator==, py::is_operator())
        .def("__getitem__", [](const VECTOR<T, dim>& v, int i){ return v.data[i];})
        .def("dot", &VECTOR<T, dim>::dot)
        .def("length", &VECTOR<T, dim>::length)
        .def("length2", &VECTOR<T, dim>::length2)
        .def("normalized", &VECTOR<T, dim>::Normalized)
        .def("sum", &VECTOR<T, dim>::Sum)
        .def("average", &VECTOR<T, dim>::Average);
}

template <class T, int dim, typename Class>
void Register_Matrix(py::module& m, Class& cls)
{
    cls.def(py::init<T>())
            .def(py::init<VECTOR<T, dim>>())
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self * T())
            .def(py::self / T())
            .def(py::self += py::self)
            .def(py::self -= py::self)
            .def(py::self *= T())
            .def(py::self /= T())
            .def("__getitem__", [](const MATRIX<T, dim>& matrix, std::tuple<int, int> index){ return matrix.data[std::get<1>(index)](std::get<0>(index));})
            .def("transpose", &MATRIX<T, dim>::transpose)
            .def("determinant", &MATRIX<T, dim>::determinant)
            .def("invert", &MATRIX<T, dim>::invert)
            .def("cofactor", &MATRIX<T, dim>::cofactor);
}

void Export_Vector(py::module& m) {
    Register_Vector<float, 2>(m, py::class_<VECTOR<float, 2>>(m, "Vector2f").def(py::init<>()).def(py::init<float, float>()));
    Register_Vector<float, 3>(m, py::class_<VECTOR<float, 3>>(m, "Vector3f").def(py::init<>()).def(py::init<float, float, float>()));
    Register_Vector<float, 4>(m, py::class_<VECTOR<float, 4>>(m, "Vector4f").def(py::init<>()).def(py::init<float, float, float, float>()));

    Register_Matrix<float, 2>(m, py::class_<MATRIX<float, 2>>(m, "Matrix2f").def(py::init<>()));
    Register_Matrix<float, 3>(m, py::class_<MATRIX<float, 3>>(m, "Matrix3f").def(py::init<>()));
    py::class_<SCALAR<float>>(m, "Scalarf").def(py::init<>());

    Register_Vector<double, 2>(m, py::class_<VECTOR<double, 2>>(m, "Vector2d").def(py::init<>()).def(py::init<double, double>()));
    Register_Vector<double, 3>(m, py::class_<VECTOR<double, 3>>(m, "Vector3d").def(py::init<>()).def(py::init<double, double, double>()));
    Register_Vector<double, 4>(m, py::class_<VECTOR<double, 4>>(m, "Vector4d").def(py::init<>()).def(py::init<double, double, double, double>()));
    Register_Matrix<double, 2>(m, py::class_<MATRIX<double, 2>>(m, "Matrix2d").def(py::init<>()));
    Register_Matrix<double, 3>(m, py::class_<MATRIX<double, 3>>(m, "Matrix3d").def(py::init<>()));
    py::class_<SCALAR<double>>(m, "Scalard").def(py::init<>());

    py::class_<VECTOR<int, 2>>(m, "Vector2i").def(py::init<>()).def(py::init<int>()).def(py::init<int, int>());
    py::class_<VECTOR<int, 3>>(m, "Vector3i").def(py::init<>()).def(py::init<int>()).def(py::init<int, int, int>());
    py::class_<VECTOR<int, 4>>(m, "Vector4i").def(py::init<>()).def(py::init<int, int, int, int>());
    py::class_<SCALAR<int>>(m, "Scalari").def(py::init<>());
}

};