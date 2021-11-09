#pragma once

//#####################################################################
// Helper Function Given Rotation
//#####################################################################

#include <Math/VECTOR.h>
#include <Math/MATH_TOOLS.h>

namespace JGSL {
/**
    Class for givens rotation.
    Row rotation G*A corresponds to something like
    c -s  0
    ( s  c  0 ) A
    0  0  1
    Column rotation A G' corresponds to something like
    c -s  0
    A ( s  c  0 )
    0  0  1

    c and s are always computed so that
    ( c -s ) ( a )  =  ( * )
    s  c     b       ( 0 )

    Assume rowi<rowk.
*/
template <class T>
class GIVENS_ROTATION {
public:
    int rowi;
    int rowk;
    T c;
    T s;

    JGSL_FORCE_INLINE GIVENS_ROTATION(int rowi_in, int rowk_in)
            : rowi(rowi_in)
            , rowk(rowk_in)
            , c(1)
            , s(0)
    {
    }

    JGSL_FORCE_INLINE GIVENS_ROTATION(T a, T b, int rowi_in, int rowk_in)
            : rowi(rowi_in)
            , rowk(rowk_in)
    {
        compute(a, b);
    }

    ~GIVENS_ROTATION() {}

    JGSL_FORCE_INLINE void setIdentity()
    {
        c = 1;
        s = 0;
    }

    JGSL_FORCE_INLINE void transposeInPlace()
    {
        s = -s;
    }

    /**
        Compute c and s from a and b so that
        ( c -s ) ( a )  =  ( * )
        s  c     b       ( 0 )
        */
    JGSL_FORCE_INLINE void compute(const T a, const T b)
    {
        using std::sqrt;

        T d = a * a + b * b;
        c = 1;
        s = 0;
        T sqrtd = sqrt(d);
        //T t = MATH_TOOLS::rsqrt(d);
        if (sqrtd) {
            T t = 1 / sqrtd;
            c = a * t;
            s = -b * t;
        }
    }

    /**
        This function computes c and s so that
        ( c -s ) ( a )  =  ( 0 )
        s  c     b       ( * )
    */
    JGSL_FORCE_INLINE void computeUnconventional(const T a, const T b)
    {
        using std::sqrt;

        T d = a * a + b * b;
        c = 0;
        s = 1;
        T sqrtd = sqrt(d);
        //T t = MATH_TOOLS::rsqrt(d);
        if (sqrtd) {
            T t = 1 / sqrtd;
            s = a * t;
            c = b * t;
        }
    }

    /**
      Fill the R with the entries of this rotation
    */
    template <int dim>
    JGSL_FORCE_INLINE void fill(const MATRIX<T, dim>& R) const
    {
        MATRIX<T, dim>& A = const_cast<MATRIX<T, dim>&>(R);
        A = MATRIX<T, dim>(1);
        A(rowi, rowi) = c;
        A(rowk, rowi) = -s;
        A(rowi, rowk) = s;
        A(rowk, rowk) = c;
    }

    /**
        This function does something like Q^T A -> A
        [ c -s  0 ]
        [ s  c  0 ] A -> A
        [ 0  0  1 ]
        It only affects row i and row k of A.
    */
    template <int dim>
    JGSL_FORCE_INLINE void rowRotation(MATRIX<T, dim>& A) const
    {
        for (int j = 0; j < dim; j++) {
            T tau1 = A(rowi, j);
            T tau2 = A(rowk, j);
            A(rowi, j) = c * tau1 - s * tau2;
            A(rowk, j) = s * tau1 + c * tau2;
        }
        //not type safe :/
    }

    /**
        This function does something like A Q -> A
           [ c  s  0 ]
        A  [-s  c  0 ]  -> A
           [ 0  0  1 ]
        It only affects column i and column k of A.
     */
    template <int dim>
    JGSL_FORCE_INLINE void columnRotation(MATRIX<T, dim>& A) const
    {
        for (int j = 0; j < dim; j++) {
            T tau1 = A(j, rowi);
            T tau2 = A(j, rowk);
            A(j, rowi) = c * tau1 - s * tau2;
            A(j, rowk) = s * tau1 + c * tau2;
        }
        //not type safe :/
    }

    /**
      Multiply givens must be for same row and column
    **/
    JGSL_FORCE_INLINE void operator*=(const GIVENS_ROTATION<T>& A)
    {
        T new_c = c * A.c - s * A.s;
        T new_s = s * A.c + c * A.s;
        c = new_c;
        s = new_s;
    }

    /**
      Multiply givens must be for same row and column
    **/
    JGSL_FORCE_INLINE GIVENS_ROTATION<T> operator*(const GIVENS_ROTATION<T>& A) const
    {
        GIVENS_ROTATION<T> r(*this);
        r *= A;
        return r;
    }
};

}