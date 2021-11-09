#pragma once

//#####################################################################
// Helper Function Polar Decomposition 2d and 3d
//#####################################################################

#include <Math/VECTOR.h>
#include <Math/MATH_TOOLS.h>
#include <Math/GIVEN_ROTATION.h>
#include <Math/SINGULAR_VALUE_DECOMPOSITION.h>

namespace JGSL {

/**
   \brief 1x1 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix
   \param[out] S_Sym Symmetric.

   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
template <class T>
void polarDecomposition(const MATRIX<T, 1>& A, MATRIX<T ,1>& R, MATRIX<T, 1>& S_Sym)
{
    S_Sym = A;
    R(0, 0) = 1;
}

/**
   \brief 2x2 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix in givens form
   \param[out] S_Sym Symmetric. Whole matrix is stored

   Whole matrix S is stored since its faster to calculate due to simd vectorization
   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
template <class T>
void polarDecomposition(const MATRIX<T, 2>& A,
                        GIVENS_ROTATION<T>& R,
                        const MATRIX<T, 2>& S_Sym)
{
    VECTOR<T, 2> x(A(0, 0) + A(1, 1), A(1, 0) - A(0, 1));
    T denominator = x.length();
    R.c = (T)1;
    R.s = (T)0;
    if (denominator != 0) {
        /*
          No need to use a tolerance here because x(0) and x(1) always have
          smaller magnitude then denominator, therefore overflow never happens.
        */
        R.c = x(0) / denominator;
        R.s = -x(1) / denominator;
    }
    auto& S = const_cast<MATRIX<T, 2>&>(S_Sym);
    S = A;
    R.rowRotation(S);
}

/**
   \brief 2x2 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix.
   \param[out] S_Sym Symmetric. Whole matrix is stored

   Whole matrix S is stored since its faster to calculate due to simd vectorization
   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
template <class T>
void polarDecomposition(const MATRIX<T, 2>& A,
                        const MATRIX<T, 2>& R,
                        const MATRIX<T, 2>& S_Sym)
{
    GIVENS_ROTATION<T> r(0, 1);
    polarDecomposition(A, r, S_Sym);
    r.fill(R);
}

/**
   \brief 3X3 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix.
   \param[out] S_Sym Symmetric. Whole matrix is stored

   Whole matrix S is stored
   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
template <class T>
void polarDecomposition(const MATRIX<T, 3>& A,
                        MATRIX<T, 3>& R,
                        MATRIX<T, 3>& S_Sym)
{
    MATRIX<T, 3> U;
    VECTOR<T, 3> sigma;
    MATRIX<T, 3> V;

    Singular_Value_Decomposition(A, U, sigma, V);
    R = U * V.transpose();
    S_Sym = V * MATRIX<T, 3>(sigma) * V.transpose();
}

}