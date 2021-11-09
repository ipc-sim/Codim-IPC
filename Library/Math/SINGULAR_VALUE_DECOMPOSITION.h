#pragma once

//#####################################################################
// Helper Singular Value Decomposition 2d and 3d
//#####################################################################

#include <Math/VECTOR.h>
#include <Math/MATH_TOOLS.h>
#include <Math/EIGEN.h>
#include <Math/GIVEN_ROTATION.h>
#include <Math/POLAR_DECOMPOSITION.h>

namespace JGSL {

/**
   \brief 2x2 SVD (singular value decomposition) A=USV'
   \param[in] A Input matrix.
   \param[out] U Robustly a rotation matrix in Givens form
   \param[out] Sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
   \param[out] V Robustly a rotation matrix in Givens form
*/

template <class T>
void Singular_Value_Decomposition(
        const MATRIX<T, 2>& A,
        GIVENS_ROTATION<T>& U,
        const VECTOR<T, 2>& Sigma,
        GIVENS_ROTATION<T>& V)
{
    using std::sqrt;
    auto& sigma = const_cast<VECTOR<T, 2>&>(Sigma);

    MATRIX<T, 2> S_Sym;
    polarDecomposition(A, U, S_Sym);
    T cosine, sine;
    T x = S_Sym(0, 0);
    T y = S_Sym(0, 1);
    T z = S_Sym(1, 1);
    T y2 = y * y;
    if (y2 == 0) {
        // S is already diagonal
        cosine = 1;
        sine = 0;
        sigma(0) = x;
        sigma(1) = z;
    } else {
        T tau = T(0.5) * (x - z);
        T w = sqrt(tau * tau + y2);
        // w > y > 0
        T t;
        if (tau > 0) {
            // tau + w > w > y > 0 ==> division is safe
            t = y / (tau + w);
        } else {
            // tau - w < -w < -y < 0 ==> division is safe
            t = y / (tau - w);
        }
        cosine = T(1) / sqrt(t * t + T(1));
        sine = -t * cosine;
        /*
          V = [cosine -sine; sine cosine]
          Sigma = V'SV. Only compute the diagonals for efficiency.
          Also utilize symmetry of S and don't form V yet.
        */
        T c2 = cosine * cosine;
        T csy = 2 * cosine * sine * y;
        T s2 = sine * sine;
        sigma(0) = c2 * x - csy + s2 * z;
        sigma(1) = s2 * x + csy + c2 * z;
    }
    // Sorting
    // Polar already guarantees negative sign is on the small magnitude singular value.
    if (sigma(0) < sigma(1)) {
        std::swap(sigma(0), sigma(1));
        V.c = -sine;
        V.s = cosine;
    }
    else {
        V.c = cosine;
        V.s = sine;
    }
    U *= V;
}

/**
   \brief 2x2 SVD (singular value decomposition) A=USV'
   \param[in] A Input matrix.
   \param[out] U Robustly a rotation matrix.
   \param[out] Sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
   \param[out] V Robustly a rotation matrix.
*/
template <class T>
void Singular_Value_Decomposition(
            const MATRIX<T, 2>& A,
            const MATRIX<T, 2>& U,
            const VECTOR<T, 2>& Sigma,
            const MATRIX<T, 2>& V)
{
    GIVENS_ROTATION<T> gv(0, 1);
    GIVENS_ROTATION<T> gu(0, 1);
    Singular_Value_Decomposition(A, gu, Sigma, gv);
    gu.fill(U);
    gv.fill(V);
}

/*
    3X3 SVD
*/

void Singular_Value_Decomposition_(
    const MATRIX<double, 3>&A,
    MATRIX<double, 3>&U,
    VECTOR<double, 3>&Sigma,
    MATRIX<double, 3> &V)
{
    using T = double;
    VECTOR<T, 3> lambda;

    MATRIX<T, 3> A_Sym = A.transpose() * A;
    Get_Eigen_Values(A_Sym, lambda);
    Get_Eigen_Vectors(A_Sym, lambda, V);

    // compute singular values
    if (lambda(2) < 0) {
        for (int i = 0; i < 3; i++) {
            lambda(i) = lambda(i) >= (double)0? lambda(i): (double)0;
        }
    }
    Sigma = lambda.sqrt();
    if (A.determinant() < 0) {
        Sigma(2) = -Sigma(2);
    }
    // compute singular vectors
    U(0) = A * V(0);
    T norm = U(0).norm();
    if (norm != 0) {
        T one_over_norm = (T)1 / norm;
        U(0) = U(0) * one_over_norm;
    } else {
        U(0) = VECTOR<T ,3>(1,0,0);
    }

    VECTOR<double, 3> v1_orthogonal = MATH_TOOLS::Get_Unit_Orthogonal<T>(U(0));
    MATRIX<double, 3> other_v;// 3 2
    other_v(0) = v1_orthogonal;
    other_v(1) = MATH_TOOLS::Get_Cross_Product<T>(U(0), v1_orthogonal);
    VECTOR<double, 2> w(other_v.transpose() * A * V(1));
    norm = w.norm();
    if (norm != 0) {
        T one_over_norm = (T)1 / norm;
        w = w * one_over_norm;
    } else {
        w = VECTOR<double, 2>(1, 0);
    }

    U(1) = VECTOR<double, 3>(other_v(0,0)*w(0) + other_v(0,1)*w(1), other_v(1,0)*w(0) + other_v(1,1)*w(1), other_v(2,0) * w(0)+other_v(2,1) * w(1));
    U(2) = MATH_TOOLS::Get_Cross_Product<T>(U(0), U(1));
}

template <class T>
void Singular_Value_Decomposition(
    const MATRIX<T, 3>& A,
    MATRIX<T, 3>& U,
    VECTOR<T, 3>& Sigma,
    MATRIX<T, 3>& V)
{
    VECTOR<double, 3> sd;

    MATRIX<double, 3> Ud;
    MATRIX<double, 3> Vd;
    Singular_Value_Decomposition_(A.template cast<double>(), Ud, sd, Vd);

    V = Vd.template cast<T>();
    U = Ud.template cast<T>();
    Sigma = sd.template cast<T>();
}

}