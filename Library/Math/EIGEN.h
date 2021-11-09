#pragma once

//#####################################################################
// Helper Function Eigen Value and Eigen Vector
//#####################################################################

#include <Math/VECTOR.h>
#include <Math/MATH_TOOLS.h>

namespace JGSL {

void Get_Eigen_Values(
    const MATRIX<double, 3>& A_Sym,
    VECTOR<double, 3>& lambda) {
    using T = double;
    using std::sqrt;
    using std::max;
    using std::swap;

    T m = ((T)1 / 3) * (A_Sym(0, 0) + A_Sym(1, 1) + A_Sym(2, 2));
    T a00 = A_Sym(0, 0) - m;
    T a11 = A_Sym(1, 1) - m;
    T a22 = A_Sym(2, 2) - m;
    T a12_sqr = A_Sym(0, 1) * A_Sym(0, 1);
    T a13_sqr = A_Sym(0, 2) * A_Sym(0, 2);
    T a23_sqr = A_Sym(1, 2) * A_Sym(1, 2);
    T p = ((T)1 / 6) * (a00 * a00 + a11 * a11 + a22 * a22 + 2 * (a12_sqr + a13_sqr + a23_sqr));
    T q = (T).5 * (a00 * (a11 * a22 - a23_sqr) - a11 * a13_sqr - a22 * a12_sqr) + A_Sym(0, 1) * A_Sym(0, 2) * A_Sym(1, 2);
    T sqrt_p = sqrt(p);
    T disc = p * p * p - q * q;
    T phi = ((T)1 / 3) * atan2(sqrt(max((T)0, disc)), q);
    T c = cos(phi), s = sin(phi);
    T sqrt_p_cos = sqrt_p * c;
    T root_three_sqrt_p_sin = sqrt((T)3) * sqrt_p * s;

    lambda(0) = m + 2 * sqrt_p_cos;
    lambda(1) = m - sqrt_p_cos - root_three_sqrt_p_sin;
    lambda(2) = m - sqrt_p_cos + root_three_sqrt_p_sin;

    if (lambda(0) < lambda(1))
        swap(lambda(0), lambda(1));
    if (lambda(1) < lambda(2))
        swap(lambda(1), lambda(2));
    if (lambda(0) < lambda(1))
        swap(lambda(0), lambda(1));
}


void Get_Eigen_Vectors(
    const MATRIX<double,3>& A_Sym,
    const VECTOR<double,3>& lambda,
    MATRIX<double,3>& V) {
    using T = double;
    using std::swap;
    using std::sqrt;

    bool flipped = false;
    VECTOR<T, 3> lambda_flip(lambda);
    if (lambda(0) - lambda(1) < lambda(1) - lambda(2)) {
        swap(lambda_flip(0), lambda_flip(2));
        flipped = true;
    }

    // get first eigenvector
    MATRIX<T, 3> C1 = (A_Sym - lambda_flip(0) * MATRIX<T, 3>(VECTOR<T, 3>(1))).cofactor();

    VECTOR<T ,3> SquaredNorm(0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
           SquaredNorm(i) += C1(i, j) * C1(i, j);
        }
    }

    int i = -1;
    T norm2 = MATH_TOOLS::Get_Max_Coeff<T>(SquaredNorm, i);

    VECTOR<double, 3> v1(0);
    if (norm2 != 0) {
        T one_over_sqrt = (T)1 / sqrt(norm2);
        v1 = C1(i) * one_over_sqrt;
    } else {
        v1 = VECTOR<T ,3>(1, 0, 0);
    }
    // form basis for orthogonal complement to v1, and reduce A to this space need this function.
    VECTOR<T, 3> v1_orthogonal = MATH_TOOLS::Get_Unit_Orthogonal<T>(v1);
    MATRIX<T, 3> other_v(0); // in fact is 3, 2
    other_v(0) = v1_orthogonal;
    other_v(1) = MATH_TOOLS::Get_Cross_Product<T>(v1, v1_orthogonal);
    MATRIX<T, 2> A_reduced = MATRIX<T, 2>(other_v.transpose() * A_Sym * other_v);
    MATRIX<T, 2> C3 = (A_reduced - lambda_flip(2) * MATRIX<T ,2>(VECTOR<T, 2>(1))).cofactor();
    norm2 = -1;
    VECTOR<T, 2> SquaredNorm2;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
           SquaredNorm2(i) += C3(i, j) * C3(i, j);
        }
    }
    int j = -1;
    norm2 = MATH_TOOLS::Get_Max_Coeff<T>(SquaredNorm2, j);

    VECTOR<T, 3> v3;
    if (norm2 != 0) {
        T one_over_sqrt = (T)1 / sqrt(norm2);
        VECTOR<T, 3> tmp;
        tmp(0) = other_v(0, 0) * C3(j, 0) + other_v(0, 1) * C3(j, 1);
        tmp(1) = other_v(1, 0) * C3(j, 0) + other_v(1, 1) * C3(j, 1);
        tmp(2) = other_v(2, 0) * C3(j, 0) + other_v(2, 1) * C3(j, 1);
        v3 = tmp * one_over_sqrt;
    } else {
        v3 = other_v(0);
    }

    VECTOR<T, 3> v2 = MATH_TOOLS::Get_Cross_Product<T>(v3, v1);

    if (flipped) {
        V(0) = v3;
        V(1) = v2;
        V(2) = (T)-1 * v1;
    } else {
        V(0) = v1;
        V(1) = v2;
        V(2) = v3;
    }
}

}