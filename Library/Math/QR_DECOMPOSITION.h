#pragma once

//#####################################################################
// Helper QR Decomposition 2d and 3d
//#####################################################################

#include <Math/VECTOR.h>
#include <Math/MATH_TOOLS.h>

namespace JGSL {
/*
template <class T, int dim>
inline void simultaneousGivensQR(MATRIX<T, dim>& A, MATRIX<T, dim>& M)
{
    for (int j = 0; j < A.cols(); j++) {
        for (int i = A.rows() - 1; i > j; i--) {
            GivensRotation<T> r(A(i - 1, j), A(i, j), i - 1, i);
            r.rowRotation(A);
            r.rowRotation(M);
        }
    }
    // A <- Q.transpose() * A = R
    // M <- Q.transpose() * M
    // thin or thick A, M not yet tested
}*/


/*
   QR
*/
/*template <class T, int dim>
void GivensQR(const MATRIX<T, dim>& A, MATRIX<T, dim>& Q, MATRIX<T, dim>& R) {
    R = A;
    //inplaceGivensQR(R, Q);
    Q.setIdentity();
    simultaneousGivensQR(A, Q);
    Q.transposeInPlace();
}*/

/*
   Thin QR
*/
/*template <class T, int dim>
void thinGivensQR(const MATRIX<T, m, n, 0, m, n>& A, MATRIX<T, m, n, 0, m, n>& Q0, MATRIX<T, n, n, 0, n, n>& R0)
{
    MATRIX<T, m, m> Q(A.rows(), A.rows());
    MATRIX<T, m, n> R = A;
    inplaceGivensQR(R, Q);
    Q0 = Q.topLeftCorner(A.rows(), A.cols());
    R0 = R.topLeftCorner(A.cols(), A.cols());
}*/
}