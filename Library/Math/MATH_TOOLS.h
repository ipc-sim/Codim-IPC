#pragma once
//#####################################################################
// Function Int_Floor
// Function Deviatoric
// Function Get_Max_Coeff
// Function Get_Cross_Product
// Function Is_Much_Smaller_Than
// Function Get_Unit_Orthogonal
//#####################################################################
#include <Math/VECTOR.h>

namespace JGSL {
namespace MATH_TOOLS {

//#####################################################################
// Function Int_Floor
//#####################################################################
template <class T>
inline static int Int_Floor(T x)
{
    int i = (int)x; /* truncate */
    return i - (i > x); /* convert trunc to floor */
}

//#####################################################################
// Function Deviatoric
//#####################################################################
template <class T, int dim>
inline static VECTOR<T,dim> Deviatoric(const VECTOR<T,dim>& input)
{
    return input-input.Average();
}

//#####################################################################
// Function Get_Max_Coeff
//#####################################################################
template<class T, int dim>
inline static T Get_Max_Coeff(VECTOR<T, dim>& V, int& index) {
    T max = V(0);
    index = 0;
    for (int i = 1; i < dim; i++) {
        if (max < V(i)) {
            max = V(i);
            index = i;
        }
    }
    return max;
}

//#####################################################################
// Function Get_Cross_Product
// a cross b = a2b3 - a3b2, a3b1-a1b3, a1b2-a2b1
//#####################################################################
template<class T>
inline static VECTOR<T, 3> Get_Cross_Product(VECTOR<T, 3>& a, VECTOR<T, 3>& b) {
    VECTOR<T, 3> result(0);
    result(0) = a(1)*b(2) - a(2)*b(1);
    result(1) = a(2)*b(0) - a(0)*b(2);
    result(2) = a(0)*b(1) - a(1)*b(0);
    return result;
}

//#####################################################################
// Function Is_Much_Smaller_Than
//#####################################################################
template<class T>
inline static bool Is_Much_Smaller_Than(T& a, T& b) {
    if (std::abs(a) < std::abs(b) * 1e-12) {
        return true;
    } else {
        return false;
    }
}

//#####################################################################
// Function Get_Unit_Orthogonal
//#####################################################################
template<class T>
inline static VECTOR<T, 3> Get_Unit_Orthogonal(VECTOR<T, 3>& V) {
    if ((!Is_Much_Smaller_Than(V(0), V(2))) || (!Is_Much_Smaller_Than(V(1), V(2)))) {
        T invnm = (T) 1.0 / std::sqrt(V(0) * V(0) + V(1) * V(1));
        return VECTOR<T, 3>(-V(1) * invnm, V(0) * invnm, 0.0);
    } else {
        T invnm = (T) 1.0 / std::sqrt(V(1) * V(1) + V(2) * V(2));
        return VECTOR<T, 3>(0.0, -V(2) * invnm, V(1) * invnm);
    }
}

//#####################################################################
// Function Clamp
//#####################################################################
template<class T>
inline static void clamp(T& x, const T min, const T max) {
    if (x < min)
        x = min;
    else if (x > max)
        x = max;
}

//#####################################################################
// Function log_1px_over_x
// Robustly computing log(x+1)/x
//#####################################################################
template<class T>
inline static T log_1px_over_x(const T x, const T eps)
{
    assert(eps > 0);
    if (std::fabs(x) < eps)
        return (T)1;
    else
        return std::log1p(x) / x;
}

//#####################################################################
// Function diff_log_over_diff
// Robustly computing (logx-logy)/(x-y)
//#####################################################################
template<class T>
inline static T diff_log_over_diff(const T x, const T y, const T eps)
{
    assert(eps > 0);
    T p = x / y - 1;
    return log_1px_over_x(p, eps) / y;
}

//#####################################################################
// Function diff_interlock_log_over_diff
// Robustly computing (x logy- y logx)/(x-y)
//#####################################################################
template<class T>
inline static T diff_interlock_log_over_diff(const T x, const T y, const T logy, const T eps)
{
    assert(eps > 0);
    return logy - y * diff_log_over_diff(x, y, eps);
}
}
}