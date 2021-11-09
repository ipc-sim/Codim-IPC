#pragma once
#include <Math/VECTOR.h>
#include <Grid/SPARSE_GRID.h>
#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <Math/LINEAR.h>

namespace py = pybind11;
namespace JGSL {

/*//#####################################################################
// Class ABSTRACT_SHAPE
//#####################################################################
template <class T, int dim>
class ABSTRACT_SHAPE {
public:
    COLLISION_OBJECT_TYPE type;
    explicit ABSTRACT_SHAPE(COLLISION_OBJECT_TYPE type) : type(type), s(1), dsdt(0) {};
    virtual ~ABSTRACT_SHAPE() = default;
    virtual T Get_Signed_Distance(const VECTOR<T, dim>& X) = 0;
    virtual VECTOR<T, dim> Get_Normal(const VECTOR<T, dim>& X) = 0;
    void Set_Translation(VECTOR<T,dim> b_in, VECTOR<T,dim> dbdt_in){
        b = b_in;
        dbdt = dbdt_in;
    }
    void Set_Rotation(VECTOR<T,4> R_in, VECTOR<T,3> omega_in){
        R = R_in;
        omega = omega_in;
    }
    bool Is_Collide(const VECTOR<T, dim>& x) {
        VECTOR<T, dim> x_minus_b = x - b;
        T one_over_s = 1 / s;
        VECTOR<T, dim> X = R.rotation.transpose() * x_minus_b * one_over_s; // material space
        return Get_Signed_Distance(X) < 0;
    }
    VECTOR<T, dim> Get_Velocity(const VECTOR<T, dim>& x) {
        VECTOR<T, dim> x_minus_b = x - b;
        T one_over_s = 1 / s;
        VECTOR<T, dim> v_object = omega.cross(x_minus_b) + (dsdt * one_over_s) * x_minus_b + R.rotation * s + dbdt;
        return v_object;
    }
};*/

template <class T, int dim>
T Box_Phi(const VECTOR<T, dim> p,
          const VECTOR<T, dim> center,
          const VECTOR<T, dim> expand)
{
    VECTOR<T,dim> t = p - center;
    VECTOR<T,dim> d;
    for (int i = 0; i < dim; i++) {
        d(i) = t(i) > 0? t(i) : -t(i);
    }
    T d1 = 0.0, d2 = 0.0;
    if constexpr (dim == 2) {
        d = d - expand;
        d1 = std::min(std::max(d(0),d(1)), 0.0);
        VECTOR<T,dim> tmp(std::max(d(0),0.0),std::max(d(1),0.0));
        d2 = tmp.norm();
    } else if constexpr (dim == 3) {
        d = d - expand;
        d1 = std::min(std::max(d(0), std::max(d(1),d(2))), 0.0);
        VECTOR<T,dim> tmp(std::max(d(0),0.0),std::max(d(1),0.0),std::max(d(2),0.0));
        d2 = tmp.norm();
    }
    return d1 + d2;
}

template <class T, int dim>
T Sphere_Phi(const VECTOR<T, dim> p,
             const VECTOR<T, dim> center,
             T radius)
{
    VECTOR<T,dim> t = p - center;
    T len = t.norm();
    return len - radius;
}
//?
template <class T, int dim>
T Cylinder_Phi(VECTOR<T, dim> p,
          VECTOR<T, dim> center,
          VECTOR<T, dim> b)
{
    VECTOR<T,dim> t = p - center;
    VECTOR<T,dim> d;
    for (int i = 0; i < dim; i++) {
        d(i) = t(i) > 0? t(i) : -t(i);
    }
    d = d - b;
    T d1 = std::min(std::max(d(0), std::max(d(1),d(2))), 0.0);

    VECTOR<T,dim> temp(std::max(d(0),0.0),std::max(d(1),0.0),std::max(d(2),0.0));
    T d2 = temp.norm();

    return d1 + d2;
}
//?
template <class T, int dim>
T Torus_Phi(VECTOR<T, dim> p,
          VECTOR<T, 2> t)
{

    return 1;
}
//?
template <class T, int dim>
T Triangle_Phi(VECTOR<T, dim> p,
          VECTOR<T, dim> center,
          VECTOR<T, dim> b)
{
    VECTOR<T,dim> t = p - center;
    VECTOR<T,dim> d;
    for (int i = 0; i < dim; i++) {
        d(i) = t(i) > 0? t(i) : -t(i);
    }
    d = d - b;
    T d1 = std::min(std::max(d(0), std::max(d(1),d(2))), 0.0);

    VECTOR<T,dim> temp(std::max(d(0),0.0),std::max(d(1),0.0),std::max(d(2),0.0));
    T d2 = temp.norm();

    return d1 + d2;
}
}
