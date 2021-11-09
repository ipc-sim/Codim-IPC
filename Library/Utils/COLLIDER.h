#pragma once
//#####################################################################
// Class ABSTRACT_SHAPE, PLANE, BOX, SPHERE, TORUS
// Class COLLIDER
//#####################################################################
#include <pybind11/pybind11.h>
#include <functional>

namespace JGSL {

/** derivation:

  x = \phi(X,t) = R(t)s(t)X+b(t)
  X = \phi^{-1}(x,t) = (1/s) R^{-1} (x-b)
  V(X,t) = \frac{\partial \phi}{\partial t}
         = R'sX + Rs'X + RsX' + b'
  v(x,t) = V(\phi^{-1}(x,t),t)
         = R'R^{-1}(x-b) + (s'/s)(x-b) + RsX' + b'
         = omega \cross (x-b) + (s'/s)(x-b) +b'
*/
enum COLLISION_OBJECT_TYPE {
    STICKY,
    SLIP,
    SEPARATE
};

//#####################################################################
// Class ABSTRACT_SHAPE
//#####################################################################
template <class T, int dim>
class ABSTRACT_SHAPE {
    T s;
    T dsdt;
    VECTOR<T, 4> R;
    VECTOR<T, 3> omega;
    VECTOR<T, dim> b;
    VECTOR<T, dim> dbdt;
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
};

//#####################################################################
// Class PLANE
//#####################################################################
template <class T, int dim>
class PLANE : ABSTRACT_SHAPE<T, dim> {
    VECTOR<T, dim> origin;
    VECTOR<T, dim> normal;
public:
    PLANE(COLLISION_OBJECT_TYPE type, const VECTOR<T, dim>& origin, const VECTOR<T, dim>& normal)
        : ABSTRACT_SHAPE<T, dim>(type), origin(origin), normal(normal) {}
    T Get_Signed_Distance(const VECTOR<T, dim>& X) override { return normal.dot(X - origin);}
    VECTOR<T, dim> Get_Normal(const VECTOR<T, dim>& X) override { return normal; }
};

//#####################################################################
// Class BOX
//#####################################################################
template <class T, int dim>
class BOX : ABSTRACT_SHAPE<T, dim> {
    VECTOR<T, dim> bottom_left;
    VECTOR<T, dim> upper_right;
public:
    BOX(COLLISION_OBJECT_TYPE type, const VECTOR<T, dim>& bottom_left, const VECTOR<T, dim>& upper_right)
        : ABSTRACT_SHAPE<T, dim>(type), bottom_left(bottom_left), upper_right(upper_right) {}
    T Get_Signed_Distance(const VECTOR<T, dim>& X) override {
        VECTOR<T, dim> center = (upper_right + bottom_left) / 2;
        VECTOR<T, dim> point = (X - center).abs() - (upper_right - bottom_left) / 2;
        T max = point.max();
        for (int i = 0; i < dim; i++) {
            if (point(i) < 0) point(i) = 0;
        }
        return std::min((T)0, max) + point.length();
    }
    VECTOR<T, dim> Get_Normal(const VECTOR<T, dim>& X) override {
        VECTOR<T, dim> diff;
        VECTOR<T, dim> v1;
        VECTOR<T, dim> v2;
        T esp = (T)std::pow(10, -6);
        for (int i = 0; i < dim; i++) {
            v1 = X;
            v2 = X;
            v1(i) = X(i) + esp;
            v2(i) = X(i) - esp;
            diff(i) = (Get_Signed_Distance(v1) - Get_Signed_Distance(v2)) / (2 * esp);
        }
        return diff.Normalized();
    }
};

//#####################################################################
// Class SPHERE
//#####################################################################
template <class T, int dim>
class SPHERE : ABSTRACT_SHAPE<T, dim> {
    VECTOR<T,dim> center;
    T radius;
public:
    SPHERE(COLLISION_OBJECT_TYPE type, const VECTOR<T, dim>& center, const T& radius)
            : ABSTRACT_SHAPE<T, dim>(type), center(center), radius(radius) {}
    T Get_Signed_Distance(const VECTOR<T, dim>& X) override { return (X-center).length()-radius;}
    VECTOR<T, dim> Get_Normal(const VECTOR<T, dim>& X) override { return (X-center).Normalized();}
};

//#####################################################################
// Class TORUS
//#####################################################################
template <class T, int dim>
class TORUS : ABSTRACT_SHAPE<T, dim> {
    T minor_radius;
    T major_radius;
public:
    TORUS(COLLISION_OBJECT_TYPE type, const T& minor_radius, const T& major_radius)
            : ABSTRACT_SHAPE<T, dim>(type), minor_radius(minor_radius), major_radius(major_radius) {}
    T Get_Signed_Distance(const VECTOR<T, dim>& X) override {
        VECTOR<T, dim> tmp(X.x, 0, X.z);
        return (X - tmp.Normalized() * major_radius).length() - minor_radius;
    }
    VECTOR<T, dim> Get_Normal(const VECTOR<T, dim>& X) override {
        VECTOR<T, dim> tmp(X.x, 0, X.z);
        return (X - tmp.Normalized() * major_radius).Normalized();
    }
};

//#####################################################################
// Class COLLIDER
//#####################################################################
template <class T, int dim>
class COLLIDER {
    std::vector<std::shared_ptr<ABSTRACT_SHAPE<T, dim>>> shapes;
public:
    void Add(std::shared_ptr<ABSTRACT_SHAPE<T, dim>> shape) {
        shapes.push_back(shape);
    }
    void Clear() {
        shapes.clear();
    }
    bool Resolve(const VECTOR<T, dim>& x, VECTOR<T, dim>& v, T erosion = 0) const {
        bool inside = false;
        for (auto shape : shapes)
            if (shape->Get_Signed_Distance(x) < -erosion) {
                inside = true;
                VECTOR<T, dim> normal = shape->Get_Normal(x);
                if (shape->type == STICKY) v = VECTOR<T, dim>(0);
                if (shape->type == SLIP) v -= normal.dot(v) * normal;
                if (shape->type == SEPARATE && normal.dot(v) < 0) v -= normal.dot(v) * normal;
            }
        return inside;
    }
    bool IsCollided(const VECTOR<T, dim>& x) const {
        for (auto shape : shapes)
            if (shape->Get_Signed_Distance(x) < 0)
                return true;
        return false;
    }
    bool IsSticky(const VECTOR<T, dim>& x) const {
        for (auto shape : shapes)
            if (shape->Get_Signed_Distance(x) < 0){
                if (shape->type == STICKY) return true;
            }
        return false;
    }
    VECTOR<T, dim> Get_Closest_Point(const VECTOR<T, dim>& x, T erosion = 0) const {
        VECTOR<T, dim> v;
        for (auto shape : shapes) {
            T dist = shape->Get_Signed_Distance(x);
            if (dist < -erosion) {
                VECTOR<T, dim> normal = shape->Get_Normal(x);
                v -= (dist + erosion) * normal;
            }
        }
        return v;
    }
};

//#####################################################################
template <class T, int dim>
void Export_Collider_Impl(py::module& m) {
    auto suffix = std::string("_") + (dim == 2 ? "2" : "3") + (std::is_same<T, float>::value ? "F" : "D");
    py::class_<ABSTRACT_SHAPE<T, dim>, std::shared_ptr<ABSTRACT_SHAPE<T, dim>>> abstract_shape(m, ("ABSTRACT_SHAPE" + suffix).c_str());
    py::class_<PLANE<T, dim>, std::shared_ptr<PLANE<T, dim>>>(m, ("PLANE" + suffix).c_str(), abstract_shape).def(py::init<COLLISION_OBJECT_TYPE, const VECTOR<T, dim>&, const VECTOR<T, dim>&>());
    py::class_<BOX<T, dim>, std::shared_ptr<BOX<T, dim>>>(m, ("BOX" + suffix).c_str(), abstract_shape).def(py::init<COLLISION_OBJECT_TYPE, const VECTOR<T, dim>&, const VECTOR<T, dim>&>());
    py::class_<SPHERE<T, dim>, std::shared_ptr<SPHERE<T, dim>>>(m, ("SPHERE" + suffix).c_str(), abstract_shape).def(py::init<COLLISION_OBJECT_TYPE, const VECTOR<T, dim>&, const T&>());
    py::class_<COLLIDER<T, dim>>(m, ("COLLIDER" + suffix).c_str())
        .def(py::init<>())
        .def("Add", &COLLIDER<T, dim>::Add)
        .def("Clear", &COLLIDER<T, dim>::Clear);
}

void Export_Collider(py::module& m) {
    Export_Collider_Impl<double, 2>(m);
    Export_Collider_Impl<double, 3>(m);
    py::enum_<COLLISION_OBJECT_TYPE>(m, "COLLISION_OBJECT_TYPE")
            .value("STICKY", STICKY)
            .value("SLIP", SLIP)
            .value("SEPARATE", SEPARATE)
            .export_values();
}

}