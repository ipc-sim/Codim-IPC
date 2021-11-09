#pragma once
#include <Math/VECTOR.h>
#include <Grid/SPARSE_GRID.h>
#include <Eigen/Eigen>
#include <Math/LINEAR.h>

namespace JGSL {

template <class T>
T Fraction_Inside(T phi_left, T phi_right)
{
    if (phi_left < 0 && phi_right < 0) {
        return (T)1;
    }
    if (phi_left < 0 && phi_right >= 0) {
        return (T) phi_left / (phi_left - phi_right);
    }
    if (phi_left >= 0 && phi_right < 0) {
        return (T) phi_right / (phi_right - phi_left);
    }
    return (T)0;
}

template <class T>
void cycle_array(T *arr, int size)
{
    T t = arr[0];
    for (int i = 0; i < size-1; ++i) {
        arr[i] = arr[i + 1];
    }
    arr[size - 1] = t;
}

template <class T>
T Fraction_Inside(T phi_bl, T phi_br, T phi_tl, T phi_tr)
{
    int inside_count = (phi_bl<0?1:0) + (phi_tl<0?1:0) + (phi_br<0?1:0) + (phi_tr<0?1:0);
    T list[] = {phi_bl, phi_br, phi_tr, phi_tl};
    if (inside_count == 4) {
        return (T)1;
    }
    else if (inside_count == 3) {
        while(list[0] < 0) {
            cycle_array(list, 4);
        }
        T side0 = (T)1 - Fraction_Inside(list[0], list[3]);
        T side1 = (T)1 - Fraction_Inside(list[0], list[1]);
        return (T)(1 - 0.5 * side0 * side1);
    }
    else if (inside_count == 2) {
        while (list[0] >= 0 || !(list[1] < 0 || list[2] < 0)) {
            cycle_array(list, 4);
        }
        if (list[1] < 0) {
            T side_left = Fraction_Inside(list[0], list[3]);
            T side_right = Fraction_Inside(list[1], list[2]);
            return (T)(0.5 * (side_left + side_right));
        } else {
            T middle_point = (T)0.25 * (list[0] + list[1] + list[2] + list[3]);
            if (middle_point < 0) {
                T area = 0;
                T side1 = (T) 1 - Fraction_Inside(list[0], list[3]);
                T side3 = (T) 1 - Fraction_Inside(list[2], list[3]);
                area += (T)(0.5 * side1 * side3);

                T side0 = (T) 1 - Fraction_Inside(list[2], list[1]);
                T side2 = (T) 1 - Fraction_Inside(list[0], list[1]);
                area += (T)(0.5 * side0 * side2);

                return (T)1 - area;
            } else {
                T area = 0;
                T side0 = (T) Fraction_Inside(list[0], list[1]);
                T side1 = (T) Fraction_Inside(list[0], list[3]);
                area += (T)(0.5 * side0 * side1);

                T side2 = (T) Fraction_Inside(list[2], list[1]);
                T side3 = (T) Fraction_Inside(list[2], list[3]);
                area += (T)(0.5 * side2 * side3);

                return area;
            }
        }
    } else if (inside_count == 1) {
        while (list[0] >= 0) {
            cycle_array(list, 4);
        }
        T side0 = Fraction_Inside(list[0], list[3]);
        T side1 = Fraction_Inside(list[0], list[1]);
        return (T)(0.5 * side0 * side1);
    } else
        return (T)0;
}

}