#pragma once

#include <Math/Distance/DISTANCE_TYPE.h>
#include <Math/Distance/DISTANCE_UNCLASSIFIED.h>

#include <Math/Distance/EVCTCD/CTCD.h>

#include <bsc_tightbound.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim>
bool Point_Edge_CD_Broadphase(
    const Eigen::Matrix<T, dim, 1>& x0, 
    const Eigen::Matrix<T, dim, 1>& x1, 
    const Eigen::Matrix<T, dim, 1>& x2,
    T dist)
{
    const Eigen::Array<T, dim, 1> max_e = x1.array().max(x2.array());
    const Eigen::Array<T, dim, 1> min_e = x1.array().min(x2.array());
    if ((x0.array() - max_e > dist).any() || (min_e - x0.array() > dist).any()) {
        return false;
    }
    else {
        return true;
    }
}

template <class T>
bool Point_Edge_CCD_Broadphase(
    const Eigen::Matrix<T, 2, 1>& p, 
    const Eigen::Matrix<T, 2, 1>& e0, 
    const Eigen::Matrix<T, 2, 1>& e1,
    const Eigen::Matrix<T, 2, 1>& dp, 
    const Eigen::Matrix<T, 2, 1>& de0, 
    const Eigen::Matrix<T, 2, 1>& de1,
    T dist)
{
    const Eigen::Array<T, 2, 1> max_p = p.array().max((p + dp).array());
    const Eigen::Array<T, 2, 1> min_p = p.array().min((p + dp).array());
    const Eigen::Array<T, 2, 1> max_e = e0.array().max(e1.array()).
        max((e0 + de0).array()).max((e1 + de1).array());
    const Eigen::Array<T, 2, 1> min_e = e0.array().min(e1.array()).
        min((e0 + de0).array()).min((e1 + de1).array());
    if ((min_p - max_e > dist).any() || (min_e - max_p > dist).any()) {
        return false;
    }
    else {
        return true;
    }
}

template <class T>
bool Point_Edge_CCD(const Eigen::Matrix<T, 2, 1>& x0, 
    const Eigen::Matrix<T, 2, 1>& x1, 
    const Eigen::Matrix<T, 2, 1>& x2,
    const Eigen::Matrix<T, 2, 1>& d0, 
    const Eigen::Matrix<T, 2, 1>& d1, 
    const Eigen::Matrix<T, 2, 1>& d2,
    T eta, T& toc)
{
    T a = d0[0] * (d2[1] - d1[1]) + d0[1] * (d1[0] - d2[0]) + d2[0] * d1[1] - d2[1] * d1[0];
    T b = x0[0] * (d2[1] - d1[1]) + d0[0] * (x2[1] - x1[1]) + 
        d0[1] * (x1[0] - x2[0]) + x0[1] * (d1[0] - d2[0]) +
        d1[1] * x2[0] + d2[0] * x1[1] - d1[0] * x2[1] - d2[1] * x1[0];
    T c = x0[0] * (x2[1] - x1[1]) + x0[1] * (x1[0] - x2[0]) + x2[0] * x1[1] - x2[1] * x1[0];

    T roots[2];
    int rootAmt = 0;
    if (a == 0) {
        if (b == 0) {
            // parallel motion, only need to handle colinear case
            if (c == 0) {
                // colinear PP CCD
                if ((x0 - x1).dot(d0 - d1) < 0) {
                    roots[rootAmt] = std::sqrt((x0 - x1).squaredNorm() / (d0 - d1).squaredNorm());
                    if (roots[rootAmt] > 0 && roots[rootAmt] <= 1) {
                        ++rootAmt;
                    }
                }
                if ((x0 - x2).dot(d0 - d2) < 0) {
                    roots[rootAmt] = std::sqrt((x0 - x2).squaredNorm() / (d0 - d2).squaredNorm());
                    if (roots[rootAmt] > 0 && roots[rootAmt] <= 1) {
                        ++rootAmt;
                    }
                }

                if (rootAmt == 2) {
                    toc = std::min(roots[0], roots[1]) * (1 - eta);
                    return true;
                }
                else if (rootAmt == 1) {
                    toc = roots[0] * (1 - eta);
                    return true;
                }
                else {
                    return false;
                }
            }
        }
        else {
            rootAmt = 1;
            roots[0] = -c / b;
        }
    }
    else {
        T delta = b * b - 4 * a * c;
        if (delta == 0) {
            rootAmt = 1;
            roots[0] = -b / (2 * a);
        }
        else if (delta > 0) {
            rootAmt = 2;
            // accurate expression differs in b's sign
            if (b > 0) {
                roots[0] = (-b - std::sqrt(delta)) / (2 * a);
                roots[1] = 2 * c / (-b - std::sqrt(delta));
            }
            else {
                roots[0] = 2 * c / (-b + std::sqrt(delta));
                roots[1] = (-b + std::sqrt(delta)) / (2 * a);
            }

            if (roots[0] > roots[1]) {
                std::swap(roots[0], roots[1]);
            }
        }
    }

    for (int i = 0; i < rootAmt; ++i) {
        if (roots[i] > 0 && roots[i] <= 1) {
            // check overlap
            T ratio;
            if(Point_Edge_Distance_Type(Eigen::Matrix<T, 2, 1>(x0 + roots[i] * d0), 
                Eigen::Matrix<T, 2, 1>(x1 + roots[i] * d1), 
                Eigen::Matrix<T, 2, 1>(x2 + roots[i] * d2), ratio) == 2) {
                toc = roots[i] * (1 - eta); //TODO: distance eta
                return true;
            }
        }
    }

    return false;
}

template <class T>
bool Point_Triangle_CD_Broadphase(
    const Eigen::Matrix<T, 3, 1>& p, 
    const Eigen::Matrix<T, 3, 1>& t0, 
    const Eigen::Matrix<T, 3, 1>& t1,
    const Eigen::Matrix<T, 3, 1>& t2,
    T dist)
{
    const Eigen::Array<T, 3, 1> max_tri = t0.array().max(t1.array()).max(t2.array());
    const Eigen::Array<T, 3, 1> min_tri = t0.array().min(t1.array()).min(t2.array());
    if ((p.array() - max_tri > dist).any() || (min_tri - p.array() > dist).any()) {
        return false;
    }
    else {
        return true;
    }
}

template <class T>
bool Edge_Edge_CD_Broadphase(
    const Eigen::Matrix<T, 3, 1>& ea0, 
    const Eigen::Matrix<T, 3, 1>& ea1, 
    const Eigen::Matrix<T, 3, 1>& eb0,
    const Eigen::Matrix<T, 3, 1>& eb1,
    T dist)
{
    const Eigen::Array<T, 3, 1> max_a = ea0.array().max(ea1.array());
    const Eigen::Array<T, 3, 1> min_a = ea0.array().min(ea1.array());
    const Eigen::Array<T, 3, 1> max_b = eb0.array().max(eb1.array());
    const Eigen::Array<T, 3, 1> min_b = eb0.array().min(eb1.array());
    if ((min_a - max_b > dist).any() || (min_b - max_a > dist).any()) {
        return false;
    }
    else {
        return true;
    }
}

template <class T>
bool Point_Triangle_CCD_Broadphase(
    const Eigen::Matrix<T, 3, 1>& p, 
    const Eigen::Matrix<T, 3, 1>& t0, 
    const Eigen::Matrix<T, 3, 1>& t1,
    const Eigen::Matrix<T, 3, 1>& t2,
    const Eigen::Matrix<T, 3, 1>& dp, 
    const Eigen::Matrix<T, 3, 1>& dt0, 
    const Eigen::Matrix<T, 3, 1>& dt1,
    const Eigen::Matrix<T, 3, 1>& dt2,
    T dist)
{
    const Eigen::Array<T, 3, 1> max_p = p.array().max((p + dp).array());
    const Eigen::Array<T, 3, 1> min_p = p.array().min((p + dp).array());
    const Eigen::Array<T, 3, 1> max_tri = t0.array().max(t1.array()).max(t2.array()).
        max((t0 + dt0).array()).max((t1 + dt1).array()).max((t2 + dt2).array());
    const Eigen::Array<T, 3, 1> min_tri = t0.array().min(t1.array()).min(t2.array()).
        min((t0 + dt0).array()).min((t1 + dt1).array()).min((t2 + dt2).array());
    if ((min_p - max_tri > dist).any() || (min_tri - max_p > dist).any()) {
        return false;
    }
    else {
        return true;
    }
}

template <class T>
bool Edge_Edge_CCD_Broadphase(
    const Eigen::Matrix<T, 3, 1>& ea0, 
    const Eigen::Matrix<T, 3, 1>& ea1, 
    const Eigen::Matrix<T, 3, 1>& eb0,
    const Eigen::Matrix<T, 3, 1>& eb1,
    const Eigen::Matrix<T, 3, 1>& dea0, 
    const Eigen::Matrix<T, 3, 1>& dea1, 
    const Eigen::Matrix<T, 3, 1>& deb0,
    const Eigen::Matrix<T, 3, 1>& deb1,
    T dist)
{
    const Eigen::Array<T, 3, 1> max_a = ea0.array().max(ea1.array()).max((ea0 + dea0).array()).max((ea1 + dea1).array());
    const Eigen::Array<T, 3, 1> min_a = ea0.array().min(ea1.array()).min((ea0 + dea0).array()).min((ea1 + dea1).array());
    const Eigen::Array<T, 3, 1> max_b = eb0.array().max(eb1.array()).max((eb0 + deb0).array()).max((eb1 + deb1).array());
    const Eigen::Array<T, 3, 1> min_b = eb0.array().min(eb1.array()).min((eb0 + deb0).array()).min((eb1 + deb1).array());
    if ((min_a - max_b > dist).any() || (min_b - max_a > dist).any()) {
        return false;
    }
    else {
        return true;
    }
}

template <class T>
bool Point_Edge_CCD_Broadphase(
    const Eigen::Matrix<T, 3, 1>& p, 
    const Eigen::Matrix<T, 3, 1>& e0, 
    const Eigen::Matrix<T, 3, 1>& e1,
    const Eigen::Matrix<T, 3, 1>& dp, 
    const Eigen::Matrix<T, 3, 1>& de0, 
    const Eigen::Matrix<T, 3, 1>& de1,
    T dist)
{
    const Eigen::Array<T, 3, 1> max_p = p.array().max((p + dp).array());
    const Eigen::Array<T, 3, 1> min_p = p.array().min((p + dp).array());
    const Eigen::Array<T, 3, 1> max_e = e0.array().max(e1.array()).max((e0 + de0).array()).max((e1 + de1).array());
    const Eigen::Array<T, 3, 1> min_e = e0.array().min(e1.array()).min((e0 + de0).array()).min((e1 + de1).array());
    if ((min_p - max_e > dist).any() || (min_e - max_p > dist).any()) {
        return false;
    }
    else {
        return true;
    }
}

template <class T>
bool Point_Point_CCD_Broadphase(
    const Eigen::Matrix<T, 3, 1>& p0, 
    const Eigen::Matrix<T, 3, 1>& p1,
    const Eigen::Matrix<T, 3, 1>& dp0, 
    const Eigen::Matrix<T, 3, 1>& dp1,
    T dist)
{
    const Eigen::Array<T, 3, 1> max_p0 = p0.array().max((p0 + dp0).array());
    const Eigen::Array<T, 3, 1> min_p0 = p0.array().min((p0 + dp0).array());
    const Eigen::Array<T, 3, 1> max_p1 = p1.array().max((p1 + dp1).array());
    const Eigen::Array<T, 3, 1> min_p1 = p1.array().min((p1 + dp1).array());
    if ((min_p0 - max_p1 > dist).any() || (min_p1 - max_p0 > dist).any()) {
        return false;
    }
    else {
        return true;
    }
}

template <class T>
bool Point_Triangle_CCD(
    Eigen::Matrix<T, 3, 1> p, 
    Eigen::Matrix<T, 3, 1> t0, 
    Eigen::Matrix<T, 3, 1> t1,
    Eigen::Matrix<T, 3, 1> t2,
    Eigen::Matrix<T, 3, 1> dp, 
    Eigen::Matrix<T, 3, 1> dt0, 
    Eigen::Matrix<T, 3, 1> dt1,
    Eigen::Matrix<T, 3, 1> dt2,
    T eta, T thickness, T& toc)
{
    T t = toc;
    Eigen::Matrix<T, 3, 1> pend = p + t * dp;
    Eigen::Matrix<T, 3, 1> t0end = t0 + t * dt0;
    Eigen::Matrix<T, 3, 1> t1end = t1 + t * dt1;
    Eigen::Matrix<T, 3, 1> t2end = t2 + t * dt2;
    while(bsc_tightbound::Intersect_VF_robust(
        // Triangle at t = 0
        Vec3d(t0.data()),
        Vec3d(t1.data()),
        Vec3d(t2.data()),
        // Point at t=0
        Vec3d(p.data()),
        // Triangle at t = 1
        Vec3d(t0end.data()),
        Vec3d(t1end.data()),
        Vec3d(t2end.data()),
        // Point at t=1
        Vec3d(pend.data())))
    {
        t /= 2;
        pend = p + t * dp;
        t0end = t0 + t * dt0;
        t1end = t1 + t * dt1;
        t2end = t2 + t * dt2;

        if (t < 1e-8) {
            printf("%.10le %.10le %.10le\n", p[0], p[1], p[2]);
            printf("%.10le %.10le %.10le\n", t0[0], t0[1], t0[2]);
            printf("%.10le %.10le %.10le\n", t1[0], t1[1], t1[2]);
            printf("%.10le %.10le %.10le\n", t2[0], t2[1], t2[2]);
            printf("%.10le %.10le %.10le\n", dp[0], dp[1], dp[2]);
            printf("%.10le %.10le %.10le\n", dt0[0], dt0[1], dt0[2]);
            printf("%.10le %.10le %.10le\n", dt1[0], dt1[1], dt1[2]);
            printf("%.10le %.10le %.10le\n", dt2[0], dt2[1], dt2[2]);
            printf("%ld %ld %ld\n", *(long*)&p[0], *(long*)&p[1], *(long*)&p[2]);
            printf("%ld %ld %ld\n", *(long*)&t0[0], *(long*)&t0[1], *(long*)&t0[2]);
            printf("%ld %ld %ld\n", *(long*)&t1[0], *(long*)&t1[1], *(long*)&t1[2]);
            printf("%ld %ld %ld\n", *(long*)&t2[0], *(long*)&t2[1], *(long*)&t2[2]);
            printf("%ld %ld %ld\n", *(long*)&dp[0], *(long*)&dp[1], *(long*)&dp[2]);
            printf("%ld %ld %ld\n", *(long*)&dt0[0], *(long*)&dt0[1], *(long*)&dt0[2]);
            printf("%ld %ld %ld\n", *(long*)&dt1[0], *(long*)&dt1[1], *(long*)&dt1[2]);
            printf("%ld %ld %ld\n", *(long*)&dt2[0], *(long*)&dt2[1], *(long*)&dt2[2]);
            T dist2_cur;
            Point_Triangle_Distance_Unclassified(p, t0, t1, t2, dist2_cur);
            printf("PT%.10le\n", dist2_cur);
            exit(-1);
        }
    }

    if (t == toc) {
        return false;
    }
    else {
        toc = t * (1 - eta);
        return true;
    }
}

template <class T>
bool Edge_Edge_CCD(
    Eigen::Matrix<T, 3, 1> ea0, 
    Eigen::Matrix<T, 3, 1> ea1, 
    Eigen::Matrix<T, 3, 1> eb0,
    Eigen::Matrix<T, 3, 1> eb1,
    Eigen::Matrix<T, 3, 1> dea0, 
    Eigen::Matrix<T, 3, 1> dea1, 
    Eigen::Matrix<T, 3, 1> deb0,
    Eigen::Matrix<T, 3, 1> deb1,
    T eta, T thickness, T& toc)
{
    T t = toc;
    Eigen::Matrix<T, 3, 1> ea0end = ea0 + t * dea0;
    Eigen::Matrix<T, 3, 1> ea1end = ea1 + t * dea1;
    Eigen::Matrix<T, 3, 1> eb0end = eb0 + t * deb0;
    Eigen::Matrix<T, 3, 1> eb1end = eb1 + t * deb1;
    while(bsc_tightbound::Intersect_EE_robust(
        // Point at t=0
        Vec3d(ea0.data()),
        // Triangle at t = 0
        Vec3d(ea1.data()),
        Vec3d(eb0.data()),
        Vec3d(eb1.data()),
        // Point at t=1
        Vec3d(ea0end.data()),
        // Triangle at t = 1
        Vec3d(ea1end.data()),
        Vec3d(eb0end.data()),
        Vec3d(eb1end.data())))
    {
        t /= 2;
        ea0end = ea0 + t * dea0;
        ea1end = ea1 + t * dea1;
        eb0end = eb0 + t * deb0;
        eb1end = eb1 + t * deb1;

        if (t < 1e-8) {
            printf("%.10le %.10le %.10le\n", ea0[0], ea0[1], ea0[2]);
            printf("%.10le %.10le %.10le\n", ea1[0], ea1[1], ea1[2]);
            printf("%.10le %.10le %.10le\n", eb0[0], eb0[1], eb0[2]);
            printf("%.10le %.10le %.10le\n", eb1[0], eb1[1], eb1[2]);
            printf("%.10le %.10le %.10le\n", dea0[0], dea0[1], dea0[2]);
            printf("%.10le %.10le %.10le\n", dea1[0], dea1[1], dea1[2]);
            printf("%.10le %.10le %.10le\n", deb0[0], deb0[1], deb0[2]);
            printf("%.10le %.10le %.10le\n", deb1[0], deb1[1], deb1[2]);
            printf("%ld %ld %ld\n", *(long*)&ea0[0], *(long*)&ea0[1], *(long*)&ea0[2]);
            printf("%ld %ld %ld\n", *(long*)&ea1[0], *(long*)&ea1[1], *(long*)&ea1[2]);
            printf("%ld %ld %ld\n", *(long*)&eb0[0], *(long*)&eb0[1], *(long*)&eb0[2]);
            printf("%ld %ld %ld\n", *(long*)&eb1[0], *(long*)&eb1[1], *(long*)&eb1[2]);
            printf("%ld %ld %ld\n", *(long*)&dea0[0], *(long*)&dea0[1], *(long*)&dea0[2]);
            printf("%ld %ld %ld\n", *(long*)&dea1[0], *(long*)&dea1[1], *(long*)&dea1[2]);
            printf("%ld %ld %ld\n", *(long*)&deb0[0], *(long*)&deb0[1], *(long*)&deb0[2]);
            printf("%ld %ld %ld\n", *(long*)&deb1[0], *(long*)&deb1[1], *(long*)&deb1[2]);
            T dist2_cur;
            Edge_Edge_Distance_Unclassified(ea0, ea1, eb0, eb1, dist2_cur);
            printf("EE%.10le\n", dist2_cur);
            exit(-1);
        }
    }

    if (t == toc) {
        return false;
    }
    else {
        toc = t * (1 - eta);
        return true;
    }
}

template <class T>
bool Point_Edge_CCD(
    Eigen::Matrix<T, 3, 1> p, 
    Eigen::Matrix<T, 3, 1> e0,
    Eigen::Matrix<T, 3, 1> e1,
    Eigen::Matrix<T, 3, 1> dp, 
    Eigen::Matrix<T, 3, 1> de0,
    Eigen::Matrix<T, 3, 1> de1,
    T eta, T thickness, T& toc)
{
    Eigen::Matrix<T, 3, 1> mov = (dp + de0 + de1) / 3;
    de0 -= mov;
    de1 -= mov;
    dp -= mov;
    T maxDispMag = dp.norm() + std::sqrt(std::max(de0.squaredNorm(), de1.squaredNorm()));
    if (maxDispMag == 0) {
        return false;
    }

    T dist2_cur;
    Point_Edge_Distance_Unclassified(p, e0, e1, dist2_cur);
    T dist_cur = std::sqrt(dist2_cur);
    T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    T toc_prev = toc;
    toc = 0;
    while (true) {
        T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness) / ((dist_cur + thickness) * maxDispMag);
        
        p += tocLowerBound * dp;
        e0 += tocLowerBound * de0;
        e1 += tocLowerBound * de1;
        Point_Edge_Distance_Unclassified(p, e0, e1, dist2_cur);
        dist_cur = std::sqrt(dist2_cur);
        if (toc && (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap) {
            break;
        }
        
        toc += tocLowerBound;
        if (toc > toc_prev) {
            return false;
        }
    }

    return true;
}

template <class T>
bool Point_Point_CCD(
    Eigen::Matrix<T, 3, 1> p0, 
    Eigen::Matrix<T, 3, 1> p1,
    Eigen::Matrix<T, 3, 1> dp0, 
    Eigen::Matrix<T, 3, 1> dp1,
    T eta, T thickness, T& toc)
{
    Eigen::Matrix<T, 3, 1> mov = (dp0 + dp1) / 2;
    dp1 -= mov;
    dp0 -= mov;
    T maxDispMag = dp0.norm() + dp1.norm();
    if (maxDispMag == 0) {
        return false;
    }

    T dist2_cur;
    Point_Point_Distance(p0, p1, dist2_cur);
    T dist_cur = std::sqrt(dist2_cur);
    T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    T toc_prev = toc;
    toc = 0;
    while (true) {
        T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness) / ((dist_cur + thickness) * maxDispMag);

        p0 += tocLowerBound * dp0;
        p1 += tocLowerBound * dp1;
        Point_Point_Distance(p0, p1, dist2_cur);
        dist_cur = std::sqrt(dist2_cur);
        if (toc && (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap) {
            break;
        }
        
        toc += tocLowerBound;
        if (toc > toc_prev) {
            return false;
        }
    }

    return true;
}

}