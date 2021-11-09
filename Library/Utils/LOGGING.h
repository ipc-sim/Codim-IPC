#pragma once

#include <Math/VECTOR.h>
#include <Utils/PARAMETER.h>
#include <Eigen/Eigen>

namespace JGSL {

//#####################################################################
// Function JGSL_DEBUG
// for vectors
//#####################################################################
template <class T, int dim>
void JGSL_DEBUG(const VECTOR<T, dim>& v) {
    for (int i = 0; i < dim; ++i) printf("%.4f\t\n", (float)v(i));
    puts("");
}

//#####################################################################
// Function JGSL_DEBUG
// for matricies
//#####################################################################
template <class T, int dim>
void JGSL_DEBUG(const MATRIX<T, dim>& m) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) printf("%.4f\t", (float)m(i, j));
        puts("");
    }
    puts("");
}

template<class T, int size>
void JGSL_DEBUG(const Eigen::Matrix<T, size, size>& m) {
    int n = ((size == Eigen::Dynamic) ? m.rows() : size);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) printf("%.4f\t", (float)m(i, j));
        puts("");
    }
    puts("");
}

void JGSL_FILE(std::string name, double value) {
    static std::map<std::string, bool> first_run;
    std::string fp = PARAMETER::Get("Basic.log_folder", std::string("")) + name + std::string(".txt");
    if (!first_run[name]) {
        first_run[name] = true;
        FILE* f = fopen(fp.c_str(), "w");
        fclose(f);
    }
    FILE* f = fopen(fp.c_str(), "a");
    fprintf(f, "%.20f\n", value);
    fclose(f);
}

std::map<std::string, double> analyze_min;
std::map<std::string, double> analyze_max;
std::map<std::string, double> analyze_total;
std::map<std::string, double> analyze_number;
void JGSL_ANALYZE(std::string name, double value) {
    JGSL_FILE(name, value);
    ++analyze_number[name];
    if (analyze_number[name] == 1) {
        analyze_min[name] = value;
        analyze_max[name] = value;
    }
    analyze_min[name] = std::min(analyze_min[name], value);
    analyze_max[name] = std::max(analyze_max[name], value);
    analyze_total[name] += value;
    printf("%s :\n", name.c_str());
    printf("Min %.20f\n", analyze_min[name]);
    printf("Max %.20f\n", analyze_max[name]);
    printf("Average %.20f\n", analyze_total[name] / analyze_number[name]);
    printf("Total %.20f\n\n", analyze_total[name]);
}

}