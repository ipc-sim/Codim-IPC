#pragma once

#include <pybind11/pybind11.h>

#include <Math/CSR_MATRIX.h>

#ifdef CHOLMOD_DIRECT_SOLVER
#include "cholmod.h"
#endif

namespace JGSL {

template <class T>
bool Solve_Direct(
    CSR_MATRIX<T> &sysMtr, 
    const std::vector<T> &rhs, 
    std::vector<T> &sol) 
{
    if (sysMtr.Get_Matrix().rows() != rhs.size()) {
        printf("sysMtr dimension does not match with rhs!\n");
        return false;
    }

#ifdef CHOLMOD_DIRECT_SOLVER
    cholmod_common cm;
    cholmod_start(&cm);

    // setup matrix
    cholmod_sparse* A = cholmod_allocate_sparse(sysMtr.Get_Matrix().rows(), 
        sysMtr.Get_Matrix().cols(), sysMtr.Get_Matrix().nonZeros(), 
        true, true, -1, CHOLMOD_REAL, &cm);
    void *Ax = A->x;
    void *Ap = A->p;
    void *Ai = A->i;
    A->i = sysMtr.Get_Matrix().innerIndexPtr();
    A->p = sysMtr.Get_Matrix().outerIndexPtr();
    A->x = sysMtr.Get_Matrix().valuePtr();

    // factorization
    cholmod_factor* L = cholmod_analyze(A, &cm);
    cholmod_factorize(A, L, &cm);
    if (cm.status == CHOLMOD_NOT_POSDEF) {
        return false;
    }

    // back solve
    cholmod_dense *b = cholmod_allocate_dense(rhs.size(), 1, rhs.size(), CHOLMOD_REAL, &cm);
    void *bx = b->x;
    b->x = const_cast<T*>(rhs.data());
    cholmod_dense* x = cholmod_solve(CHOLMOD_A, L, b, &cm);
    sol.resize(rhs.size());
    std::memcpy(sol.data(), x->x, sol.size() * sizeof(T));
    cholmod_free_dense(&x, &cm);

    // free memory
    A->i = Ai;
    A->p = Ap;
    A->x = Ax;
    cholmod_free_sparse(&A, &cm);
    cholmod_free_factor(&L, &cm);
    b->x = bx;
    cholmod_free_dense(&b, &cm);
    cholmod_finish(&cm);

    return true;
#else
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver;
    solver.compute(sysMtr.Get_Matrix());
    if (solver.info() != Eigen::Success) {
        printf("Eigen::SimplicialLDLT factorization failed\n");
        return false;
    }
    else {
        Eigen::VectorXd rhsE(rhs.size());
        std::memcpy(rhsE.data(), rhs.data(), sizeof(T) * rhs.size());
        Eigen::VectorXd solE = solver.solve(rhsE);
        if (solver.info() != Eigen::Success) {
            printf("Eigen::SimplicialLDLT back solve failed\n");
            return false;
        }
        else {
            sol.resize(solE.size());
            std::memcpy(sol.data(), solE.data(), sizeof(T) * solE.size());
            return true;
        }
    }
#endif
}

#ifdef CHOLMOD_DIRECT_SOLVER
template <class T>
class Solver_Direct_Helper {
public:
    cholmod_common cm;
    cholmod_sparse* A;
    cholmod_dense *b;
    void *Ax, *Ap, *Ai, *bx;
    cholmod_factor* L;

    Solver_Direct_Helper(CSR_MATRIX<T> &sysMtr) {
        cholmod_common cm;
        cholmod_start(&cm);

        // setup matrix
        A = cholmod_allocate_sparse(sysMtr.Get_Matrix().rows(),
                                                    sysMtr.Get_Matrix().cols(), sysMtr.Get_Matrix().nonZeros(),
                                                    true, true, -1, CHOLMOD_REAL, &cm);
        Ax = A->x;
        Ap = A->p;
        Ai = A->i;
        A->i = sysMtr.Get_Matrix().innerIndexPtr();
        A->p = sysMtr.Get_Matrix().outerIndexPtr();
        A->x = sysMtr.Get_Matrix().valuePtr();

        // factorization
        L = cholmod_analyze(A, &cm);
        cholmod_factorize(A, L, &cm);
        if (cm.status == CHOLMOD_NOT_POSDEF) {
            exit(0);
        }

        // prepare rhs data structure
        b = cholmod_allocate_dense(sysMtr.Get_Matrix().rows(), 1, sysMtr.Get_Matrix().rows(), CHOLMOD_REAL, &cm);
        bx = b->x;
    }

    ~Solver_Direct_Helper() {
        // free memory
        A->i = Ai;
        A->p = Ap;
        A->x = Ax;
        cholmod_free_sparse(&A, &cm);
        cholmod_free_factor(&L, &cm);
        b->x = bx;
        cholmod_free_dense(&b, &cm);
        cholmod_finish(&cm);
    }

    void Solve(const std::vector<T> &rhs, std::vector<T> &sol) {
        // back solve
        b->x = const_cast<T*>(rhs.data());
        cholmod_dense* x = cholmod_solve(CHOLMOD_A, L, b, &cm);
        sol.resize(rhs.size());
        std::memcpy(sol.data(), x->x, sol.size() * sizeof(T));
        cholmod_free_dense(&x, &cm);
    }
};
#else
template <class T>
class Solver_Direct_Helper {
public:
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver;
    Solver_Direct_Helper(CSR_MATRIX<T> &sysMtr) {
        solver.compute(sysMtr.Get_Matrix());
        if (solver.info() != Eigen::Success) {
            exit(0);
        }
    }
    void Solve(const std::vector<T> &rhs, std::vector<T> &sol) {
        Eigen::VectorXd rhsE(rhs.size());
        std::memcpy(rhsE.data(), rhs.data(), sizeof(T) * rhs.size());
        Eigen::VectorXd solE = solver.solve(rhsE);
        if (solver.info() != Eigen::Success) {
            exit(0);
        }
        else {
            sol.resize(solE.size());
            std::memcpy(sol.data(), solE.data(), sizeof(T) * solE.size());
        }
    }
};
#endif

}