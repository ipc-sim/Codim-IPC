#pragma once
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#ifdef ENABLE_AMGCL_CUDA
#include <amgcl/backend/vexcl.hpp>
#endif
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/reorder.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/profiler.hpp>

#include <pybind11/pybind11.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Math/CSR_MATRIX.h>

#define check_init double min_delta = 1e30; size_t nrows, ncols; std::tie(nrows, ncols) = A.Size();
#define check_optimum if(delta < min_delta && resid <= 1e-5) { min_delta = delta; prm = prm_tmp; }

namespace amgcl { profiler<> prof; }

namespace JGSL {
    using amgcl::prof;

boost::property_tree::ptree Default_Params(){
    boost::property_tree::ptree prm;
    prm.put("precond.class", "amg");
    prm.put("precond.coarsening.type", "smoothed_aggregation");
    prm.put("solver.type", "cg");
    return prm;
}

template<int dim>
boost::property_tree::ptree Default_FEM_Params(){
    boost::property_tree::ptree prm;
    prm.put("precond.class", "amg");
    prm.put("precond.coarsening.type", "smoothed_aggregation");
    prm.put("precond.coarsening.aggr.block_size", dim);
    prm.put("solver.type", "lgmres");
    prm.put("solver.M", 100);
    prm.put("precond.coarsening.aggr.eps_strong", 0.0);
    prm.put("precond.relax.type", "gauss_seidel");
    //TODO: blockwise backend
    return prm;
}

template <class T>
std::tuple<size_t, double, double> Solve(CSR_MATRIX<T> &A, const std::vector<T> &residual, std::vector<T> &x, T rel_tol=-1, int max_iter=-1, boost::property_tree::ptree prm=Default_Params(), bool verbose=false) {
    if (rel_tol > 0) 
        prm.put("solver.tol", rel_tol); // relative
    else
        prm.put("solver.tol", T(1e-8));
    if (max_iter > 0)
        prm.put("solver.maxiter", max_iter);
    else
        prm.put("solver.maxiter", 100);

    bool reorder = false;
    if(prm.count("reorder")){
        reorder = prm.get<bool>("reorder");
        prm.erase("reorder");
    }

#ifdef ENABLE_AMGCL_CUDA
    typedef amgcl::backend::vexcl<T> Backend;
#else
    typedef amgcl::backend::builtin<T> Backend;
#endif
    using Solver = amgcl::make_solver<
        amgcl::runtime::preconditioner<Backend>,
        amgcl::runtime::solver::wrapper<Backend>>;
    
    typename Backend::params bprm;

    prof.tic("AMGCL Solve");

    if (reorder) {
        amgcl::adapter::reorder<> perm(A.Get_Matrix());
#ifdef ENABLE_AMGCL_CUDA
        vex::Context ctx(vex::Filter::Env);
        if (verbose)
            std::cout << "Computation Context: " << ctx;
        bprm.q = ctx;
        Solver solver(perm(A.Get_Matrix()), prm, bprm);
#else
        Solver solver(perm(A.Get_Matrix()), prm);
#endif
        if(verbose)
            std::cout << solver.precond() << std::endl;
        size_t iters;
        double resid;

        
        std::vector<T> F(residual.data(), residual.data() + residual.size());
        std::vector<T> X(x.data(), x.data() + x.size());

        std::vector<T> tmp(A.Get_Matrix().rows());
        perm.forward(F, tmp);
        auto f_b = Backend::copy_vector(tmp, bprm);
        perm.forward(X, tmp);
        auto x_b = Backend::copy_vector(tmp, bprm);

        std::tie(iters, resid) = solver(*f_b, *x_b);
        if(verbose){
            std::cout << "Iterations: " << iters << std::endl;
            std::cout << "Error:      " << resid << std::endl;
        }

#ifdef ENABLE_AMGCL_CUDA
        vex::copy(*x_b, tmp);
#else
        std::copy(&(*x_b)[0], &(*x_b)[0] + x.size(), tmp.data());
#endif
        T* xptr = x.data();
        perm.inverse(tmp, xptr);
        double delta = prof.toc("AMGCL Solve");
        return std::make_tuple(iters, resid, delta);
    } else {
#ifdef ENABLE_AMGCL_CUDA
        vex::Context ctx(vex::Filter::Env);
        if (verbose)
            std::cout << "Computation Context: " << ctx;
        bprm.q = ctx;
        Solver solver(A.Get_Matrix(), prm, bprm);
#else
        Solver solver(A.Get_Matrix(), prm);
#endif
        if(verbose)
            std::cout << solver.precond() << std::endl;
        size_t iters;
        double resid;

        std::vector<T> F(residual.data(), residual.data() + residual.size());
        std::vector<T> X(x.size());

        auto f_b = Backend::copy_vector(F, bprm);
        auto x_b = Backend::copy_vector(X, bprm);

        std::tie(iters, resid) = solver(*f_b, *x_b);
        if(verbose) {
            std::cout << "Iterations: " << iters << std::endl;
            std::cout << "Error:      " << resid << std::endl;
        }

#ifdef ENABLE_AMGCL_CUDA
        vex::copy(*x_b, X);
        std::copy(&X[0], &X[0] + x.size(), x.data());
#else
        std::copy(&(*x_b)[0], &(*x_b)[0] + x.size(), x.data());
#endif
        double delta = prof.toc("AMGCL Solve");
        return std::make_tuple(iters, resid, delta);
    }
}

template <class T>
void Write_Matrix(std::string filename, CSR_MATRIX<T> A){
    amgcl::io::mm_write("matrix.amgcl", A);
}

template <class T>
void Export_AMGCL_Impl(py::module& m) { 
    m.def("Solve", &Solve<T>);
    m.def("Params", 
        [](pybind11::dict params_dict) {
            boost::property_tree::ptree params = Default_Params();
            for (auto item : params_dict) {
                params.put(py::cast<std::string>(item.first), item.second);
            }
            return params;});
}

void Export_AMGCL(py::module& m) {
    py::class_<boost::property_tree::ptree>(m, "BOOST_PTREE")
        .def(py::init<>())
        .def("__repr__",
            [](const boost::property_tree::ptree &pt){
                std::ostringstream oss;
                boost::property_tree::json_parser::write_json(oss, pt);
                return oss.str();
            });
    Export_AMGCL_Impl<float>(m);
    Export_AMGCL_Impl<double>(m);
}

}