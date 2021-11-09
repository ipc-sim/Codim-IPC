#pragma once

#include <FEM/Shell/UTILS.h>
#include <FEM/Shell/MEMBRANE.h>
#include <FEM/Shell/BENDING.h>
#include <FEM/Shell/INEXT.h>
#include <FEM/Shell/ANISO_INEXT.h>
#include <FEM/Shell/Rod/ROD_STRETCHING.h>
#include <FEM/Shell/Rod/ROD_BENDING.h>
#include <FEM/Shell/DIRICHLET.h>
#include <FEM/Shell/STITCH.h>

namespace JGSL {

template <class T, int dim, bool KL, bool elasticIPC, bool flow>
bool Compute_IncPotential(
    MESH_ELEM<dim - 1>& Elem, T h, 
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    const T thickness, T bendingStiffMult,
    const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xtilde,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    bool withCollision,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo,
    T dHat2, T kappa[],
    bool staticSolve,
    const std::vector<T>& b,
    MESH_ELEM<dim>& tet,
    MESH_ELEM_ATTR<T, dim>& tetAttr,
    FIXED_COROTATED<T, dim>& tetElasticityAttr,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    T k_stitch,
    T& value)
{
    TIMER_FLAG("Compute_IncPotential");
    value = 0;

    if constexpr (flow) {
        Eigen::VectorXd Lx = Eigen::VectorXd::Zero(X.size * dim);
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F, vol, lambda, mu] = data;
            const Eigen::Matrix<T, dim, 1> x[dim] = {
                Eigen::Matrix<T, dim, 1>(std::get<0>(X.Get_Unchecked(elemVInd[0])).data),
                Eigen::Matrix<T, dim, 1>(std::get<0>(X.Get_Unchecked(elemVInd[1])).data),
                Eigen::Matrix<T, dim, 1>(std::get<0>(X.Get_Unchecked(elemVInd[2])).data)
            };
            for (int i = 0; i < dim; ++i) {
                Lx.template segment<dim>(elemVInd[i] * dim) += vol / 6 * (x[(i + 1) % dim] +
                    x[(i + 2) % dim] - 2 * x[i]);
            }
        });
        X.Each([&](int id, auto data) {
            auto &[x] = data;
            for (int d = 0; d < dim; ++d) {
                value += -h * 0.5 * x[d] * Lx[id * dim + d];
            }
        });
    }
    else {
        if (kappa_s[0] > 0) {
            if (!Compute_Inextensibility_Energy(Elem, staticSolve ? 1.0 : h, s, sHat, kappa_s, DBCb, X, nodeAttr, elemAttr, elasticityAttr, value)) {
                return false;
            }
        }

        // membrane
        if (fiberStiffMult[0] > 0 || fiberStiffMult[1] > 0) {
            if (Check_Fiber_Feasibility(Elem, staticSolve ? 1.0 : h, fiberLimit, DBCb, X, nodeAttr, elemAttr, elasticityAttr)) {
                Compute_Fiber_Energy(Elem, staticSolve ? 1.0 : h, fiberStiffMult, fiberLimit, DBCb, X, nodeAttr, elemAttr, elasticityAttr, value);
            }
            else {
                return false;
            }
        }
        else {
            Compute_Membrane_Energy(Elem, staticSolve ? 1.0 : h, DBCb, X, nodeAttr, elemAttr, elasticityAttr, value);
        }
        if (bendingStiffMult) {
            Compute_Bending_Energy<T, dim, KL>(Elem, staticSolve ? 1.0 : h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, DBCb, X, nodeAttr, elemAttr, elasticityAttr, value);
        }
        
        // volumetric elasticity
        Compute_Deformation_Gradient(X, tet, tetAttr, tetElasticityAttr);
        std::vector<int> degenerate(tet.size);
        tetElasticityAttr.Par_Each([&](const int i, auto data) {
            auto& [F, vol, lambda, mu] = data;

            degenerate[i] = (F.determinant() <= 0);
        });
        int hasDegeneracy = std::accumulate(degenerate.begin(), degenerate.end(), 0);
        if (hasDegeneracy) {
            return false;
        }
        NEOHOOKEAN_FUNCTOR<T, dim>::Compute_Psi(tetElasticityAttr, h * h, tetAttr, value);

        // rod spring
        Compute_Rod_Spring_Energy(X, rod, rodInfo, h, value);
        Compute_Rod_Bending_Energy(X, rodHinge, rodHingeInfo, h, value);

        // garment
        Compute_Stitch_Energy(X, stitchInfo, stitchRatio, DBCb, k_stitch, h, value);
    }

    if (staticSolve) {
        std::vector<T> xb(X.size, T(0));
        X.Par_Each([&](int id, auto data) {
            auto &[x] = data;
            xb[id] += x[0] * b[id * dim];
            xb[id] += x[1] * b[id * dim + 1];
            if constexpr (dim == 3) {
                xb[id] += x[2] * b[id * dim + 2];
            }
        });
        value -= std::accumulate(xb.begin(), xb.end(), T(0));
    }
    else {
        // inertia
        Eigen::VectorXd xDiff(X.size * dim);
        X.Join(Xtilde).Par_Each([&](int id, auto data) {
            auto &[x, xtilde] = data;
            xDiff[id * dim] = x[0] - xtilde[0];
            xDiff[id * dim + 1] = x[1] - xtilde[1];
            if constexpr (dim == 3) {
                xDiff[id * dim + 2] = x[2] - xtilde[2];
            }
        });
        Eigen::VectorXd MXDiff = M.Get_Matrix() * xDiff;
        value += 0.5 * MXDiff.dot(xDiff);
    }

    if (withCollision) {
        // IPC
        Compute_Barrier<T, dim, elasticIPC>(X, nodeAttr, constraintSet, stencilInfo, dHat2, kappa, thickness, value);
    }

    return true;
}

template <class T, int dim, bool KL, bool elasticIPC, bool flow>
void Compute_IncPotential_Gradient(
    MESH_ELEM<dim - 1>& Elem, T h, 
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    const T thickness, T bendingStiffMult,
    const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xtilde,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 2 * dim * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    bool withCollision,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo,
    T dHat2, T kappa[],
    bool staticSolve,
    const std::vector<T>& b,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    MESH_ELEM<dim>& tet,
    MESH_ELEM_ATTR<T, dim>& tetAttr,
    FIXED_COROTATED<T, dim>& tetElasticityAttr,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    T k_stitch)
{
    TIMER_FLAG("Compute_IncPotential_Gradient");
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));

    if constexpr (flow) {
        Eigen::VectorXd Lx = Eigen::VectorXd::Zero(X.size * dim);
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F, vol, lambda, mu] = data;
            const Eigen::Matrix<T, dim, 1> x[dim] = {
                Eigen::Matrix<T, dim, 1>(std::get<0>(X.Get_Unchecked(elemVInd[0])).data),
                Eigen::Matrix<T, dim, 1>(std::get<0>(X.Get_Unchecked(elemVInd[1])).data),
                Eigen::Matrix<T, dim, 1>(std::get<0>(X.Get_Unchecked(elemVInd[2])).data)
            };
            for (int i = 0; i < dim; ++i) {
                Lx.template segment<dim>(elemVInd[i] * dim) += vol / 6 * (x[(i + 1) % dim] +
                    x[(i + 2) % dim] - 2 * x[i]);
            }
        });
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            for (int d = 0; d < dim; ++d) {
                g[d] += -h * Lx[id * dim + d];
            }
        });
    }
    else {
        // elasticity
        if (fiberStiffMult[0] > 0 || fiberStiffMult[1] > 0) {
            Compute_Fiber_Gradient(Elem, staticSolve ? 1.0 : h, fiberStiffMult, fiberLimit, DBCb, X, nodeAttr, elemAttr, elasticityAttr);
        }
        else {
            Compute_Membrane_Gradient(Elem, staticSolve ? 1.0 : h, DBCb, X, nodeAttr, elemAttr, elasticityAttr);
        }
        if (kappa_s[0] > 0) {
            Compute_Inextensibility_Gradient(Elem, staticSolve ? 1.0 : h, s, sHat, kappa_s, DBCb, X, nodeAttr, elemAttr, elasticityAttr);
        }
        if (bendingStiffMult) {
            Compute_Bending_Gradient<T, dim, KL>(Elem, staticSolve ? 1.0 : h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, DBCb, X, nodeAttr, elemAttr, elasticityAttr);
        }

        // volumetric elasticity
        NEOHOOKEAN_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress(tetElasticityAttr, h * h, tetAttr);
        Elem_To_Node(tet, tetAttr, nodeAttr);

        // rod spring
        Compute_Rod_Spring_Gradient(X, rod, rodInfo, h, nodeAttr);
        Compute_Rod_Bending_Gradient(X, rodHinge, rodHingeInfo, h, nodeAttr);

        // garment
        Compute_Stitch_Gradient(X, stitchInfo, stitchRatio, DBCb, k_stitch, h, nodeAttr);
    }

    // inertia
    if (staticSolve) {
        nodeAttr.Par_Each([&](int id, auto data){
            auto &[x0, v, g, m] = data;
            g[0] -= b[id * dim];
            g[1] -= b[id * dim + 1];
            if constexpr (dim == 3) {
                g[2] -= b[id * dim + 2];
            }
        });
    }
    else {
        Eigen::VectorXd xDiff(X.size * dim);
        X.Join(Xtilde).Par_Each([&](int id, auto data) {
            auto &[x, xtilde] = data;
            xDiff[id * dim] = x[0] - xtilde[0];
            xDiff[id * dim + 1] = x[1] - xtilde[1];
            if constexpr (dim == 3) {
                xDiff[id * dim + 2] = x[2] - xtilde[2];
            }
        });
        Eigen::VectorXd MXDiff = M.Get_Matrix() * xDiff;
        nodeAttr.Par_Each([&](int id, auto data){
            auto &[x0, v, g, m] = data;
            g[0] += MXDiff[id * dim];
            g[1] += MXDiff[id * dim + 1];
            if constexpr (dim == 3) {
                g[2] += MXDiff[id * dim + 2];
            }
        });
    }

    if (withCollision) {
        // IPC
        Compute_Barrier_Gradient<T, dim, elasticIPC>(X, constraintSet, stencilInfo, dHat2, kappa, thickness, nodeAttr);
    }
}

template <class T, int dim, bool KL, bool elasticIPC, bool flow>
void Compute_IncPotential_Hessian(
    MESH_ELEM<dim - 1>& Elem, T h, 
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    const T thickness, T bendingStiffMult,
    const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const std::vector<bool>& DBCb,
    const std::vector<bool>& DBCb_fixed,
    T DBCStiff,
    MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xn,
    MESH_NODE<T, dim>& Xtilde,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 2 * dim * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    bool withCollision,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo,
    std::vector<VECTOR<int, dim + 1>>& fricConstraintSet,
    std::vector<Eigen::Matrix<T, dim - 1, 1>>& closestPoint,
    std::vector<Eigen::Matrix<T, dim, dim - 1>>& tanBasis,
    std::vector<T>& normalForce,
    T dHat2, T kappa[], T mu, T epsv2,
    bool staticSolve,
    const std::vector<T>& b,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    MESH_ELEM<dim>& tet,
    MESH_ELEM_ATTR<T, dim>& tetAttr,
    FIXED_COROTATED<T, dim>& tetElasticityAttr,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    T k_stitch,
    bool projectSPD,
    CSR_MATRIX<T>& sysMtr)
{
    std::vector<Eigen::Triplet<T>> triplets;

    if constexpr (flow) {
        int tripletIndStart = triplets.size();
        triplets.resize(tripletIndStart + Elem.size * dim * dim * 3);
        Elem.Join(elasticityAttr).Par_Each([&](int id, auto data) {
            auto &[elemVInd, F, vol, lambda, mu] = data;
            for (int i = 0; i < dim; ++i) {
                for (int d = 0; d < dim; ++d) {
                    triplets[tripletIndStart + id * dim * dim * 3 + i * dim * 3 + d * 3] = 
                        Eigen::Triplet<T>(elemVInd[i] * dim + d, elemVInd[(i + 1) % dim] * dim + d, -h * vol / 6);
                    triplets[tripletIndStart + id * dim * dim * 3 + i * dim * 3 + d * 3 + 1] = 
                        Eigen::Triplet<T>(elemVInd[i] * dim + d, elemVInd[(i + 2) % dim] * dim + d, -h * vol / 6);
                    triplets[tripletIndStart + id * dim * dim * 3 + i * dim * 3 + d * 3 + 2] = 
                        Eigen::Triplet<T>(elemVInd[i] * dim + d, elemVInd[i] * dim + d, 2 * h * vol / 6);
                }
            }
        });
    }
    else {
        if (fiberStiffMult[0] > 0 || fiberStiffMult[1] > 0) {
            Compute_Fiber_Hessian(Elem, staticSolve ? 1.0 : h, projectSPD, fiberStiffMult, fiberLimit, DBCb, X, nodeAttr, elemAttr, elasticityAttr, triplets);
        }
        else {
            Compute_Membrane_Hessian(Elem, staticSolve ? 1.0 : h, projectSPD, DBCb, X, nodeAttr, elemAttr, elasticityAttr, triplets);
        }
        if (kappa_s[0] > 0) {
            Compute_Inextensibility_Hessian(Elem, staticSolve ? 1.0 : h, projectSPD, s, sHat, kappa_s, DBCb, X, nodeAttr, elemAttr, elasticityAttr, triplets);
        }
        if (bendingStiffMult) {
            Compute_Bending_Hessian<T, dim, KL>(Elem, staticSolve ? 1.0 : h, projectSPD, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, DBCb, X, nodeAttr, elemAttr, elasticityAttr, triplets);
        }
        
        // volumetric elasticity:
        {
            TIMER_FLAG("Compute_Volumetric_Elasticity_Hessian");
            typename NEOHOOKEAN_FUNCTOR<T, dim>::DIFFERENTIAL dP_div_dF;
            dP_div_dF.Reserve(tetAttr.size);
            for (int i = 0; i < tetAttr.size; ++i) {
                dP_div_dF.Insert(i, Eigen::Matrix<T, dim * dim, dim * dim>::Zero());
            }
            NEOHOOKEAN_FUNCTOR<T, dim>::Compute_First_PiolaKirchoff_Stress_Derivative(tetElasticityAttr, h * h, projectSPD, dP_div_dF);
            Elem_To_Node(tet, tetAttr, dP_div_dF, triplets);
        }

        Compute_Rod_Spring_Hessian(X, rod, rodInfo, h, projectSPD, triplets);
        Compute_Rod_Bending_Hessian(X, rodHinge, rodHingeInfo, h, projectSPD, triplets);

        // garment
        Compute_Stitch_Hessian(X, stitchInfo, stitchRatio, DBCb, k_stitch, h, projectSPD, triplets);
    }

    if (withCollision) {
        Compute_Barrier_Hessian<T, dim, elasticIPC>(X, nodeAttr, constraintSet, stencilInfo, dHat2, kappa, thickness, projectSPD, triplets);
        if (mu > 0) {
            Compute_Friction_Hessian(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, projectSPD, triplets);
        }
    }
    if (DBCStiff) {
        Compute_DBC_Hessian(X, nodeAttr, DBC, DBCStiff, triplets);
    }
    sysMtr.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
    if (!staticSolve) {
        TIMER_FLAG("add mass matrix");
        sysMtr.Get_Matrix() += M.Get_Matrix();
    }
    if (!DBCStiff) {
        // project Matrix for Dirichlet boundary condition
        sysMtr.Project_DBC(DBCb, dim);
        std::cout << "project Matrix for Dirichlet boundary condition" << std::endl;
    }
    else {
        sysMtr.Project_DBC(DBCb_fixed, dim);
    }
}

}