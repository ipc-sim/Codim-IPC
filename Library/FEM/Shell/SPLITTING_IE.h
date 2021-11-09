#pragma once

#include <FEM/Shell/IMPLICIT_EULER.h>

namespace JGSL {

template <class T, int dim, bool elasticIPC>
bool Compute_SLP_Energy( // SLP: strain limiting projection
    MESH_NODE<T, dim>& X, 
    MESH_NODE<T, dim>& Xn, 
    MESH_NODE<T, dim>& X_prim, 
    CSR_MATRIX<T>& M, 
    MESH_NODE_ATTR<T, dim>& nodeAttr, 
    MESH_ELEM<dim - 1>& Elem, 
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr, 
    FIXED_COROTATED<T, dim - 1>& elasticityAttr, 
    std::vector<bool>& DBCb, 
    std::vector<bool>& DBCb_fixed,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    T DBCStiff,
    bool staticSolve, T h, 
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s, 
    bool withCollision,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo, // weight, dHat2
    T dHat2, T kappa[], T thickness,
    const std::vector<VECTOR<int, dim + 1>>& fricConstraintSet,
    const std::vector<Eigen::Matrix<T, dim - 1, 1>>& closestPoint,
    const std::vector<Eigen::Matrix<T, dim, dim - 1>>& tanBasis,
    const std::vector<T>& normalForce,
    T epsv2, T mu,
    T& E) 
{
    TIMER_FLAG("Compute_SLP_Energy");
    E = 0;

    if (kappa_s[0] > 0) {
        if (!Compute_Inextensibility_Energy(Elem, staticSolve ? 1.0 : h, s, sHat, kappa_s, 
            DBCb, X, nodeAttr, elemAttr, elasticityAttr, E))
        {
            return false;
        }
    }

    Eigen::VectorXd xDiff(X.size * dim);
    X.Join(X_prim).Par_Each([&](int id, auto data) {
        auto &[x, xprim] = data;
        xDiff[id * dim] = x[0] - xprim[0];
        xDiff[id * dim + 1] = x[1] - xprim[1];
        if constexpr (dim == 3) {
            xDiff[id * dim + 2] = x[2] - xprim[2];
        }
    });
    Eigen::VectorXd MXDiff = M.Get_Matrix() * xDiff;
    E += 0.5 * MXDiff.dot(xDiff);

    if (withCollision) {
        Compute_Barrier<T, dim, elasticIPC>(X, nodeAttr, constraintSet, stencilInfo, dHat2, kappa, thickness, E);
        if (mu > 0) {
            Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, E);
        }
    }

    if (DBCStiff) {
        Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, E);
    }

    return true;
}

template <class T, int dim, bool elasticIPC>
void Compute_SLP_Gradient( // SLP: strain limiting projection
    MESH_NODE<T, dim>& X, 
    MESH_NODE<T, dim>& Xn, 
    MESH_NODE<T, dim>& X_prim, 
    CSR_MATRIX<T>& M, 
    MESH_NODE_ATTR<T, dim>& nodeAttr, 
    MESH_ELEM<dim - 1>& Elem, 
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr, 
    FIXED_COROTATED<T, dim - 1>& elasticityAttr, 
    std::vector<bool>& DBCb, 
    std::vector<bool>& DBCb_fixed,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    T DBCStiff, 
    bool staticSolve, T h, 
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    bool withCollision,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo, // weight, dHat2
    T dHat2, T kappa[], T thickness,
    const std::vector<VECTOR<int, dim + 1>>& fricConstraintSet,
    const std::vector<Eigen::Matrix<T, dim - 1, 1>>& closestPoint,
    const std::vector<Eigen::Matrix<T, dim, dim - 1>>& tanBasis,
    const std::vector<T>& normalForce,
    T epsv2, T mu,
    std::vector<T>& rhs) 
{
    TIMER_FLAG("Compute_SLP_Gradient");
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));

    if (kappa_s[0] > 0) {
        Compute_Inextensibility_Gradient(Elem, staticSolve ? 1.0 : h, s, sHat, kappa_s, 
            DBCb, X, nodeAttr, elemAttr, elasticityAttr);
    }

    Eigen::VectorXd xDiff(X.size * dim);
    X.Join(X_prim).Par_Each([&](int id, auto data) {
        auto &[x, xprim] = data;
        xDiff[id * dim] = x[0] - xprim[0];
        xDiff[id * dim + 1] = x[1] - xprim[1];
        if constexpr (dim == 3) {
            xDiff[id * dim + 2] = x[2] - xprim[2];
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

    if (withCollision) {
        Compute_Barrier_Gradient<T, dim, elasticIPC>(X, constraintSet, stencilInfo, dHat2, kappa, thickness, nodeAttr);
        if (mu > 0) {
            Compute_Friction_Gradient(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, nodeAttr);
        }
    }

    if (DBCStiff) {
        Compute_DBC_Gradient(X, nodeAttr, DBC, DBCStiff);
        for (int vI = 0; vI < DBCb_fixed.size(); ++vI) {
            if (DBCb_fixed[vI]) {
                std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(vI)).setZero();
            }
        }
    }
    else {
        // project rhs for Dirichlet boundary condition
        DBC.Par_Each([&](int id, auto data) {
            auto &[dbcI] = data;
            std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(dbcI(0))).setZero();
        });
        std::cout << "project rhs for Dirichlet boundary condition " << DBC.size << std::endl;
    }
    rhs.resize(X.size * dim);
    nodeAttr.Par_Each([&](int id, auto data) {
        auto &[x0, v, g, m] = data;
        rhs[id * dim] = -g[0];
        rhs[id * dim + 1] = -g[1];
        if constexpr (dim == 3) {
            rhs[id * dim + 2] = -g[2];
        }
    });
}

template <class T, int dim, bool elasticIPC>
void Compute_SLP_Hessian( // SLP: strain limiting projection
    MESH_NODE<T, dim>& X, 
    MESH_NODE<T, dim>& Xn, 
    MESH_NODE<T, dim>& X_prim, 
    CSR_MATRIX<T>& M, 
    MESH_NODE_ATTR<T, dim>& nodeAttr, 
    MESH_ELEM<dim - 1>& Elem, 
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr, 
    FIXED_COROTATED<T, dim - 1>& elasticityAttr, 
    std::vector<bool>& DBCb, 
    std::vector<bool>& DBCb_fixed, 
    VECTOR_STORAGE<T, dim + 1>& DBC,
    T DBCStiff,
    bool staticSolve, T h, 
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    bool withCollision,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo, // weight, dHat2
    T dHat2, T kappa[], T thickness,
    const std::vector<VECTOR<int, dim + 1>>& fricConstraintSet,
    const std::vector<Eigen::Matrix<T, dim - 1, 1>>& closestPoint,
    const std::vector<Eigen::Matrix<T, dim, dim - 1>>& tanBasis,
    const std::vector<T>& normalForce,
    T epsv2, T mu,
    bool projectSPD,
    CSR_MATRIX<T>& sysMtr)
{
    TIMER_FLAG("Compute_SLP_Hessian");
    std::vector<Eigen::Triplet<T>> triplets; 

    if (kappa_s[0] > 0) {
        Compute_Inextensibility_Hessian(Elem, staticSolve ? 1.0 : h, projectSPD, s, sHat, kappa_s, 
            DBCb, X, nodeAttr, elemAttr, elasticityAttr, triplets);
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
    sysMtr.Get_Matrix() += M.Get_Matrix();
    if (!DBCStiff) {
        // project Matrix for Dirichlet boundary condition
        sysMtr.Project_DBC(DBCb, dim);
        std::cout << "project Matrix for Dirichlet boundary condition" << std::endl;
    }
    else {
        sysMtr.Project_DBC(DBCb_fixed, dim);
    }
}

template <class T, int dim>
void DBC_Info_Init(
    MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& Xn,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM<dim - 1>& Elem, 
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    std::vector<bool>& DBCb, 
    std::vector<bool>& DBCb_fixed, 
    std::vector<T>& DBCDisp,
    bool staticSolve, T h, 
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    T& DBCStiff, T& DBCAlpha, T& DBCPenaltyXn)
{
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        int vI = dbcI(0);
        const VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(vI));
        
        DBCDisp[vI * dim] = dbcI(1) - x(0);
        DBCDisp[vI * dim + 1] = dbcI(2) - x(1);
        if constexpr (dim == 3) {
            DBCDisp[vI * dim + 2] = dbcI(3) - x(2);
            if (!(DBCDisp[vI * dim] || DBCDisp[vI * dim + 1] || DBCDisp[vI * dim + 2])) {
                DBCb_fixed[vI] = true;
            }
        }
        else {
            if (!(DBCDisp[vI * dim] || DBCDisp[vI * dim + 1])) {
                DBCb_fixed[vI] = true;
            }
        }

        DBCb[dbcI(0)] = true; // bool array cannot be written in parallel by entries
    });
    if (kappa_s[0] > 0) {
        bool valid = false;
        do {
            DBC.Each([&](int id, auto data) {
                auto &[dbcI] = data;
                VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
                const VECTOR<T, dim> &xn = std::get<0>(Xn.Get_Unchecked(dbcI(0)));

                x(0) = xn[0] + DBCAlpha * (dbcI(1) - xn[0]);
                x(1) = xn[1] + DBCAlpha * (dbcI(2) - xn[1]);
                if constexpr (dim == 3) {
                    x(2) = xn[2] + DBCAlpha * (dbcI(3) - xn[2]);
                }
            });

            valid = true;
            if (kappa_s[0] > 0) {
                T E;
                valid &= Compute_Inextensibility_Energy(Elem, staticSolve ? 1.0 : h, s, sHat, kappa_s, 
                    std::vector<bool>(X.size, false), X, nodeAttr, elemAttr, elasticityAttr, E);
            }

            if (!valid) {
                DBCAlpha /= 2.0;
            }
        } while (!valid);
        printf("DBCAlpha under inextensibility: %le\n", DBCAlpha);
        Xn.deep_copy_to(X);
    }
    if (DBCAlpha == 1) {
        DBC.Each([&](int id, auto data) {
            auto &[dbcI] = data;
            VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
            x(0) = dbcI(1);
            x(1) = dbcI(2);
            if constexpr (dim == 3) {
                x(2) = dbcI(3);
            }
        });
        printf("DBC handled\n");
    }
    else {
        printf("moved DBC by %le, turn on Augmented Lagrangian\n", DBCAlpha);
        DBCStiff = 1e8;
        Compute_DBC_Dist2(Xn, DBC, DBCPenaltyXn);
    }
}

template <class T, int dim, bool elasticIPC>
void DBC_Dual_Update(
    MESH_NODE<T, dim>& X, 
    MESH_NODE<T, dim>& Xn, 
    MESH_NODE<T, dim>& X_prim, 
    CSR_MATRIX<T>& M, 
    MESH_NODE_ATTR<T, dim>& nodeAttr, 
    MESH_ELEM<dim - 1>& Elem, 
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr, 
    FIXED_COROTATED<T, dim - 1>& elasticityAttr, 
    std::vector<bool>& DBCb, 
    std::vector<bool>& DBCb_fixed,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    T& DBCStiff,
    bool staticSolve, T h, 
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s, 
    bool withCollision,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo, // weight, dHat2
    T dHat2, T kappa[], T thickness,
    const std::vector<VECTOR<int, dim + 1>>& fricConstraintSet,
    const std::vector<Eigen::Matrix<T, dim - 1, 1>>& closestPoint,
    const std::vector<Eigen::Matrix<T, dim, dim - 1>>& tanBasis,
    const std::vector<T>& normalForce,
    T epsv2, T mu,
    T DBCPenaltyXn, T& Eprev,
    T& infNorm, T NewtonTol)
{
    T penaltyCur = 0;
    Compute_DBC_Dist2(X, DBC, penaltyCur);
    T progress = 1 - std::sqrt(penaltyCur / DBCPenaltyXn);
    printf("MDBC progress: %le\n", progress);

    if(progress < 0.99) {
        //TODO: update Augmented Lagrangian parameters if necessary
        if (infNorm < NewtonTol * 10) {
            if (DBCStiff < 1e10) {
                DBCStiff *= 2;
                printf("updated DBCStiff to %le\n", DBCStiff);
                
                Compute_SLP_Energy<T, dim, elasticIPC>(X, Xn, X_prim, M, nodeAttr, Elem, elemAttr, elasticityAttr, 
                    DBCb, DBCb_fixed, DBC, DBCStiff, staticSolve, h, s, sHat, kappa_s, 
                    withCollision, constraintSet, stencilInfo, dHat2, kappa, thickness,
                    fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2, mu, Eprev);
            }
        }
        infNorm = NewtonTol * 10; // ensures not exit Newton loop
    }
    else {
        DBCStiff = 0;
        printf("DBC moved to target, turn off Augmented Lagrangian\n");

        Compute_SLP_Energy<T, dim, elasticIPC>(X, Xn, X_prim, M, nodeAttr, Elem, elemAttr, elasticityAttr, 
            DBCb, DBCb_fixed, DBC, DBCStiff, staticSolve, h, s, sHat, kappa_s, 
            withCollision, constraintSet, stencilInfo, dHat2, kappa, thickness,
            fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2, mu, Eprev);
    }
}

template <class T, int dim, bool KL, bool elasticIPC>
int Advance_One_Step_SIE_Discrete_Shell(
    MESH_ELEM<dim - 1>& Elem,
    const std::vector<VECTOR<int, 2>>& seg,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    const T thickness, T bendingStiffMult,
    const VECTOR<T, 4>& fiberStiffMult, // aniso strain-limiting
    const VECTOR<T, 3>& fiberLimit, // aniso strain-limiting
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,  // isotropic strain-limiting
    const std::vector<T>& b, 
    T h, T NewtonTol,
    bool withCollision,
    T dHat2, VECTOR<T, 3>& kappaVec,
    T mu, T epsv2,
    const std::vector<int>& compNodeRange,
    const std::vector<T>& muComp,
    bool staticSolve,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    MESH_ELEM<dim>& tet,
    MESH_ELEM_ATTR<T, dim>& tetAttr,
    FIXED_COROTATED<T, dim>& tetElasticityAttr,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo,
    const std::vector<int>& particle,
    const std::string& outputFolder)
{
    Eigen::setNbThreads(1);
    TIMER_FLAG("splitting_implicitEuler");

    MESH_NODE<T, dim> Xn;
    Append_Attribute(X, Xn);

    //====== elastodynamic substep
    VECTOR<T, 2> zero2T(0, 0);
    std::vector<VECTOR<int, 3>> stitchInfo;
    std::vector<T> stitchRatio;
    int PNIter = Advance_One_Step_IE_Discrete_Shell<T, dim, KL, elasticIPC>(
        Elem, seg, DBC, edge2tri, edgeStencil, edgeInfo,
        thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
        s, sHat, zero2T, // kappa_s
        b, h, NewtonTol, false, // withCollision
        dHat2, kappaVec, mu, epsv2, 1, compNodeRange, muComp, staticSolve,
        X, nodeAttr, M, elemAttr, elasticityAttr, tet, tetAttr, tetElasticityAttr, 
        rod, rodInfo, rodHinge, rodHingeInfo, stitchInfo, stitchRatio, 0, particle, outputFolder);
    FILE *out = fopen((outputFolder + "/counter.txt").c_str(), "a+");
    fprintf(out, "%d", PNIter);
    fclose(out);

    //====== strain-limiting substep
    //TODO: MDBC contact, adaptive kappa, others...;
    MESH_NODE<T, dim> X_prim;
    Append_Attribute(X, X_prim);
    Xn.deep_copy_to(X);

    CSR_MATRIX<T> sysMtr;
    std::vector<T> rhs(X.size * dim), sol(X.size * dim);

    // compute contact primitives
    T kappa[] = {kappaVec[0], kappaVec[1], kappaVec[2]}; // dumb pybind does not support c array
    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    std::vector<VECTOR<int, 3>> boundaryTri;
    std::vector<T> BNArea, BEArea, BTArea;
    VECTOR<int, 2> codimBNStartInd;
    if (withCollision) {
        if constexpr (dim == 2) {
            //TODO
        }
        else {
            BASE_STORAGE<int> TriVI2TetVI;
            BASE_STORAGE<VECTOR<int, 3>> Tri;
            Find_Surface_TriMesh<T, false>(X, tet, TriVI2TetVI, Tri);
            Append_Attribute(Elem, Tri);

            Find_Surface_Primitives_And_Compute_Area(X, Tri, boundaryNode, boundaryEdge, boundaryTri,
                BNArea, BEArea, BTArea);
            
            boundaryEdge.insert(boundaryEdge.end(), seg.begin(), seg.end());
            for (const auto& segI : seg) {
                boundaryNode.emplace_back(segI[0]);
                boundaryNode.emplace_back(segI[1]);
                //TODO: handle duplicates
            }

            boundaryEdge.insert(boundaryEdge.end(), rod.begin(), rod.end());
            BEArea.reserve(boundaryEdge.size());
            std::map<int, T> rodNodeArea;
            int segIInd = 0;
            for (const auto& segI : rod) {
                const VECTOR<T, dim>& v0 = std::get<0>(X.Get_Unchecked(segI[0]));
                const VECTOR<T, dim>& v1 = std::get<0>(X.Get_Unchecked(segI[1]));
                BEArea.emplace_back((v0 - v1).length() * M_PI * rodInfo[segIInd][2] / 6); // 1/6 of the cylinder surface participate in one contact
                
                rodNodeArea[segI[0]] += BEArea.back() / 2;
                rodNodeArea[segI[1]] += BEArea.back() / 2;

                BEArea.back() /= 2; // due to PE approx of \int_E PP and EE approx of \int_E PE
                
                ++segIInd;
            }
            codimBNStartInd[0] = boundaryNode.size();
            boundaryNode.reserve(boundaryNode.size() + rodNodeArea.size());
            BNArea.reserve(BNArea.size() + rodNodeArea.size());
            for (const auto& nodeI : rodNodeArea) {
                boundaryNode.emplace_back(nodeI.first);
                BNArea.emplace_back(nodeI.second);
            }
            codimBNStartInd[1] = boundaryNode.size();

            for (const auto& vI : particle) {
                boundaryNode.emplace_back(vI);
            }
        }
        std::cout << "surface primitives found" << std::endl;
    }
    //TODO: dHat relative to bbox, adapt kappa
    std::vector<VECTOR<int, dim + 1>> constraintSet, constraintSet_prev;
    std::vector<T> dist2_prev;
    std::vector<VECTOR<int, 2>> constraintSetPTEE;
    std::vector<VECTOR<T, 2>> stencilInfo;
    // friction:
    std::vector<VECTOR<int, dim + 1>> fricConstraintSet;
    std::vector<Eigen::Matrix<T, dim - 1, 1>> closestPoint;
    std::vector<Eigen::Matrix<T, dim, dim - 1>> tanBasis;
    std::vector<T> normalForce;

    T DBCAlpha = 1, DBCStiff = 0, DBCPenaltyXn = 0;
    std::vector<bool> DBCb(X.size, false); // this mask does not change with whether augmented Lagrangian is turned on
    std::vector<bool> DBCb_fixed(X.size, false); // this masks nodes that are fixed (DBC with 0 velocity)
    std::vector<T> DBCDisp(X.size * dim, T(0));
    DBC_Info_Init(X, Xn, nodeAttr, Elem, elemAttr, elasticityAttr, DBC, DBCb, DBCb_fixed, DBCDisp,
        staticSolve, h, s, sHat, kappa_s, DBCStiff, DBCAlpha, DBCPenaltyXn);

    // compute energy record
    if (withCollision) {
        Compute_Constraint_Set<T, dim, false, elasticIPC>(X, nodeAttr, boundaryNode, boundaryEdge, boundaryTri, 
            particle, rod, std::map<int, std::set<int>>(), BNArea, BEArea, BTArea, codimBNStartInd, DBCb, dHat2, thickness, false, constraintSet, constraintSetPTEE, stencilInfo);
        if (mu > 0 || (muComp.size() && muComp.size() == compNodeRange.size() * compNodeRange.size())) {
            Compute_Friction_Basis<T, dim, elasticIPC>(X, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce, dHat2, kappa, thickness);
            if (muComp.size() && muComp.size() == compNodeRange.size() * compNodeRange.size()) {
                Compute_Friction_Coef<T, dim>(fricConstraintSet, compNodeRange, muComp, normalForce, mu); 
                // mu will be set to 1, normalForce will be multipled with different mu's in muComp
            }
        }
    }
    T Eprev;
    if (!Compute_SLP_Energy<T, dim, elasticIPC>(X, Xn, X_prim, M, nodeAttr, Elem, elemAttr, elasticityAttr, 
        DBCb, DBCb_fixed, DBC, DBCStiff, staticSolve, h, s, sHat, kappa_s, 
        withCollision, constraintSet, stencilInfo, dHat2, kappa, thickness,
        fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2, mu, Eprev)) 
    {
        printf("beginning of time step X violate strain limit!\n");
        exit(-1);
    }

    // Newton loop
    T L2Norm = 0;
    std::deque<T> resRecord;
    bool useGD = false;
    do
    {
        // compute gradient
        Compute_SLP_Gradient<T, dim, elasticIPC>(X, Xn, X_prim, M, nodeAttr, Elem, elemAttr, elasticityAttr, 
            DBCb, DBCb_fixed, DBC, DBCStiff, staticSolve, h, s, sHat, kappa_s,
            withCollision, constraintSet, stencilInfo, dHat2, kappa, thickness,
            fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2, mu, rhs);

        // compute Hessian
        if (!useGD) {
            Compute_SLP_Hessian<T, dim, elasticIPC>(X, Xn, X_prim, M, nodeAttr, Elem, elemAttr, elasticityAttr, 
                DBCb, DBCb_fixed, DBC, DBCStiff, staticSolve, h, s, sHat, kappa_s, 
                withCollision, constraintSet, stencilInfo, dHat2, kappa, thickness,
                fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2, mu, true, sysMtr);
        }

        // compute search direction
        {
            TIMER_FLAG("linearSolve");

            if (useGD) {
                printf("use gradient descent\n");
                std::memcpy(sol.data(), rhs.data(), sizeof(T) * rhs.size());
            }
            else {
#ifdef AMGCL_LINEAR_SOLVER
                // AMGCL
                std::memset(sol.data(), 0, sizeof(T) * sol.size());
                Solve(sysMtr, rhs, sol, 1.0e-5, 1000, Default_FEM_Params<dim>(), true);
#else
                // direct factorization
                if(!Solve_Direct(sysMtr, rhs, sol)) {
                    useGD = true;
                    printf("use gradient descent\n");
                    std::memcpy(sol.data(), rhs.data(), sizeof(T) * rhs.size());
                }
#endif
            }
        }

        // line search
        MESH_NODE<T, dim> Xprev;
        Append_Attribute(X, Xprev);
        bool valid = true;
        T alpha = 1, feasibleAlpha = 0, E;
        T minDist2;
        if (withCollision) {
            Compute_Intersection_Free_StepSize<T, dim, false, elasticIPC>(X, boundaryNode, boundaryEdge, boundaryTri, 
                particle, rod, std::map<int, std::set<int>>(), codimBNStartInd, DBCb, sol, thickness, alpha); // CCD
            printf("intersection free step size = %le\n", alpha);
        }
        do {
            X.Join(Xprev).Par_Each([&](int id, auto data) {
                auto &[x, xprev] = data;
                x[0] = xprev[0] + alpha * sol[id * dim];
                x[1] = xprev[1] + alpha * sol[id * dim + 1];
                if constexpr (dim == 3) {
                    x[2] = xprev[2] + alpha * sol[id * dim + 2];
                }
            });

            valid = Compute_SLP_Energy<T, dim, elasticIPC>(X, Xn, X_prim, M, nodeAttr, Elem, elemAttr, elasticityAttr, 
                DBCb, DBCb_fixed, DBC, DBCStiff, staticSolve, h, s, sHat, kappa_s, 
                false, constraintSet, stencilInfo, dHat2, kappa, thickness,
                fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2, mu, E);
            if (valid) {
                if (!feasibleAlpha) {
                    feasibleAlpha = alpha;
                }
                if (withCollision) {
                    Compute_Constraint_Set<T, dim, false, elasticIPC>(X, nodeAttr, boundaryNode, boundaryEdge, boundaryTri, 
                        particle, rod, std::map<int, std::set<int>>(), BNArea, BEArea, BTArea, codimBNStartInd, DBCb, dHat2, thickness, false, constraintSet, constraintSetPTEE, stencilInfo);
                    if (!constraintSet.empty()) {
                        std::vector<T> dist2;
                        Compute_Min_Dist2<T, dim, elasticIPC>(X, constraintSet, thickness, dist2, minDist2);
                        if (minDist2 <= 0) {
                            std::cout << "safe guard backtrack!" << std::endl;
                            alpha /= 2;
                            valid = false;
                            feasibleAlpha = 0;
                            continue;
                        }
                    }

                    Compute_SLP_Energy<T, dim, elasticIPC>(X, Xn, X_prim, M, nodeAttr, Elem, elemAttr, elasticityAttr, 
                        DBCb, DBCb_fixed, DBC, DBCStiff, staticSolve, h, s, sHat, kappa_s, 
                        withCollision, constraintSet, stencilInfo, dHat2, kappa, thickness,
                        fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2, mu, E);
                }
            }
            
            alpha /= 2;
            printf("E %le, Eprev %le, alpha %le, valid %d\n", E, Eprev, alpha * 2, valid ? 1 : 0);
        } while (E > Eprev || !valid);
        Eprev = E;

        if (constraintSet.size()) {
            printf("minDist2 = %le\n", minDist2);
        }

        // stopping criteria
        T maxRes = 0.0, avgResMag = 0.0;
        L2Norm = 0.0;
        for (int i = 0; i < sol.size() / dim; ++i) {
            T curMag = 0;
            for (int j = 0; j < dim; ++j) {
                curMag += sol[i * dim + j] * sol[i * dim + j];
                if (maxRes < std::abs(sol[i * dim + j])) {
                    maxRes = std::abs(sol[i * dim + j]);
                }
                L2Norm += sol[i * dim + j] * sol[i * dim + j];
            }
            avgResMag += std::sqrt(curMag);
        }
        avgResMag /= sol.size() / dim - DBC.size;
        avgResMag /= (staticSolve ? 1 : h);
        maxRes /= (staticSolve ? 1 : h);
        L2Norm = std::sqrt(L2Norm / (sol.size() / dim - DBC.size));
        L2Norm /= (staticSolve ? 1 : h);
        printf("[strain-limiting projection] totalPNIter%d: Newton res = %le, tol = %le\n", PNIter++, L2Norm, NewtonTol);

        FILE *out = fopen((outputFolder + "/residual.txt").c_str(), "a+");
        fprintf(out, "%d %le %le %le %le\n", PNIter, avgResMag, maxRes, Eprev, L2Norm);
        fclose(out);

        resRecord.push_back(L2Norm);
        if (resRecord.size() > 3) {
            resRecord.pop_front();
        }
        L2Norm = *std::max_element(resRecord.begin(), resRecord.end());

        // switch to GD when necessary
        if (useGD) {
            L2Norm = NewtonTol * 10; // ensures not exit Newton loop
        }
        if (alpha * 2 < 1e-6 && feasibleAlpha > 1e-6) {
            if (!useGD) {
                useGD = true;
                Eigen::VectorXd pe(sol.size()), mge(rhs.size());
                std::memcpy(pe.data(), sol.data(), sizeof(T) * sol.size());
                std::memcpy(mge.data(), rhs.data(), sizeof(T) * rhs.size());
                printf("-gdotp = %le, -gpcos = %le\n", mge.dot(pe), 
                    mge.dot(pe) / std::sqrt(mge.squaredNorm() * pe.squaredNorm()));
                printf("linear solve relErr = %le\n", 
                    std::sqrt((sysMtr.Get_Matrix() * pe - mge).squaredNorm() / mge.squaredNorm()));
            }
            else {
                printf("GD tiny step size!\n");
            }
        }
        else {
            useGD = false;
        }

        if (DBCStiff) {
            DBC_Dual_Update<T, dim, elasticIPC>(X, Xn, X_prim, M, nodeAttr, Elem, elemAttr, elasticityAttr, 
                DBCb, DBCb_fixed, DBC, DBCStiff, staticSolve, h, s, sHat, kappa_s,
                withCollision, constraintSet, stencilInfo, dHat2, kappa, thickness,
                fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2, mu,
                DBCPenaltyXn, Eprev, L2Norm, NewtonTol);
        }
    } while ((resRecord.size() < 3) || (L2Norm > NewtonTol));

    // update velocity
    if (!staticSolve) {
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            v = (x - std::get<0>(Xn.Get_Unchecked(id))) / h;
        });
    }

    // output information
    out = fopen((outputFolder + "/counter.txt").c_str(), "a+");
    fprintf(out, " %d", PNIter);
    if (withCollision) {
        printf("contact #: %lu\n", constraintSet.size());
        fprintf(out, " %lu", constraintSet.size());
    }
    fprintf(out, "\n");
    fclose(out);

    T maxs, avgs, minc, avgc;
    if (Elem.size) {
        Compute_Max_And_Avg_Stretch(Elem, staticSolve ? 1.0 : h, fiberStiffMult, DBCb, X, nodeAttr, elemAttr, elasticityAttr, maxs, avgs, minc, avgc);
        printf("maxs = %le, avgs = %le\n", maxs, avgs);
        out = fopen((outputFolder + "/stretch.txt").c_str(), "a+");
        fprintf(out, "%le %le %le %le\n", maxs, avgs, minc, avgc);
        fclose(out);
    }
    if (rod.size()) {
        Compute_Max_And_Avg_Stretch_Rod(X, rod, rodInfo, maxs, avgs);
        printf("rod: maxs = %le, avgs = %le\n", maxs, avgs);
        out = fopen((outputFolder + "/stretch_rod.txt").c_str(), "a+");
        fprintf(out, "%le %le\n", maxs, avgs);
        fclose(out);
    }

    return PNIter;
}

}