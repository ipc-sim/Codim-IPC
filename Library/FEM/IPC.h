#pragma once

#include <Grid/SPATIAL_HASH.h>
#include <Utils/MESHIO.h>
#include <Math/VECTOR.h>
#include <Math/BARRIER.h>
#include <Math/UTILS.h>
#include <Math/Distance/DISTANCE_TYPE.h>
#include <Math/Distance/POINT_POINT.h>
#include <Math/Distance/POINT_EDGE.h>
#include <Math/Distance/POINT_TRIANGLE.h>
#include <Math/Distance/EDGE_EDGE.h>
#include <Math/Distance/EDGE_EDGE_MOLLIFIER.h>
#include <Math/Distance/CCD.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim, bool shell = false, bool elasticIPC = false>
void Compute_Constraint_Set(MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    const std::vector<int>& boundaryNode, // tet surf and tri nodes, seg nodes, rod nodes, particle nodes
    const std::vector<VECTOR<int, 2>>& boundaryEdge, // tet surf and tri edges, seg, rod
    const std::vector<VECTOR<int, 3>>& boundaryTri, // tet surf tris, tris
    const std::vector<int>& particle,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::map<int, std::set<int>>& NNExclusion,
    const std::vector<T>& BNArea,
    const std::vector<T>& BEArea,
    const std::vector<T>& BTArea,
    const VECTOR<int, 2>& codimBNStartInd,
    const std::vector<bool>& DBCb,
    T dHat2, T thickness, bool getPTEE,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    std::vector<VECTOR<int, 2>>& cs_PTEE,
    std::vector<VECTOR<T, 2>>& stencilInfo) // weight, dHat2
{
    TIMER_FLAG("Compute_Constraint_Set");

#define USE_SH_CCS
#ifdef USE_SH_CCS
    SPATIAL_HASH<T, dim> sh;
    {
        TIMER_FLAG("Compute_Constraint_Set_Build_Hash");
        sh.Build(X, boundaryNode, boundaryEdge, boundaryTri, 1.0);
    }
#endif

    if constexpr (elasticIPC) {
        thickness = 0;
    }

    T dHat = std::sqrt(dHat2) + thickness;
    dHat2 = dHat * dHat;
    if constexpr (dim == 2) {
        BASE_STORAGE<int> threads(boundaryNode.size());
        for (int i = 0; i < boundaryNode.size(); ++i) {
            threads.Append(i);
        }

        std::vector<std::vector<VECTOR<int, 2>>> PESetNI;
        if (getPTEE) {
            PESetNI.resize(boundaryNode.size());
        }

        std::vector<std::vector<VECTOR<int, 3>>> constraintSetNI(boundaryNode.size());
        threads.Par_Each([&](int bNI, auto data){
            int nI = boundaryNode[bNI];
            const VECTOR<T, 2>& Xp = std::get<0>(X.Get_Unchecked(nI));
            Eigen::Matrix<T, 2, 1> p0(Xp.data);
#ifdef USE_SH_CCS
            std::unordered_set<int> eInds; //NOTE: different constraint order will result in numerically different results
            sh.Query_Point_For_Edges(p0, dHat, eInds);
            for (const auto& eInd : eInds) {
                const auto& eI = boundaryEdge[eInd];
#else
            for (const auto& eI : boundaryEdge) {
#endif
                if (nI == eI[0] || nI == eI[1] ||
                    (DBCb[nI] && DBCb[eI[0]] && DBCb[eI[1]])) 
                {
                    continue;
                }
                if constexpr (shell) {
                    if (((nI % 3 == 0) && (nI + 2 == eI[0] || nI + 2 == eI[1])) ||
                        ((nI % 3 == 2) && (nI - 2 == eI[0] || nI - 2 == eI[1]))) { continue; }
                }

                const VECTOR<T, 2>& Xe0 = std::get<0>(X.Get_Unchecked(eI[0]));
                const VECTOR<T, 2>& Xe1 = std::get<0>(X.Get_Unchecked(eI[1]));
                Eigen::Matrix<T, 2, 1> e0(Xe0.data), e1(Xe1.data);

                if (!Point_Edge_CD_Broadphase(p0, e0, e1, dHat)) {
                    continue;
                }

                T ratio, dist2;
                switch(Point_Edge_Distance_Type(p0, e0, e1, ratio)) {
                    case 0: {
                        Point_Point_Distance(p0, e0, dist2);
                        if (dist2 < dHat2) {
                            constraintSetNI[bNI].emplace_back(nI, eI[0], -1);
                        }
                        break;
                    }

                    case 1: {
                        Point_Point_Distance(p0, e1, dist2);
                        if (dist2 < dHat2) {
                            constraintSetNI[bNI].emplace_back(nI, eI[1], -1);
                        }
                        break;
                    }

                    case 2: {
                        Point_Edge_Distance(p0, e0, e1, dist2);
                        if (dist2 < dHat2) {
                            constraintSetNI[bNI].emplace_back(nI, eI[0], eI[1]);
                        }
                        break;
                    }
                }

                if (getPTEE && dist2 < dHat2) {
                    PESetNI[bNI].emplace_back(nI, eInd);
                }
            }
        });

        //TODO: handle PP duplication?
        constraintSet.resize(0);
        for (const auto& csI : constraintSetNI) {
            constraintSet.insert(constraintSet.end(), csI.begin(), csI.end());
        }

        if (getPTEE) {
            cs_PTEE.resize(0);
            for (const auto& setI : PESetNI) {
                cs_PTEE.insert(cs_PTEE.end(), setI.begin(), setI.end());
            }
        }
    }
    else {
        // point-triangle
        std::vector<std::vector<VECTOR<int, 4>>> constraintSetPT(boundaryNode.size());
        std::vector<std::vector<VECTOR<T, 2>>> constraintInfoPT(boundaryNode.size());
        std::vector<std::vector<int>> cs_PT;
        { TIMER_FLAG("Compute_Constraint_Set_PT");
        if (getPTEE) {
            cs_PT.resize(boundaryNode.size());
        }

        BASE_STORAGE<int> threadsPT(boundaryNode.size());
        for (int i = 0; i < boundaryNode.size(); ++i) {
            threadsPT.Append(i);
        }

        threadsPT.Par_Each([&](int svI, auto data) {
            int vI = boundaryNode[svI];
            const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(vI));
            Eigen::Matrix<T, 3, 1> p(Xp.data);
#ifdef USE_SH_CCS
            std::unordered_set<int> triInds; //NOTE: different constraint order will result in numerically different results
            sh.Query_Point_For_Triangles(p, dHat, triInds);
            for (const auto& sfI : triInds)
#else
            for (int sfI = 0; sfI < boundaryTri.size(); ++sfI)
#endif
            {
                const VECTOR<int, 3>& sfVInd = boundaryTri[sfI];
                if (!(vI == sfVInd[0] || vI == sfVInd[1] || vI == sfVInd[2]) &&
                    !(DBCb[vI] && DBCb[sfVInd[0]] && DBCb[sfVInd[1]] && DBCb[sfVInd[2]])) 
                {
                    auto NNEFinder = NNExclusion.find(vI);
                    if (NNEFinder != NNExclusion.end() &&
                        (NNEFinder->second.find(sfVInd[0]) != NNEFinder->second.end() ||
                        NNEFinder->second.find(sfVInd[1]) != NNEFinder->second.end() ||
                        NNEFinder->second.find(sfVInd[2]) != NNEFinder->second.end()))
                    {
                        continue;
                    }
                    if constexpr (shell) {
                        int vpair = (vI % 2 == 0) ? (vI + 1) : vI - 1;
                        if (vpair == sfVInd[0] || vpair == sfVInd[1] || vpair == sfVInd[2]) {
                            continue;
                        }
                    }
                    const VECTOR<T, 3>& Xt0 = std::get<0>(X.Get_Unchecked(sfVInd[0]));
                    const VECTOR<T, 3>& Xt1 = std::get<0>(X.Get_Unchecked(sfVInd[1]));
                    const VECTOR<T, 3>& Xt2 = std::get<0>(X.Get_Unchecked(sfVInd[2]));
                    Eigen::Matrix<T, 3, 1> t0(Xt0.data), t1(Xt1.data), t2(Xt2.data);

                    if (!Point_Triangle_CD_Broadphase(p, t0, t1, t2, dHat)) {
                        continue;
                    }

                    T d;
                    switch (Point_Triangle_Distance_Type(p, t0, t1, t2)) {
                    case 0: {
                        Point_Point_Distance(p, t0, d);
                        if (d < dHat2) {
                            constraintSetPT[svI].emplace_back(-vI - 1, sfVInd[0], -1, -1);
                        }
                        break;
                    }

                    case 1: {
                        Point_Point_Distance(p, t1, d);
                        if (d < dHat2) {
                            constraintSetPT[svI].emplace_back(-vI - 1, sfVInd[1], -1, -1);
                        }
                        break;
                    }

                    case 2: {
                        Point_Point_Distance(p, t2, d);
                        if (d < dHat2) {
                            constraintSetPT[svI].emplace_back(-vI - 1, sfVInd[2], -1, -1);
                        }
                        break;
                    }

                    case 3: {
                        Point_Edge_Distance(p, t0, t1, d);
                        if (d < dHat2) {
                            constraintSetPT[svI].emplace_back(-vI - 1, sfVInd[0], sfVInd[1], -1);
                        }
                        break;
                    }

                    case 4: {
                        Point_Edge_Distance(p, t1, t2, d);
                        if (d < dHat2) {
                            constraintSetPT[svI].emplace_back(-vI - 1, sfVInd[1], sfVInd[2], -1);
                        }
                        break;
                    }

                    case 5: {
                        Point_Edge_Distance(p, t2, t0, d);
                        if (d < dHat2) {
                            constraintSetPT[svI].emplace_back(-vI - 1, sfVInd[2], sfVInd[0], -1);
                        }
                        break;
                    }

                    case 6: {
                        Point_Triangle_Distance(p, t0, t1, t2, d);
                        if (d < dHat2) {
                            constraintSetPT[svI].emplace_back(-vI - 1, sfVInd[0], sfVInd[1], sfVInd[2]);
                        }
                        break;
                    }

                    default:
                        break;
                    }

                    if (getPTEE && d < dHat2) {
                        cs_PT[svI].emplace_back(sfI);
                    }
                    if (d < dHat2) {
                        constraintInfoPT[svI].emplace_back(BNArea[svI] * BTArea[sfI], dHat2);
                        if (svI < codimBNStartInd[0]) { // if it is surface nodes
                            constraintInfoPT[svI].back()[0] /= 2; // normalization: 1/2 (P_PT + P_EE)
                        }
                    }
                }
            }

            if (svI >= codimBNStartInd[0]) { // rod or particle points
#ifdef USE_SH_CCS
                std::unordered_set<int> edgeInds; //NOTE: different constraint order will result in numerically different results
                sh.Query_Point_For_Edges(p, dHat, edgeInds);
                for (const auto& eI : edgeInds)
#else
                for (int eI = 0; eI < boundaryEdge.size(); ++eI)
#endif
                {
                    if (eI >= boundaryEdge.size() - rod.size()) { // rod edge
                        const VECTOR<int, 2>& edgeVInd = boundaryEdge[eI];
                        if (vI != edgeVInd[0] && vI != edgeVInd[1] && !(DBCb[vI] && DBCb[edgeVInd[0]] && DBCb[edgeVInd[1]])) {
                            const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                            const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                            Eigen::Matrix<T, 3, 1> e0(Xe0.data), e1(Xe1.data);

                            if (!Point_Edge_CD_Broadphase(p, e0, e1, dHat)) {
                                continue;
                            }

                            T d, ratio;
                            switch (Point_Edge_Distance_Type(p, e0, e1, ratio)) {
                            case 0: {
                                Point_Point_Distance(p, e0, d);
                                if (d < dHat2) {
                                    constraintSetPT[svI].emplace_back(-vI - 1, edgeVInd[0], -1, -1);
                                }
                                break;
                            }

                            case 1: {
                                Point_Point_Distance(p, e1, d);
                                if (d < dHat2) {
                                    constraintSetPT[svI].emplace_back(-vI - 1, edgeVInd[1], -1, -1);
                                }
                                break;
                            }

                            case 2: {
                                Point_Edge_Distance(p, e0, e1, d);
                                if (d < dHat2) {
                                    constraintSetPT[svI].emplace_back(-vI - 1, edgeVInd[0], edgeVInd[1], -1);
                                }
                                break;
                            }

                            default:
                                break;
                            }

                            if (d < dHat2) {
                                constraintInfoPT[svI].emplace_back(1, dHat2); //TODO: integration view
                            }
                        }
                    }
                }

                if (svI >= codimBNStartInd[1]) { // particle point
#ifdef USE_SH_CCS
                    std::unordered_set<int> pointInds; //NOTE: different constraint order will result in numerically different results
                    sh.Query_Point_For_Points(p, dHat, pointInds);
                    for (const auto& svJ : pointInds)
#else
                    for (int svJ = 0; svJ < boundaryNode.size(); ++svJ)
#endif
                    {
                        int vJ = boundaryNode[svJ];
                        if (svJ > svI && !(DBCb[vI] && DBCb[vJ])) {
                            const VECTOR<T, 3>& Xq = std::get<0>(X.Get_Unchecked(vJ));
                            Eigen::Matrix<T, 3, 1> q(Xq.data);

                            // since Point_Point_Distance is cheap,
                            // we don't need Point_Point_CD_Broadphase
                            T d;
                            Point_Point_Distance(p, q, d);
                            if (d < dHat2) {
                                constraintSetPT[svI].emplace_back(-vI - 1, vJ, -1, -1);
                                constraintInfoPT[svI].emplace_back(1, dHat2); //TODO: integration view
                            }
                        }
                    }
                }
            }
        });
        } //TIMER_FLAG

        // edge-edge
        std::vector<std::vector<VECTOR<int, 4>>> constraintSetEE(boundaryEdge.size());
        std::vector<std::vector<VECTOR<T, 2>>> constraintInfoEE(boundaryEdge.size());
        std::vector<std::vector<int>> cs_EE;
        { TIMER_FLAG("Compute_Constraint_Set_EE");
        if (getPTEE) {
            cs_EE.resize(boundaryEdge.size());
        }

        BASE_STORAGE<int> threadsEE(boundaryEdge.size());
        for (int i = 0; i < boundaryEdge.size(); ++i) {
            threadsEE.Append(i);
        }

        threadsEE.Par_Each([&](int eI, auto data) {
            const VECTOR<int, 2>& meshEI = boundaryEdge[eI];
            const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(meshEI[0]));
            const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(meshEI[1]));
            Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data);
#ifdef USE_SH_CCS
            std::vector<int> edgeInds; //NOTE: different constraint order will result in numerically different results
            sh.Query_Edge_For_Edges(ea0, ea1, dHat, edgeInds, eI);
            for (const auto& eJ : edgeInds) {
#else
            for (int eJ = eI + 1; eJ < boundaryEdge.size(); ++eJ) {
#endif
                const VECTOR<int, 2>& meshEJ = boundaryEdge[eJ];
                if (!(meshEI[0] == meshEJ[0] || meshEI[0] == meshEJ[1] || meshEI[1] == meshEJ[0] || meshEI[1] == meshEJ[1] || eI > eJ) &&
                    !(DBCb[meshEI[0]] && DBCb[meshEI[1]] && DBCb[meshEJ[0]] && DBCb[meshEJ[1]])) 
                {
                    auto NNEFinder0 = NNExclusion.find(meshEI[0]);
                    auto NNEFinder1 = NNExclusion.find(meshEI[1]);
                    if ((NNEFinder0 != NNExclusion.end() &&
                        (NNEFinder0->second.find(meshEJ[0]) != NNEFinder0->second.end() ||
                        NNEFinder0->second.find(meshEJ[1]) != NNEFinder0->second.end())) ||
                        (NNEFinder1 != NNExclusion.end() && 
                        (NNEFinder1->second.find(meshEJ[0]) != NNEFinder1->second.end() ||
                        NNEFinder1->second.find(meshEJ[1]) != NNEFinder1->second.end())))
                    {
                        continue;
                    }
                    if constexpr (shell) {
                        if (meshEI[0] % 2 == 1) {
                            if (meshEI[0] - 1 == meshEJ[0] || meshEI[0] - 1 == meshEJ[1]) { continue; }
                        }
                        else if (meshEI[0] % 2 == 0) {
                            if (meshEI[0] + 1 == meshEJ[0] || meshEI[0] + 1 == meshEJ[1]) { continue; }
                        }

                        if (meshEI[1] % 2 == 1) {
                            if (meshEI[1] - 1 == meshEJ[0] || meshEI[1] - 1 == meshEJ[1]) { continue; }
                        }
                        else if (meshEI[1] % 2 == 0) {
                            if (meshEI[1] + 1 == meshEJ[0] || meshEI[1] + 1 == meshEJ[1]) { continue; }
                        }
                    }
                    const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(meshEJ[0]));
                    const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(meshEJ[1]));
                    Eigen::Matrix<T, 3, 1> eb0(Xeb0.data), eb1(Xeb1.data);

                    if (!Edge_Edge_CD_Broadphase(ea0, ea1, eb0, eb1, dHat)) {
                        continue;
                    }

                    const VECTOR<T, 3>& Xea0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(meshEI[0]));
                    const VECTOR<T, 3>& Xea1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(meshEI[1]));
                    const VECTOR<T, 3>& Xeb0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(meshEJ[0]));
                    const VECTOR<T, 3>& Xeb1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(meshEJ[1]));
                    Eigen::Matrix<T, 3, 1> ea0_rest(Xea0_rest.data), ea1_rest(Xea1_rest.data), 
                        eb0_rest(Xeb0_rest.data), eb1_rest(Xeb1_rest.data);

                    T EECrossSqNorm, eps_x;
                    Edge_Edge_Cross_Norm2(ea0, ea1, eb0, eb1, EECrossSqNorm);
                    Edge_Edge_Mollifier_Threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest, eps_x);

                    bool mollify = (EECrossSqNorm < eps_x);
                    T d;
                    switch (Edge_Edge_Distance_Type(ea0, ea1, eb0, eb1)) {
                    case 0: {
                        Point_Point_Distance(ea0, eb0, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEI[0], meshEJ[0], -meshEI[1] - 1, -meshEJ[1] - 1);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(-meshEI[0] - 1, meshEJ[0], -1, -1);
                            }
                        }
                        break;
                    }

                    case 1: {
                        Point_Point_Distance(ea0, eb1, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEI[0], meshEJ[1], -meshEI[1] - 1, -meshEJ[0] - 1);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(-meshEI[0] - 1, meshEJ[1], -1, -1);
                            }
                        }
                        break;
                    }

                    case 2: {
                        Point_Edge_Distance(ea0, eb0, eb1, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEI[0], meshEJ[0], meshEJ[1], -meshEI[1] - 1);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(-meshEI[0] - 1, meshEJ[0], meshEJ[1], -1);
                            }
                        }
                        break;
                    }

                    case 3: {
                        Point_Point_Distance(ea1, eb0, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEI[1], meshEJ[0], -meshEI[0] - 1, -meshEJ[1] - 1);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(-meshEI[1] - 1, meshEJ[0], -1, -1);
                            }
                        }
                        break;
                    }

                    case 4: {
                        Point_Point_Distance(ea1, eb1, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEI[1], meshEJ[1], -meshEI[0] - 1, -meshEJ[0] - 1);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(-meshEI[1] - 1, meshEJ[1], -1, -1);
                            }
                        }
                        break;
                    }

                    case 5: {
                        Point_Edge_Distance(ea1, eb0, eb1, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEI[1], meshEJ[0], meshEJ[1], -meshEI[0] - 1);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(-meshEI[1] - 1, meshEJ[0], meshEJ[1], -1);
                            }
                        }
                        break;
                    }

                    case 6: {
                        Point_Edge_Distance(eb0, ea0, ea1, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEJ[0], meshEI[0], meshEI[1], -meshEJ[1] - 1);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(-meshEJ[0] - 1, meshEI[0], meshEI[1], -1);
                            }
                        }
                        break;
                    }

                    case 7: {
                        Point_Edge_Distance(eb1, ea0, ea1, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEJ[1], meshEI[0], meshEI[1], -meshEJ[0] - 1);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(-meshEJ[1] - 1, meshEI[0], meshEI[1], -1);
                            }
                        }
                        break;
                    }

                    case 8: {
                        Edge_Edge_Distance(ea0, ea1, eb0, eb1, d);
                        if (d < dHat2) {
                            if (mollify) {
                                constraintSetEE[eI].emplace_back(meshEI[0], meshEI[1], -meshEJ[0] - 1, meshEJ[1]);
                            }
                            else {
                                constraintSetEE[eI].emplace_back(meshEI[0], meshEI[1], meshEJ[0], meshEJ[1]);
                            }
                        }
                        break;
                    }

                    default:
                        break;
                    }

                    if (getPTEE && d < dHat2) {
                        cs_EE[eI].emplace_back(eJ);
                    }
                    if (d < dHat2) {
                        constraintInfoEE[eI].emplace_back(BEArea[eI] * BEArea[eJ], dHat2);
                        if ((eI >= boundaryEdge.size() - rod.size()) && (eJ >= boundaryEdge.size() - rod.size())) {
                            constraintInfoEE[eI].back()[0] *= 2; // no PE and no ij and ji EE at the same time here so *2 to normalize 
                        }
                    }
                }
            }
        });
        } //TIMER_FLAG

        
        { TIMER_FLAG("Compute_Constraint_Set_Merge");
        if (getPTEE) {
            cs_PTEE.resize(0);
            cs_PTEE.reserve(cs_PT.size() + cs_EE.size());
            for (int svI = 0; svI < cs_PT.size(); ++svI) {
                for (const auto& sfI : cs_PT[svI]) {
                    cs_PTEE.emplace_back(-svI - 1, sfI);
                }
            }
            for (int eI = 0; eI < cs_EE.size(); ++eI) {
                for (const auto& eJ : cs_EE[eI]) {
                    cs_PTEE.emplace_back(eI, eJ);
                }
            }
        }

        constraintSet.resize(0);
        constraintSet.reserve(constraintSetPT.size() + constraintSetEE.size());
        stencilInfo.resize(0);
        stencilInfo.reserve(constraintSetPT.size() + constraintSetEE.size());
        // if no duplication handling:
        // for (const auto& csI : constraintSetPT) {
        //     constraintSet.insert(constraintSet.end(), csI.begin(), csI.end());
        // }
        // for (const auto& csI : constraintSetEE) {
        //     constraintSet.insert(constraintSet.end(), csI.begin(), csI.end());
        // }
        // handle regular PP and PE duplication
        std::map<VECTOR<int, 4>, int> constraintCounter;
        std::map<VECTOR<int, 4>, VECTOR<T, 2>> constraintAreaCounter;
        int csIInd = 0;
        for (const auto& csI : constraintSetPT) {
            int cIInd = 0;
            for (const auto& cI : csI) {
                if (cI[3] < 0) {
                    // PP or PE
                    ++constraintCounter[cI];
                    auto finder = constraintAreaCounter.find(cI);
                    if (finder == constraintAreaCounter.end()) {
                        constraintAreaCounter[cI] = constraintInfoPT[csIInd][cIInd];
                    }
                    else {
                        finder->second[0] += constraintInfoPT[csIInd][cIInd][0];
                    }
                }
                else {
                    constraintSet.emplace_back(cI);
                    stencilInfo.emplace_back(constraintInfoPT[csIInd][cIInd]);
                }
                ++cIInd;
            }
            ++csIInd;
        }
        csIInd = 0;
        for (const auto& csI : constraintSetEE) {
            int cIInd = 0;
            for (const auto& cI : csI) {
                if (cI[0] < 0) {
                    // regular PP or PE
                    ++constraintCounter[cI];
                    auto finder = constraintAreaCounter.find(cI);
                    if (finder == constraintAreaCounter.end()) {
                        constraintAreaCounter[cI] = constraintInfoEE[csIInd][cIInd];
                    }
                    else {
                        finder->second[0] += constraintInfoEE[csIInd][cIInd][0];
                    }
                }
                else {
                    // regular EE or mollified EE, PE and PP
                    constraintSet.emplace_back(cI);
                    stencilInfo.emplace_back(constraintInfoEE[csIInd][cIInd]);
                }
                ++cIInd;
            }
            ++csIInd;
        }

        constraintSet.reserve(constraintSet.size() + constraintCounter.size());
        stencilInfo.reserve(constraintSet.size() + constraintCounter.size());
        for (const auto& ccI : constraintCounter) {
            constraintSet.emplace_back(VECTOR<int, 4>(ccI.first[0], ccI.first[1], ccI.first[2], -ccI.second));
            stencilInfo.emplace_back(constraintAreaCounter[ccI.first][0] / ccI.second, constraintAreaCounter[ccI.first][1]);
        }

        if constexpr (!elasticIPC) {
            for (auto& i : stencilInfo) {
                i[0] = 1; // no area weighting
            }
        }
        } //TIMER_FLAG

        // rod point - rod edge (to compensate parallel edge-edge mollification)
        // particle point - rod edge
        // particle point - particle point
//         if (rod.size()) {
                //TODO: TIMER_FLAG
//             std::vector<int> nodes;
//             for (const auto& segI : rod) {
//                 nodes.emplace_back(segI[0]);
//                 nodes.emplace_back(segI[1]);
//             }
//             sh.Build(X, nodes, rod, std::vector<VECTOR<int, 3>>(), 1.0);

//             // rod (and particle) point - rod edge
//             nodes.insert(nodes.end(), particle.begin(), particle.end());
//             //TODO: query point for edge
//             BASE_STORAGE<int> threadsPE(nodes.size());
//             for (int i = 0; i < nodes.size(); ++i) {
//                 threadsPE.Append(i);
//             }
//             threadsPE.Par_Each([&](int nI, auto data) {
//                 int vI = nodes[nI];
//                 const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(vI));
//                 Eigen::Matrix<T, 3, 1> p(Xp.data);
// #ifdef USE_SH_CCS
//                 std::unordered_set<int> edgeInds; //NOTE: different constraint order will result in numerically different results
//                 sh.Query_Point_For_Edges(p, dHat, edgeInds);
//                 for (const auto& rodI : edgeInds)
// #else
//                 for (int rodI = 0; rodI < rod.size(); ++rodI)
// #endif
//                 {
//                     const VECTOR<int, 2>& rodVInd = rod[rodI];
//                     if (!(vI == rodVInd[0] || vI == rodVInd[1]) &&
//                         !(DBCb[vI] && DBCb[rodVInd[0]] && DBCb[rodVInd[1]])) 
//                     {
//                         const VECTOR<T, 3>& Xr0 = std::get<0>(X.Get_Unchecked(rodVInd[0]));
//                         const VECTOR<T, 3>& Xr1 = std::get<0>(X.Get_Unchecked(rodVInd[1]));
//                         Eigen::Matrix<T, 3, 1> r0(Xr0.data), r1(Xr1.data);

//                         if (!Point_Edge_CD_Broadphase(p, r0, r1, dHat)) {
//                             continue;
//                         }

//                         T d;
//                         switch (Point_Edge_Distance_Type(p, r0, r1)) {
//                         case 0: {
//                             Point_Point_Distance(p, r0, d);
//                             if (d < dHat2) {
//                                 constraintSetPE[vI].emplace_back(-vI - 1, rodVInd[0], -1, -1);
//                             }
//                             break;
//                         }

//                         case 1: {
//                             Point_Point_Distance(p, r1, d);
//                             if (d < dHat2) {
//                                 constraintSetPE[vI].emplace_back(-vI - 1, rodVInd[1], -1, -1);
//                             }
//                             break;
//                         }

//                         case 2: {
//                             Point_Edge_Distance(p, r0, r1, d);
//                             if (d < dHat2) {
//                                 constraintSetPE[vI].emplace_back(-vI - 1, rodVInd[0], rodVInd[1], -1);
//                             }
//                             break;
//                         }

//                         default:
//                             break;
//                         }
//                     }
//                 }
//             });
//         }
    }
}

template <class T, int dim, bool elasticIPC = false>
void Compute_Barrier(MESH_NODE<T, dim>& X, 
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo, // weight, dHat2
    T dHat2, T kappa[], T thickness,
    T& E)
{
    TIMER_FLAG("Compute_Barrier");

    if constexpr (elasticIPC) {
        thickness = 0;
    }

    const T thickness2 = thickness * thickness;
    dHat2 += 2 * std::sqrt(dHat2) * thickness;

    std::vector<T> barrier(constraintSet.size());
    if constexpr (dim == 2) {
        //TODO: parallelize
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            const VECTOR<int, 3>& cIVInd = constraintSet[cI];
            if (cIVInd[2] < 0) {
                // PP
                const VECTOR<T, 2>& Xp0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                const VECTOR<T, 2>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                Eigen::Matrix<T, 2, 1> p0(Xp0.data), p1(Xp1.data);
                
                T dist2;
                Point_Point_Distance(p0, p1, dist2);
                dist2 -= thickness2;
                if (dist2 <= 0) {
                    printf("%le distance detected during barrier evaluation!\n", dist2);
                    exit(-1);
                }
                
                Barrier<elasticIPC>(dist2, dHat2, kappa, barrier[cI]);
            }
            else {
                // PE
                const VECTOR<T, 2>& Xp = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                const VECTOR<T, 2>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                const VECTOR<T, 2>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                Eigen::Matrix<T, 2, 1> p(Xp.data), e0(Xe0.data), e1(Xe1.data);
                
                T dist2;
                Point_Edge_Distance(p, e0, e1, dist2);
                dist2 -= thickness2;
                if (dist2 <= 0) {
                    printf("%le distance detected during barrier evaluation!\n", dist2);
                    exit(-1);
                }
                
                Barrier<elasticIPC>(dist2, dHat2, kappa, barrier[cI]);
            }
        }
    }
    else {
        //TODO: parallelize
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            const VECTOR<int, 4>& cIVInd = constraintSet[cI];
            assert(cIVInd[1] >= 0);
            if (cIVInd[0] >= 0) {
                // EE
                if (cIVInd[3] >= 0 && cIVInd[2] >= 0) {
                    // ++++ EE, no mollification
                    const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                    const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                    Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);
                    
                    T dist2;
                    Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2);
                    dist2 -= thickness2;
                    if (dist2 <= 0) {
                        printf("%le distance detected during barrier evaluation!\n", dist2);
                        exit(-1);
                    }
                    
                    Barrier<elasticIPC>(dist2, dHat2, kappa, barrier[cI]);
                }
                else {
                    // EE, PE, or PP with mollification
                    std::array<int, 4> edgeVInd;
                    T dist2;
                    Eigen::Matrix<T, 3, 1> ea0, ea1, eb0, eb1;
                    if (cIVInd[3] >= 0) {
                        // ++-+ EE with mollification
                        edgeVInd = {cIVInd[0], cIVInd[1], -cIVInd[2] - 1, cIVInd[3]};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        ea0 = std::move(Eigen::Matrix<T, 3, 1>(Xea0.data));
                        ea1 = std::move(Eigen::Matrix<T, 3, 1>(Xea1.data));
                        eb0 = std::move(Eigen::Matrix<T, 3, 1>(Xeb0.data));
                        eb1 = std::move(Eigen::Matrix<T, 3, 1>(Xeb1.data));
                        
                        Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2);
                    }
                    else if (cIVInd[2] >= 0) {
                        // +++- PE with mollification, multiplicity 1
                        edgeVInd = {cIVInd[0], -cIVInd[3] - 1, cIVInd[1], cIVInd[2]};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        ea0 = std::move(Eigen::Matrix<T, 3, 1>(Xea0.data));
                        ea1 = std::move(Eigen::Matrix<T, 3, 1>(Xea1.data));
                        eb0 = std::move(Eigen::Matrix<T, 3, 1>(Xeb0.data));
                        eb1 = std::move(Eigen::Matrix<T, 3, 1>(Xeb1.data));

                        Point_Edge_Distance(ea0, eb0, eb1, dist2);
                    }
                    else {
                        // ++-- PP with mollification, multiplicity 1
                        edgeVInd = {cIVInd[0], -cIVInd[2] - 1, cIVInd[1], -cIVInd[3] - 1};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        ea0 = std::move(Eigen::Matrix<T, 3, 1>(Xea0.data));
                        ea1 = std::move(Eigen::Matrix<T, 3, 1>(Xea1.data));
                        eb0 = std::move(Eigen::Matrix<T, 3, 1>(Xeb0.data));
                        eb1 = std::move(Eigen::Matrix<T, 3, 1>(Xeb1.data));

                        Point_Point_Distance(ea0, eb0, dist2);
                    }
                    dist2 -= thickness2;
                    
                    if (dist2 <= 0) {
                        printf("%le distance detected during barrier evaluation!\n", dist2);
                        exit(-1);
                    }

                    Barrier<elasticIPC>(dist2, dHat2, kappa, barrier[cI]);

                    const VECTOR<T, 3>& Xea0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                    const VECTOR<T, 3>& Xea1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                    const VECTOR<T, 3>& Xeb0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                    const VECTOR<T, 3>& Xeb1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                    Eigen::Matrix<T, 3, 1> ea0_rest(Xea0_rest.data), ea1_rest(Xea1_rest.data), 
                        eb0_rest(Xeb0_rest.data), eb1_rest(Xeb1_rest.data);
                    T eps_x, e;
                    Edge_Edge_Mollifier_Threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest, eps_x);
                    Edge_Edge_Mollifier(ea0, ea1, eb0, eb1, eps_x, e);
                    barrier[cI] *= e;
                }
            }
            else {
                // PT, PE, and PP
                T dist2;
                if (cIVInd[3] >= 0) {
                    // -+++ PT 
                    assert(cIVInd[2] >= 0);
                    const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xt0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xt1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    const VECTOR<T, 3>& Xt2 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                    Eigen::Matrix<T, 3, 1> p(Xp.data), t0(Xt0.data), t1(Xt1.data), t2(Xt2.data);
                    
                    Point_Triangle_Distance(p, t0, t1, t2, dist2);
                }
                else if (cIVInd[2] >= 0) {
                    // -++[-] PE, last digit stores muliplicity
                    const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    Eigen::Matrix<T, 3, 1> p(Xp.data), e0(Xe0.data), e1(Xe1.data);
                    
                    Point_Edge_Distance(p, e0, e1, dist2);
                }
                else {
                    // -+-[-] PP, last digit stores muliplicity
                    const VECTOR<T, 3>& Xp0 = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    Eigen::Matrix<T, 3, 1> p0(Xp0.data), p1(Xp1.data);
                    
                    Point_Point_Distance(p0, p1, dist2);
                }
                dist2 -= thickness2;

                if (dist2 <= 0) {
                    printf("%le distance detected during barrier evaluation!\n", dist2);
                    exit(-1);
                }

                Barrier<elasticIPC>(dist2, dHat2, kappa, barrier[cI]);

                // handle muliplicity
                if (cIVInd[3] < -1) {
                    barrier[cI] *= -cIVInd[3];
                }
            }
            barrier[cI] *= stencilInfo[cI][0];
        }
    }
    E += std::accumulate(barrier.begin(), barrier.end(), T(0));
}

template <class T, int dim, bool elasticIPC = false>
void Compute_Barrier_Gradient(MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo, // weight, dHat2
    T dHat2, T kappa[], T thickness,
    MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    TIMER_FLAG("Compute_Barrier_Gradient");

    if constexpr (elasticIPC) {
        thickness = 0;
    }

    const T thickness2 = thickness * thickness;
    dHat2 += 2 * std::sqrt(dHat2) * thickness;
    
    if constexpr (dim == 2) {
        //TODO: parallelize (loop contains write conflict!)
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            const VECTOR<int, 3>& cIVInd = constraintSet[cI];
            if (cIVInd[2] < 0) {
                // PP
                const VECTOR<T, 2>& Xp0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                const VECTOR<T, 2>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                Eigen::Matrix<T, 2, 1> p0(Xp0.data), p1(Xp1.data);
                
                T dist2;
                Point_Point_Distance(p0, p1, dist2);
                dist2 -= thickness2;
                Eigen::Matrix<T, 4, 1> distGrad;
                Point_Point_Distance_Gradient(p0, p1, distGrad);

                T barrierGrad;
                Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);

                VECTOR<T, 2>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 2>>::g>(nodeAttr.Get_Unchecked(cIVInd[0]));
                VECTOR<T, 2>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 2>>::g>(nodeAttr.Get_Unchecked(cIVInd[1]));
                distGrad *= barrierGrad;
                g0 += distGrad.data();
                g1 += distGrad.data() + 2;
            }
            else {
                // PE
                const VECTOR<T, 2>& Xp = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                const VECTOR<T, 2>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                const VECTOR<T, 2>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                Eigen::Matrix<T, 2, 1> p(Xp.data), e0(Xe0.data), e1(Xe1.data);
                
                T dist2;
                Point_Edge_Distance(p, e0, e1, dist2);
                dist2 -= thickness2;
                Eigen::Matrix<T, 6, 1> distGrad;
                Point_Edge_Distance_Gradient(p, e0, e1, distGrad);

                T barrierGrad;
                Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);

                VECTOR<T, 2>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 2>>::g>(nodeAttr.Get_Unchecked(cIVInd[0]));
                VECTOR<T, 2>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 2>>::g>(nodeAttr.Get_Unchecked(cIVInd[1]));
                VECTOR<T, 2>& g2 = std::get<FIELDS<MESH_NODE_ATTR<T, 2>>::g>(nodeAttr.Get_Unchecked(cIVInd[2]));
                distGrad *= barrierGrad;
                g0 += distGrad.data();
                g1 += distGrad.data() + 2;
                g2 += distGrad.data() + 4;
            }
        }
    }
    else {
        //TODO: parallelize (loop contains write conflict!)
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            const VECTOR<int, 4>& cIVInd = constraintSet[cI];
            assert(cIVInd[1] >= 0);
            if (cIVInd[0] >= 0) {
                // EE
                if (cIVInd[3] >= 0 && cIVInd[2] >= 0) {
                    // ++++ EE, no mollification
                    const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                    const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                    Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);
                    
                    T dist2;
                    Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2);
                    dist2 -= thickness2;
                    Eigen::Matrix<T, 12, 1> distGrad;
                    Edge_Edge_Distance_Gradient(ea0, ea1, eb0, eb1, distGrad);
                    
                    T barrierGrad;
                    Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);

                    VECTOR<T, 3>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[0]));
                    VECTOR<T, 3>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[1]));
                    VECTOR<T, 3>& g2 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[2]));
                    VECTOR<T, 3>& g3 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[3]));
                    distGrad *= stencilInfo[cI][0] * barrierGrad;
                    g0 += distGrad.data();
                    g1 += distGrad.data() + 3;
                    g2 += distGrad.data() + 6;
                    g3 += distGrad.data() + 9;
                }
                else {
                    // EE, PE, or PP with mollification
                    if (cIVInd[3] >= 0) {
                        // ++-+ EE with mollification
                        std::array<int, 4> edgeVInd = {cIVInd[0], cIVInd[1], -cIVInd[2] - 1, cIVInd[3]};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);
                        
                        T dist2;
                        Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2);
                        dist2 -= thickness2;
                        Eigen::Matrix<T, 12, 1> distGrad;
                        Edge_Edge_Distance_Gradient(ea0, ea1, eb0, eb1, distGrad);
                        
                        T b, bGrad;
                        Barrier<elasticIPC>(dist2, dHat2, kappa, b);
                        Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, bGrad);

                        const VECTOR<T, 3>& Xea0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0_rest(Xea0_rest.data), ea1_rest(Xea1_rest.data), 
                            eb0_rest(Xeb0_rest.data), eb1_rest(Xeb1_rest.data);
                        T eps_x, e;
                        Edge_Edge_Mollifier_Threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest, eps_x);
                        Edge_Edge_Mollifier(ea0, ea1, eb0, eb1, eps_x, e);
                        Eigen::Matrix<T, 12, 1> eGrad;
                        Edge_Edge_Mollifier_Gradient(ea0, ea1, eb0, eb1, eps_x, eGrad);

                        VECTOR<T, 3>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        VECTOR<T, 3>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        VECTOR<T, 3>& g2 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        VECTOR<T, 3>& g3 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        eGrad = stencilInfo[cI][0] * ((e * bGrad) * distGrad + (b) * eGrad);
                        g0 += eGrad.data();
                        g1 += eGrad.data() + 3;
                        g2 += eGrad.data() + 6;
                        g3 += eGrad.data() + 9;
                    }
                    else if (cIVInd[2] >= 0) {
                        // +++- PE with mollification, multiplicity 1
                        std::array<int, 4> edgeVInd = {cIVInd[0], -cIVInd[3] - 1, cIVInd[1], cIVInd[2]};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);

                        T dist2;
                        Point_Edge_Distance(ea0, eb0, eb1, dist2);
                        dist2 -= thickness2;
                        Eigen::Matrix<T, 9, 1> distGrad;
                        Point_Edge_Distance_Gradient(ea0, eb0, eb1, distGrad);
                        
                        T b, bGrad;
                        Barrier<elasticIPC>(dist2, dHat2, kappa, b);
                        Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, bGrad);

                        const VECTOR<T, 3>& Xea0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0_rest(Xea0_rest.data), ea1_rest(Xea1_rest.data), 
                            eb0_rest(Xeb0_rest.data), eb1_rest(Xeb1_rest.data);
                        T eps_x, e;
                        Edge_Edge_Mollifier_Threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest, eps_x);
                        Edge_Edge_Mollifier(ea0, ea1, eb0, eb1, eps_x, e);
                        Eigen::Matrix<T, 12, 1> eGrad;
                        Edge_Edge_Mollifier_Gradient(ea0, ea1, eb0, eb1, eps_x, eGrad);

                        VECTOR<T, 3>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        VECTOR<T, 3>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        VECTOR<T, 3>& g2 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        VECTOR<T, 3>& g3 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        eGrad *= stencilInfo[cI][0] * b;
                        distGrad *= e * stencilInfo[cI][0] * bGrad;
                        g0 += eGrad.data();
                        g1 += eGrad.data() + 3;
                        g2 += eGrad.data() + 6;
                        g3 += eGrad.data() + 9;
                        g0 += distGrad.data();
                        g2 += distGrad.data() + 3;
                        g3 += distGrad.data() + 6;
                    }
                    else {
                        // ++-- PP with mollification, multiplicity 1
                        std::array<int, 4> edgeVInd = {cIVInd[0], -cIVInd[2] - 1, cIVInd[1], -cIVInd[3] - 1};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);

                        T dist2;
                        Point_Point_Distance(ea0, eb0, dist2);
                        dist2 -= thickness2;
                        Eigen::Matrix<T, 6, 1> distGrad;
                        Point_Point_Distance_Gradient(ea0, eb0, distGrad);
                        
                        T b, bGrad;
                        Barrier<elasticIPC>(dist2, dHat2, kappa, b);
                        Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, bGrad);

                        const VECTOR<T, 3>& Xea0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0_rest(Xea0_rest.data), ea1_rest(Xea1_rest.data), 
                            eb0_rest(Xeb0_rest.data), eb1_rest(Xeb1_rest.data);
                        T eps_x, e;
                        Edge_Edge_Mollifier_Threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest, eps_x);
                        Edge_Edge_Mollifier(ea0, ea1, eb0, eb1, eps_x, e);
                        Eigen::Matrix<T, 12, 1> eGrad;
                        Edge_Edge_Mollifier_Gradient(ea0, ea1, eb0, eb1, eps_x, eGrad);

                        VECTOR<T, 3>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        VECTOR<T, 3>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        VECTOR<T, 3>& g2 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        VECTOR<T, 3>& g3 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        eGrad *= stencilInfo[cI][0] * b;
                        distGrad *= e * stencilInfo[cI][0] * bGrad;
                        g0 += eGrad.data();
                        g1 += eGrad.data() + 3;
                        g2 += eGrad.data() + 6;
                        g3 += eGrad.data() + 9;
                        g0 += distGrad.data();
                        g2 += distGrad.data() + 3;
                    }
                }
            }
            else {
                // PT, PE, and PP
                if (cIVInd[3] >= 0) {
                    // -+++ PT 
                    assert(cIVInd[2] >= 0);
                    const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xt0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xt1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    const VECTOR<T, 3>& Xt2 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                    Eigen::Matrix<T, 3, 1> p(Xp.data), t0(Xt0.data), t1(Xt1.data), t2(Xt2.data);
                    
                    T dist2;
                    Point_Triangle_Distance(p, t0, t1, t2, dist2);
                    dist2 -= thickness2;
                    Eigen::Matrix<T, 12, 1> distGrad;
                    Point_Triangle_Distance_Gradient(p, t0, t1, t2, distGrad);
                    
                    T barrierGrad;
                    Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);

                    VECTOR<T, 3>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(-cIVInd[0] - 1));
                    VECTOR<T, 3>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[1]));
                    VECTOR<T, 3>& g2 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[2]));
                    VECTOR<T, 3>& g3 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[3]));
                    distGrad *= stencilInfo[cI][0] * barrierGrad;
                    g0 += distGrad.data();
                    g1 += distGrad.data() + 3;
                    g2 += distGrad.data() + 6;
                    g3 += distGrad.data() + 9;
                }
                else if (cIVInd[2] >= 0) {
                    // -++[-] PE, last digit stores muliplicity
                    const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    Eigen::Matrix<T, 3, 1> p(Xp.data), e0(Xe0.data), e1(Xe1.data);
                    
                    T dist2;
                    Point_Edge_Distance(p, e0, e1, dist2);
                    dist2 -= thickness2;
                    Eigen::Matrix<T, 9, 1> distGrad;
                    Point_Edge_Distance_Gradient(p, e0, e1, distGrad);
                    
                    T barrierGrad;
                    Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);

                    VECTOR<T, 3>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(-cIVInd[0] - 1));
                    VECTOR<T, 3>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[1]));
                    VECTOR<T, 3>& g2 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[2]));
                    distGrad *= -cIVInd[3] * stencilInfo[cI][0] * barrierGrad;
                    g0 += distGrad.data();
                    g1 += distGrad.data() + 3;
                    g2 += distGrad.data() + 6;
                }
                else {
                    // -+-[-] PP, last digit stores muliplicity
                    const VECTOR<T, 3>& Xp0 = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    Eigen::Matrix<T, 3, 1> p0(Xp0.data), p1(Xp1.data);
                    
                    T dist2;
                    Point_Point_Distance(p0, p1, dist2);
                    dist2 -= thickness2;
                    Eigen::Matrix<T, 6, 1> distGrad;
                    Point_Point_Distance_Gradient(p0, p1, distGrad);
                    
                    T barrierGrad;
                    Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);

                    VECTOR<T, 3>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(-cIVInd[0] - 1));
                    VECTOR<T, 3>& g1 = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::g>(nodeAttr.Get_Unchecked(cIVInd[1]));
                    distGrad *= -cIVInd[3] * stencilInfo[cI][0] * barrierGrad;
                    g0 += distGrad.data();
                    g1 += distGrad.data() + 3;
                }
            }
        }
    }
}

template <class T, int dim, bool elasticIPC = false>
void Compute_Barrier_Hessian(MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    const std::vector<VECTOR<T, 2>>& stencilInfo, // weight, dHat2
    T dHat2, T kappa[], T thickness,
    bool projectSPD,
    std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Barrier_Hessian");

    if constexpr (elasticIPC) {
        thickness = 0;
    }

    const T thickness2 = thickness * thickness;
    dHat2 += 2 * std::sqrt(dHat2) * thickness;

    if constexpr (dim == 2) {
        BASE_STORAGE<int> threads(constraintSet.size());
        int curStartInd = triplets.size();
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            threads.Append(curStartInd);
            const VECTOR<int, 3>& cIVInd = constraintSet[cI];
            if (cIVInd[2] < 0) {
                // PP, 4x4
                curStartInd += 16;
            }
            else {
                // PE, 6x6
                curStartInd += 36;
            }
        }
        triplets.resize(curStartInd);

        threads.Par_Each([&](int cI, auto data) {
            const auto &[tripletStart] = data;
            const VECTOR<int, 3>& cIVInd = constraintSet[cI];
            if (cIVInd[2] < 0) {
                // PP
                const VECTOR<T, 2>& Xp0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                const VECTOR<T, 2>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                Eigen::Matrix<T, 2, 1> p0(Xp0.data), p1(Xp1.data);
                
                T dist2;
                Point_Point_Distance(p0, p1, dist2);
                dist2 -= thickness2;
                Eigen::Matrix<T, 4, 1> distGrad;
                Point_Point_Distance_Gradient(p0, p1, distGrad);
                Eigen::Matrix<T, 4, 4> distH;
                Point_Point_Distance_Hessian(p0, p1, distH);

                T barrierGrad, barrierH;
                Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);
                Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, barrierH);

                Eigen::Matrix<T, 4, 4> HessianI = barrierH * distGrad * distGrad.transpose() +
                    barrierGrad * distH;
                if (projectSPD) {
                    makePD(HessianI);
                }
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        for (int idI = 0; idI < 2; ++idI) {
                            for (int jdI = 0; jdI < 2; ++jdI) {
                                triplets[tripletStart + (i * 2 + idI) * 4 + j * 2 + jdI] = Eigen::Triplet<T>(
                                    cIVInd[i] * 2 + idI, cIVInd[j] * 2 + jdI,
                                    HessianI(i * 2 + idI, j * 2 + jdI));
                            }
                        }
                    }
                }
            }
            else {
                // PE
                const VECTOR<T, 2>& Xp = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                const VECTOR<T, 2>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                const VECTOR<T, 2>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                Eigen::Matrix<T, 2, 1> p(Xp.data), e0(Xe0.data), e1(Xe1.data);
                
                T dist2;
                Point_Edge_Distance(p, e0, e1, dist2);
                dist2 -= thickness2;
                Eigen::Matrix<T, 6, 1> distGrad;
                Point_Edge_Distance_Gradient(p, e0, e1, distGrad);
                Eigen::Matrix<T, 6, 6> distH;
                Point_Edge_Distance_Hessian(p, e0, e1, distH);

                T barrierGrad, barrierH;
                Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);
                Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, barrierH);

                Eigen::Matrix<T, 6, 6> HessianI = barrierH * distGrad * distGrad.transpose() +
                    barrierGrad * distH;
                if (projectSPD) {
                    makePD(HessianI);
                }
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int idI = 0; idI < 2; ++idI) {
                            for (int jdI = 0; jdI < 2; ++jdI) {
                                triplets[tripletStart + (i * 2 + idI) * 6 + j * 2 + jdI] = Eigen::Triplet<T>(
                                    cIVInd[i] * 2 + idI, cIVInd[j] * 2 + jdI,
                                    HessianI(i * 2 + idI, j * 2 + jdI));
                            }
                        }
                    }
                }
            }
        });
    }
    else {
        BASE_STORAGE<int> threads(constraintSet.size());
        int curStartInd = triplets.size();
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            threads.Append(curStartInd);
            const VECTOR<int, 4>& cIVInd = constraintSet[cI];
            if (cIVInd[0] >= 0 || cIVInd[3] >= 0) {
                // EE or PT, 12x12
                curStartInd += 144;
            }
            else if (cIVInd[2] >= 0) {
                // PE, 9x9
                curStartInd += 81;    
            }
            else {
                // PP, 6x6
                curStartInd += 36;
            }
        }
        triplets.resize(curStartInd);

        threads.Par_Each([&](int cI, auto data) {
            const auto &[tripletStart] = data;
            VECTOR<int, 4> cIVInd = constraintSet[cI]; //NOTE: copy to be able to modify in the loop if needed
            assert(cIVInd[1] >= 0);
            if (cIVInd[0] >= 0) {
                // EE
                if (cIVInd[3] >= 0 && cIVInd[2] >= 0) {
                    // ++++ EE, no mollification
                    const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                    const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                    Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);
                    
                    T dist2;
                    Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2);
                    dist2 -= thickness2;
                    Eigen::Matrix<T, 12, 1> distGrad;
                    Edge_Edge_Distance_Gradient(ea0, ea1, eb0, eb1, distGrad);
                    Eigen::Matrix<T, 12, 12> distH;
                    Edge_Edge_Distance_Hessian(ea0, ea1, eb0, eb1, distH);
                    
                    T barrierGrad, barrierH;
                    Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);
                    Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, barrierH);

                    Eigen::Matrix<T, 12, 12> HessianI = ((barrierH) * distGrad) * distGrad.transpose() +
                        (barrierGrad) * distH;
                    HessianI *= stencilInfo[cI][0];
                    if (projectSPD) {
                        makePD(HessianI);
                    }
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            for (int idI = 0; idI < 3; ++idI) {
                                for (int jdI = 0; jdI < 3; ++jdI) {
                                    triplets[tripletStart + (i * 3 + idI) * 12 + j * 3 + jdI] = Eigen::Triplet<T>(
                                        cIVInd[i] * 3 + idI, cIVInd[j] * 3 + jdI,
                                        HessianI(i * 3 + idI, j * 3 + jdI));
                                }
                            }
                        }
                    }
                }
                else {
                    // EE, PE, or PP with mollification
                    if (cIVInd[3] >= 0) {
                        // ++-+ EE with mollification
                        std::array<int, 4> edgeVInd = {cIVInd[0], cIVInd[1], -cIVInd[2] - 1, cIVInd[3]};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);
                        
                        T dist2;
                        Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2);
                        dist2 -= thickness2;
                        Eigen::Matrix<T, 12, 1> distGrad;
                        Edge_Edge_Distance_Gradient(ea0, ea1, eb0, eb1, distGrad);
                        Eigen::Matrix<T, 12, 12> distH;
                        Edge_Edge_Distance_Hessian(ea0, ea1, eb0, eb1, distH);
                        
                        T b, bGrad, bH;
                        Barrier<elasticIPC>(dist2, dHat2, kappa, b);
                        Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, bGrad);
                        Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, bH);

                        const VECTOR<T, 3>& Xea0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0_rest(Xea0_rest.data), ea1_rest(Xea1_rest.data), 
                            eb0_rest(Xeb0_rest.data), eb1_rest(Xeb1_rest.data);
                        T eps_x, e;
                        Edge_Edge_Mollifier_Threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest, eps_x);
                        Edge_Edge_Mollifier(ea0, ea1, eb0, eb1, eps_x, e);
                        Eigen::Matrix<T, 12, 1> eGrad;
                        Edge_Edge_Mollifier_Gradient(ea0, ea1, eb0, eb1, eps_x, eGrad);
                        Eigen::Matrix<T, 12, 12> eH;
                        Edge_Edge_Mollifier_Hessian(ea0, ea1, eb0, eb1, eps_x, eH);

                        Eigen::Matrix<T, 12, 12> kappa_bGrad_eGradT = (bGrad * distGrad) * eGrad.transpose();
                        Eigen::Matrix<T, 12, 12> HessianI = kappa_bGrad_eGradT + kappa_bGrad_eGradT.transpose() + 
                            b * eH + (e * bH * distGrad) * distGrad.transpose() + e * bGrad * distH;
                        HessianI *= stencilInfo[cI][0];
                        if (projectSPD) {
                            makePD(HessianI);
                        }
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                for (int idI = 0; idI < 3; ++idI) {
                                    for (int jdI = 0; jdI < 3; ++jdI) {
                                        triplets[tripletStart + (i * 3 + idI) * 12 + j * 3 + jdI] = Eigen::Triplet<T>(edgeVInd[i] * 3 + idI, edgeVInd[j] * 3 + jdI,
                                            HessianI(i * 3 + idI, j * 3 + jdI));
                                    }
                                }
                            }
                        }
                    }
                    else if (cIVInd[2] >= 0) {
                        // +++- PE with mollification, multiplicity 1
                        std::array<int, 4> edgeVInd = {cIVInd[0], -cIVInd[3] - 1, cIVInd[1], cIVInd[2]};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);

                        T dist2;
                        Point_Edge_Distance(ea0, eb0, eb1, dist2);
                        dist2 -= thickness2;
                        Eigen::Matrix<T, 9, 1> distGrad;
                        Point_Edge_Distance_Gradient(ea0, eb0, eb1, distGrad);
                        Eigen::Matrix<T, 9, 9> distH;
                        Point_Edge_Distance_Hessian(ea0, eb0, eb1, distH);
                        
                        T b, bGrad, bH;
                        Barrier<elasticIPC>(dist2, dHat2, kappa, b);
                        Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, bGrad);
                        Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, bH);

                        const VECTOR<T, 3>& Xea0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0_rest(Xea0_rest.data), ea1_rest(Xea1_rest.data), 
                            eb0_rest(Xeb0_rest.data), eb1_rest(Xeb1_rest.data);
                        T eps_x, e;
                        Edge_Edge_Mollifier_Threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest, eps_x);
                        Edge_Edge_Mollifier(ea0, ea1, eb0, eb1, eps_x, e);
                        Eigen::Matrix<T, 12, 1> eGrad;
                        Edge_Edge_Mollifier_Gradient(ea0, ea1, eb0, eb1, eps_x, eGrad);
                        Eigen::Matrix<T, 12, 12> eH;
                        Edge_Edge_Mollifier_Hessian(ea0, ea1, eb0, eb1, eps_x, eH);

                        Eigen::Matrix<T, 12, 12> HessianI = (b) * eH;
                        Eigen::Matrix<T, 9, 9> kappa_e_bH = ((e * bH) * distGrad) * distGrad.transpose() + (e * bGrad) * distH;
                        HessianI.template block<3, 3>(0, 0) += kappa_e_bH.template block<3, 3>(0, 0);
                        HessianI.template block<3, 6>(0, 6) += kappa_e_bH.template block<3, 6>(0, 3);
                        HessianI.template block<6, 3>(6, 0) += kappa_e_bH.template block<6, 3>(3, 0);
                        HessianI.template block<6, 6>(6, 6) += kappa_e_bH.template block<6, 6>(3, 3);
                        Eigen::Matrix<T, 9, 12> kappa_bGrad_eGradT = ((bGrad) * distGrad) * eGrad.transpose();
                        HessianI.template block<3, 12>(0, 0) += kappa_bGrad_eGradT.template block<3, 12>(0, 0);
                        HessianI.template block<6, 12>(6, 0) += kappa_bGrad_eGradT.template block<6, 12>(3, 0);
                        HessianI.template block<12, 3>(0, 0) += kappa_bGrad_eGradT.template block<3, 12>(0, 0).transpose();
                        HessianI.template block<12, 6>(0, 6) += kappa_bGrad_eGradT.template block<6, 12>(3, 0).transpose();
                        HessianI *= stencilInfo[cI][0];
                        if (projectSPD) {
                            makePD(HessianI);
                        }
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                for (int idI = 0; idI < 3; ++idI) {
                                    for (int jdI = 0; jdI < 3; ++jdI) {
                                        triplets[tripletStart + (i * 3 + idI) * 12 + j * 3 + jdI] = Eigen::Triplet<T>(edgeVInd[i] * 3 + idI, edgeVInd[j] * 3 + jdI,
                                            HessianI(i * 3 + idI, j * 3 + jdI));
                                    }
                                }
                            }
                        }
                    }
                    else {
                        // ++-- PP with mollification, multiplicity 1
                        std::array<int, 4> edgeVInd = {cIVInd[0], -cIVInd[2] - 1, cIVInd[1], -cIVInd[3] - 1};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);

                        T dist2;
                        Point_Point_Distance(ea0, eb0, dist2);
                        dist2 -= thickness2;
                        Eigen::Matrix<T, 6, 1> distGrad;
                        Point_Point_Distance_Gradient(ea0, eb0, distGrad);
                        Eigen::Matrix<T, 6, 6> distH;
                        Point_Point_Distance_Hessian(ea0, eb0, distH);
                        
                        T b, bGrad, bH;
                        Barrier<elasticIPC>(dist2, dHat2, kappa, b);
                        Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, bGrad);
                        Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, bH);

                        const VECTOR<T, 3>& Xea0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1_rest = std::get<FIELDS<MESH_NODE_ATTR<T, 3>>::x0>(nodeAttr.Get_Unchecked(edgeVInd[3]));
                        Eigen::Matrix<T, 3, 1> ea0_rest(Xea0_rest.data), ea1_rest(Xea1_rest.data), 
                            eb0_rest(Xeb0_rest.data), eb1_rest(Xeb1_rest.data);
                        T eps_x, e;
                        Edge_Edge_Mollifier_Threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest, eps_x);
                        Edge_Edge_Mollifier(ea0, ea1, eb0, eb1, eps_x, e);
                        Eigen::Matrix<T, 12, 1> eGrad;
                        Edge_Edge_Mollifier_Gradient(ea0, ea1, eb0, eb1, eps_x, eGrad);
                        Eigen::Matrix<T, 12, 12> eH;
                        Edge_Edge_Mollifier_Hessian(ea0, ea1, eb0, eb1, eps_x, eH);

                        Eigen::Matrix<T, 12, 12> HessianI = (b) * eH;
                        Eigen::Matrix<T, 6, 6> kappa_e_bH = ((e * bH) * distGrad) * distGrad.transpose() + (e * bGrad) * distH;
                        HessianI.template block<3, 3>(0, 0) += kappa_e_bH.template block<3, 3>(0, 0);
                        HessianI.template block<3, 3>(0, 6) += kappa_e_bH.template block<3, 3>(0, 3);
                        HessianI.template block<3, 3>(6, 0) += kappa_e_bH.template block<3, 3>(3, 0);
                        HessianI.template block<3, 3>(6, 6) += kappa_e_bH.template block<3, 3>(3, 3);
                        Eigen::Matrix<T, 6, 12> kappa_bGrad_eGradT = ((bGrad) * distGrad) * eGrad.transpose();
                        HessianI.template block<3, 12>(0, 0) += kappa_bGrad_eGradT.template block<3, 12>(0, 0);
                        HessianI.template block<3, 12>(6, 0) += kappa_bGrad_eGradT.template block<3, 12>(3, 0);
                        HessianI.template block<12, 3>(0, 0) += kappa_bGrad_eGradT.template block<3, 12>(0, 0).transpose();
                        HessianI.template block<12, 3>(0, 6) += kappa_bGrad_eGradT.template block<3, 12>(3, 0).transpose();
                        HessianI *= stencilInfo[cI][0];
                        if (projectSPD) {
                            makePD(HessianI);
                        }
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                for (int idI = 0; idI < 3; ++idI) {
                                    for (int jdI = 0; jdI < 3; ++jdI) {
                                        triplets[tripletStart + (i * 3 + idI) * 12 + j * 3 + jdI] = Eigen::Triplet<T>(edgeVInd[i] * 3 + idI, edgeVInd[j] * 3 + jdI,
                                            HessianI(i * 3 + idI, j * 3 + jdI));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else {
                // PT, PE, and PP
                cIVInd[0] = -cIVInd[0] - 1;
                if (cIVInd[3] >= 0) {
                    // -+++ PT 
                    assert(cIVInd[2] >= 0);
                    const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                    const VECTOR<T, 3>& Xt0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xt1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    const VECTOR<T, 3>& Xt2 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                    Eigen::Matrix<T, 3, 1> p(Xp.data), t0(Xt0.data), t1(Xt1.data), t2(Xt2.data);
                    
                    T dist2;
                    Point_Triangle_Distance(p, t0, t1, t2, dist2);
                    dist2 -= thickness2;
                    Eigen::Matrix<T, 12, 1> distGrad;
                    Point_Triangle_Distance_Gradient(p, t0, t1, t2, distGrad);
                    Eigen::Matrix<T, 12, 12> distH;
                    Point_Triangle_Distance_Hessian(p, t0, t1, t2, distH);
                    
                    T barrierGrad, barrierH;
                    Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);
                    Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, barrierH);

                    Eigen::Matrix<T, 12, 12> HessianI = ((barrierH) * distGrad) * distGrad.transpose() +
                        (barrierGrad) * distH;
                    HessianI *= stencilInfo[cI][0];
                    if (projectSPD) {
                        makePD(HessianI);
                    }
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            for (int idI = 0; idI < 3; ++idI) {
                                for (int jdI = 0; jdI < 3; ++jdI) {
                                    triplets[tripletStart + (i * 3 + idI) * 12 + j * 3 + jdI] = Eigen::Triplet<T>(cIVInd[i] * 3 + idI, cIVInd[j] * 3 + jdI,
                                        HessianI(i * 3 + idI, j * 3 + jdI));
                                }
                            }
                        }
                    }
                }
                else if (cIVInd[2] >= 0) {
                    // -++[-] PE, last digit stores muliplicity
                    const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                    const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    Eigen::Matrix<T, 3, 1> p(Xp.data), e0(Xe0.data), e1(Xe1.data);
                    
                    T dist2;
                    Point_Edge_Distance(p, e0, e1, dist2);
                    dist2 -= thickness2;
                    Eigen::Matrix<T, 9, 1> distGrad;
                    Point_Edge_Distance_Gradient(p, e0, e1, distGrad);
                    Eigen::Matrix<T, 9, 9> distH;
                    Point_Edge_Distance_Hessian(p, e0, e1, distH);
                    
                    T barrierGrad, barrierH;
                    Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);
                    Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, barrierH);

                    Eigen::Matrix<T, 9, 9> HessianI = ((-cIVInd[3] * barrierH) * distGrad) * distGrad.transpose() +
                        (-cIVInd[3] * barrierGrad) * distH;
                    HessianI *= stencilInfo[cI][0];
                    if (projectSPD) {
                        makePD(HessianI);
                    }
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            for (int idI = 0; idI < 3; ++idI) {
                                for (int jdI = 0; jdI < 3; ++jdI) {
                                    triplets[tripletStart + (i * 3 + idI) * 9 + j * 3 + jdI] = Eigen::Triplet<T>(cIVInd[i] * 3 + idI, cIVInd[j] * 3 + jdI,
                                        HessianI(i * 3 + idI, j * 3 + jdI));
                                }
                            }
                        }
                    }
                }
                else {
                    // -+-[-] PP, last digit stores muliplicity
                    const VECTOR<T, 3>& Xp0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                    const VECTOR<T, 3>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    Eigen::Matrix<T, 3, 1> p0(Xp0.data), p1(Xp1.data);
                    
                    T dist2;
                    Point_Point_Distance(p0, p1, dist2);
                    dist2 -= thickness2;
                    Eigen::Matrix<T, 6, 1> distGrad;
                    Point_Point_Distance_Gradient(p0, p1, distGrad);
                    Eigen::Matrix<T, 6, 6> distH;
                    Point_Point_Distance_Hessian(p0, p1, distH);
                    
                    T barrierGrad, barrierH;
                    Barrier_Gradient<elasticIPC>(dist2, dHat2, kappa, barrierGrad);
                    Barrier_Hessian<elasticIPC>(dist2, dHat2, kappa, barrierH);

                    Eigen::Matrix<T, 6, 6> HessianI = ((-cIVInd[3] * barrierH) * distGrad) * distGrad.transpose() +
                        (-cIVInd[3] * barrierGrad) * distH;
                    HessianI *= stencilInfo[cI][0];
                    if (projectSPD) {
                        makePD(HessianI);
                    }
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            for (int idI = 0; idI < 3; ++idI) {
                                for (int jdI = 0; jdI < 3; ++jdI) {
                                    triplets[tripletStart + (i * 3 + idI) * 6 + j * 3 + jdI] = Eigen::Triplet<T>(cIVInd[i] * 3 + idI, cIVInd[j] * 3 + jdI,
                                        HessianI(i * 3 + idI, j * 3 + jdI));
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}

template <class T>
T getSmallestPositiveRealQuadRoot(T a, T b, T c, T tol)
{
    // return negative value if no positive real root is found
    T t;
    if (abs(a) <= tol)
        t = -c / b;
    else {
        double desc = b * b - 4 * a * c;
        if (desc > 0) {
            t = (-b - sqrt(desc)) / (2 * a);
            if (t < 0)
                t = (-b + sqrt(desc)) / (2 * a);
        }
        else // desv<0 ==> imag
            t = -1;
    }
    return t;
}

template <class T>
T getSmallestPositiveRealCubicRoot(T a, T b, T c, T d, T tol)
{
    // return negative value if no positive real root is found
    T t = -1;
    if (abs(a) <= tol)
        t = getSmallestPositiveRealQuadRoot(b, c, d, tol);
    else {
        std::complex<T> i(0, 1);
        std::complex<T> delta0(b * b - 3 * a * c, 0);
        std::complex<T> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        std::complex<T> C = pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
        if (std::abs(C) == 0.0) {
            // a corner case listed by wikipedia found by our collaborate from another project
            C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
        }
        std::complex<T> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
        std::complex<T> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;
        std::complex<T> t1 = (b + C + delta0 / C) / (-3.0 * a);
        std::complex<T> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
        std::complex<T> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);

        if ((abs(imag(t1)) < tol) && (real(t1) > 0))
            t = real(t1);
        if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
            t = real(t2);
        if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
            t = real(t3);
    }
    return t;
}

template <class T, int dim>
void Compute_Inversion_Free_StepSize(MESH_NODE<T, dim>& X,
    MESH_ELEM<dim>& Elem,
    const std::vector<T>& searchDir,
    T& stepSize)
{
    std::vector<T> stepSizeNI(Elem.size, stepSize);
    if constexpr (dim == 2) {
        Elem.Par_Each([&](int bNI, auto data) {
            auto &[elemVInd] = data;
            T x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]))(0);
            T x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]))(0);
            T x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]))(0);

            T y1 = std::get<0>(X.Get_Unchecked(elemVInd[0]))(1);
            T y2 = std::get<0>(X.Get_Unchecked(elemVInd[1]))(1);
            T y3 = std::get<0>(X.Get_Unchecked(elemVInd[2]))(1);

            T p1 = searchDir[elemVInd[0] * 2];
            T p2 = searchDir[elemVInd[1] * 2];
            T p3 = searchDir[elemVInd[2] * 2];

            T q1 = searchDir[elemVInd[0] * 2 + 1];
            T q2 = searchDir[elemVInd[1] * 2 + 1];
            T q3 = searchDir[elemVInd[2] * 2 + 1];

            T a = p1 * q2 - p2 * q1 - p1 * q3 + p3 * q1 + p2 * q3 - p3 * q2;
            T b = p1 * y2 - p2 * y1 - q1 * x2 + q2 * x1 - p1 * y3 + p3 * y1 + q1 * x3 - q3 * x1 + p2 * y3 - p3 * y2 - q2 * x3 + q3 * x2;
            T c = (1.0 - 0.1) * (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2);

            T t = getSmallestPositiveRealQuadRoot(a, b, c, 1.e-6);
            if (t >= 0) {
                stepSizeNI[bNI] = t;
            }
            else {
                stepSizeNI[bNI] = 1e20;
            }
        });
    }
    else {
        Elem.Par_Each([&](int bNI, auto data) {
            auto &[elemVInd] = data;
            T x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]))(0);
            T x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]))(0);
            T x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]))(0);
            T x4 = std::get<0>(X.Get_Unchecked(elemVInd[3]))(0);

            T y1 = std::get<0>(X.Get_Unchecked(elemVInd[0]))(1);
            T y2 = std::get<0>(X.Get_Unchecked(elemVInd[1]))(1);
            T y3 = std::get<0>(X.Get_Unchecked(elemVInd[2]))(1);
            T y4 = std::get<0>(X.Get_Unchecked(elemVInd[3]))(1);

            T z1 = std::get<0>(X.Get_Unchecked(elemVInd[0]))(2);
            T z2 = std::get<0>(X.Get_Unchecked(elemVInd[1]))(2);
            T z3 = std::get<0>(X.Get_Unchecked(elemVInd[2]))(2);
            T z4 = std::get<0>(X.Get_Unchecked(elemVInd[3]))(2);

            int _3Fii0 = elemVInd[0] * 3;
            int _3Fii1 = elemVInd[1] * 3;
            int _3Fii2 = elemVInd[2] * 3;
            int _3Fii3 = elemVInd[3] * 3;

            T p1 = searchDir[_3Fii0];
            T p2 = searchDir[_3Fii1];
            T p3 = searchDir[_3Fii2];
            T p4 = searchDir[_3Fii3];

            T q1 = searchDir[_3Fii0 + 1];
            T q2 = searchDir[_3Fii1 + 1];
            T q3 = searchDir[_3Fii2 + 1];
            T q4 = searchDir[_3Fii3 + 1];

            T r1 = searchDir[_3Fii0 + 2];
            T r2 = searchDir[_3Fii1 + 2];
            T r3 = searchDir[_3Fii2 + 2];
            T r4 = searchDir[_3Fii3 + 2];

            T a = -p1 * q2 * r3 + p1 * r2 * q3 + q1 * p2 * r3 - q1 * r2 * p3 - r1 * p2 * q3 + r1 * q2 * p3 + p1 * q2 * r4 - p1 * r2 * q4 - q1 * p2 * r4 + q1 * r2 * p4 + r1 * p2 * q4 - r1 * q2 * p4 - p1 * q3 * r4 + p1 * r3 * q4 + q1 * p3 * r4 - q1 * r3 * p4 - r1 * p3 * q4 + r1 * q3 * p4 + p2 * q3 * r4 - p2 * r3 * q4 - q2 * p3 * r4 + q2 * r3 * p4 + r2 * p3 * q4 - r2 * q3 * p4;
            T b = -x1 * q2 * r3 + x1 * r2 * q3 + y1 * p2 * r3 - y1 * r2 * p3 - z1 * p2 * q3 + z1 * q2 * p3 + x2 * q1 * r3 - x2 * r1 * q3 - y2 * p1 * r3 + y2 * r1 * p3 + z2 * p1 * q3 - z2 * q1 * p3 - x3 * q1 * r2 + x3 * r1 * q2 + y3 * p1 * r2 - y3 * r1 * p2 - z3 * p1 * q2 + z3 * q1 * p2 + x1 * q2 * r4 - x1 * r2 * q4 - y1 * p2 * r4 + y1 * r2 * p4 + z1 * p2 * q4 - z1 * q2 * p4 - x2 * q1 * r4 + x2 * r1 * q4 + y2 * p1 * r4 - y2 * r1 * p4 - z2 * p1 * q4 + z2 * q1 * p4 + x4 * q1 * r2 - x4 * r1 * q2 - y4 * p1 * r2 + y4 * r1 * p2 + z4 * p1 * q2 - z4 * q1 * p2 - x1 * q3 * r4 + x1 * r3 * q4 + y1 * p3 * r4 - y1 * r3 * p4 - z1 * p3 * q4 + z1 * q3 * p4 + x3 * q1 * r4 - x3 * r1 * q4 - y3 * p1 * r4 + y3 * r1 * p4 + z3 * p1 * q4 - z3 * q1 * p4 - x4 * q1 * r3 + x4 * r1 * q3 + y4 * p1 * r3 - y4 * r1 * p3 - z4 * p1 * q3 + z4 * q1 * p3 + x2 * q3 * r4 - x2 * r3 * q4 - y2 * p3 * r4 + y2 * r3 * p4 + z2 * p3 * q4 - z2 * q3 * p4 - x3 * q2 * r4 + x3 * r2 * q4 + y3 * p2 * r4 - y3 * r2 * p4 - z3 * p2 * q4 + z3 * q2 * p4 + x4 * q2 * r3 - x4 * r2 * q3 - y4 * p2 * r3 + y4 * r2 * p3 + z4 * p2 * q3 - z4 * q2 * p3;
            T c = -x1 * y2 * r3 + x1 * z2 * q3 + x1 * y3 * r2 - x1 * z3 * q2 + y1 * x2 * r3 - y1 * z2 * p3 - y1 * x3 * r2 + y1 * z3 * p2 - z1 * x2 * q3 + z1 * y2 * p3 + z1 * x3 * q2 - z1 * y3 * p2 - x2 * y3 * r1 + x2 * z3 * q1 + y2 * x3 * r1 - y2 * z3 * p1 - z2 * x3 * q1 + z2 * y3 * p1 + x1 * y2 * r4 - x1 * z2 * q4 - x1 * y4 * r2 + x1 * z4 * q2 - y1 * x2 * r4 + y1 * z2 * p4 + y1 * x4 * r2 - y1 * z4 * p2 + z1 * x2 * q4 - z1 * y2 * p4 - z1 * x4 * q2 + z1 * y4 * p2 + x2 * y4 * r1 - x2 * z4 * q1 - y2 * x4 * r1 + y2 * z4 * p1 + z2 * x4 * q1 - z2 * y4 * p1 - x1 * y3 * r4 + x1 * z3 * q4 + x1 * y4 * r3 - x1 * z4 * q3 + y1 * x3 * r4 - y1 * z3 * p4 - y1 * x4 * r3 + y1 * z4 * p3 - z1 * x3 * q4 + z1 * y3 * p4 + z1 * x4 * q3 - z1 * y4 * p3 - x3 * y4 * r1 + x3 * z4 * q1 + y3 * x4 * r1 - y3 * z4 * p1 - z3 * x4 * q1 + z3 * y4 * p1 + x2 * y3 * r4 - x2 * z3 * q4 - x2 * y4 * r3 + x2 * z4 * q3 - y2 * x3 * r4 + y2 * z3 * p4 + y2 * x4 * r3 - y2 * z4 * p3 + z2 * x3 * q4 - z2 * y3 * p4 - z2 * x4 * q3 + z2 * y4 * p3 + x3 * y4 * r2 - x3 * z4 * q2 - y3 * x4 * r2 + y3 * z4 * p2 + z3 * x4 * q2 - z3 * y4 * p2;
            T d = (1.0 - 0.2) * (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 - z1 * x2 * y3 + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 - y1 * x2 * z4 + y1 * z2 * x4 + z1 * x2 * y4 - z1 * y2 * x4 - x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4 - y1 * z3 * x4 - z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4 - y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);

            T t = getSmallestPositiveRealCubicRoot(a, b, c, d, 1.e-6);
            if (t >= 0) {
                stepSizeNI[bNI] = t;
            }
            else {
                stepSizeNI[bNI] = 1e20;
            }
        });
    }
    stepSize = std::min(stepSize, *std::min_element(stepSizeNI.begin(), stepSizeNI.end()));
}

template <class T, int dim, bool shell = false, bool elasticIPC = false>
void Compute_Intersection_Free_StepSize(MESH_NODE<T, dim>& X,
    const std::vector<int>& boundaryNode,
    const std::vector<VECTOR<int, 2>>& boundaryEdge,
    const std::vector<VECTOR<int, 3>>& boundaryTri,
    const std::vector<int>& particle,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::map<int, std::set<int>>& NNExclusion,
    const VECTOR<int, 2>& codimBNStartInd,
    const std::vector<bool>& DBCb,
    const std::vector<T>& searchDir, 
    T thickness, T& stepSize)
{
    TIMER_FLAG("Compute_Intersection_Free_StepSize");
    //TODO: CFL?

    if constexpr (elasticIPC) {
        thickness = 0;
    }

#define USE_SH_LFSS
#ifdef USE_SH_LFSS
    SPATIAL_HASH<T, dim> sh;
    {
        TIMER_FLAG("Compute_Intersection_Free_StepSize_Build_Hash");
        sh.Build(X, boundaryNode, boundaryEdge, boundaryTri, searchDir, stepSize, 1.0, thickness);
    }
#endif

    if constexpr (dim == 2) {
        BASE_STORAGE<int> threads(boundaryNode.size());
        for (int i = 0; i < boundaryNode.size(); ++i) {
            threads.Append(i);
        }

        std::vector<T> stepSizeNI(boundaryNode.size(), stepSize);
        threads.Par_Each([&](int bNI, auto data){
            int nI = boundaryNode[bNI];
            const VECTOR<T, 2>& Xp = std::get<0>(X.Get_Unchecked(nI));
            Eigen::Matrix<T, 2, 1> p(Xp.data), dp(searchDir.data() + nI * 2);
#ifdef USE_SH_LFSS
            std::unordered_set<int> edgeInds;
            sh.Query_Point_For_Edges(bNI, edgeInds);
            for (const auto& eInd : edgeInds) {
                const VECTOR<int, 2>& eI = boundaryEdge[eInd];
#else
            for (const auto& eI : boundaryEdge) {
#endif
                if (nI == eI[0] || nI == eI[1] || 
                    (DBCb[nI] && DBCb[eI[0]] && DBCb[eI[1]])) 
                {
                    continue;
                }
                if constexpr (shell) {
                    if (((nI % 3 == 0) && (nI + 2 == eI[0] || nI + 2 == eI[1])) ||
                        ((nI % 3 == 2) && (nI - 2 == eI[0] || nI - 2 == eI[1]))) { continue; }
                }

                const VECTOR<T, 2>& Xe0 = std::get<0>(X.Get_Unchecked(eI[0]));
                const VECTOR<T, 2>& Xe1 = std::get<0>(X.Get_Unchecked(eI[1]));
                Eigen::Matrix<T, 2, 1> e0(Xe0.data), e1(Xe1.data);
                Eigen::Matrix<T, 2, 1> de0(searchDir.data() + eI[0] * 2);
                Eigen::Matrix<T, 2, 1> de1(searchDir.data() + eI[1] * 2);

                if (!Point_Edge_CCD_Broadphase(p, e0, e1, dp, de0, de1, thickness)) {
                    continue;
                }

                T stepSizeNIEI;
                if(Point_Edge_CCD(p, e0, e1, dp, de0, de1, 0.1, stepSizeNIEI)) { //TODO: support thickness
                    if (stepSizeNI[bNI] > stepSizeNIEI) {
                        stepSizeNI[bNI] = stepSizeNIEI;
                    }
                }
            }
        });
        stepSize = *std::min_element(stepSizeNI.begin(), stepSizeNI.end());
    }
    else {
        // point-triangle
        { TIMER_FLAG("Compute_Intersection_Free_StepSize_PT");
        BASE_STORAGE<int> threadsPT(boundaryNode.size());
        for (int i = 0; i < boundaryNode.size(); ++i) {
            threadsPT.Append(i);
        }

        std::vector<T> largestAlphasPT(boundaryNode.size());
        threadsPT.Par_Each([&](int svI, auto data) {
            int vI = boundaryNode[svI];
            const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(vI));
            Eigen::Matrix<T, 3, 1> p(Xp.data), dp(searchDir.data() + vI * dim);
            largestAlphasPT[svI] = stepSize;
#ifdef USE_SH_LFSS
            std::unordered_set<int> sPointInds, sEdgeInds, sTriInds;
            sh.Query_Point_For_Primitives(svI, sPointInds, sEdgeInds, sTriInds);
            for (const auto& sfI : sTriInds)
#else
            for (int sfI = 0; sfI < boundaryTri.size(); ++sfI) 
#endif
            {
                const VECTOR<int, 3>& sfVInd = boundaryTri[sfI];
                if (!(vI == sfVInd[0] || vI == sfVInd[1] || vI == sfVInd[2]) &&
                    !(DBCb[vI] && DBCb[sfVInd[0]] && DBCb[sfVInd[1]] && DBCb[sfVInd[2]])) 
                {
                    auto NNEFinder = NNExclusion.find(vI);
                    if (NNEFinder != NNExclusion.end() &&
                        (NNEFinder->second.find(sfVInd[0]) != NNEFinder->second.end() ||
                        NNEFinder->second.find(sfVInd[1]) != NNEFinder->second.end() ||
                        NNEFinder->second.find(sfVInd[2]) != NNEFinder->second.end()))
                    {
                        continue;
                    }
                    if constexpr (shell) {
                        int vpair = (vI % 2 == 0) ? (vI + 1) : vI - 1;
                        if (vpair == sfVInd[0] || vpair == sfVInd[1] || vpair == sfVInd[2]) {
                            continue;
                        }
                    }
                    const VECTOR<T, 3>& Xt0 = std::get<0>(X.Get_Unchecked(sfVInd[0]));
                    const VECTOR<T, 3>& Xt1 = std::get<0>(X.Get_Unchecked(sfVInd[1]));
                    const VECTOR<T, 3>& Xt2 = std::get<0>(X.Get_Unchecked(sfVInd[2]));
                    Eigen::Matrix<T, 3, 1> t0(Xt0.data), t1(Xt1.data), t2(Xt2.data);
                    Eigen::Matrix<T, 3, 1> dt0(searchDir.data() + sfVInd[0] * dim), 
                        dt1(searchDir.data() + sfVInd[1] * dim), dt2(searchDir.data() + sfVInd[2] * dim);

                    if (!Point_Triangle_CCD_Broadphase(p, t0, t1, t2, dp, dt0, dt1, dt2, thickness)) {
                        continue;
                    }

                    T largestAlpha = largestAlphasPT[svI];
                    if (Point_Triangle_CCD(p, t0, t1, t2, dp, dt0, dt1, dt2, 0.1, thickness, largestAlpha)) {
                        if (largestAlphasPT[svI] > largestAlpha) {
                            largestAlphasPT[svI] = largestAlpha;
                        }
                    }
                    if (largestAlpha == 0) {
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
                        exit(-1);
                    }
                }
            }

//             // PE
// #ifdef USE_SH_LFSS
//             for (const auto& eI : sEdgeInds)
// #else
//             for (int eI = 0; eI < boundaryEdge.size(); ++eI)
// #endif
//             {
//                 const VECTOR<int, 2>& meshEI = boundaryEdge[eI];
//                 if (!(vI == meshEI[0] || vI == meshEI[1]) &&
//                     !(DBCb[vI] && DBCb[meshEI[0]] && DBCb[meshEI[1]])) 
//                 {
//                     if constexpr (shell) {
//                         int vpair = (vI % 2 == 0) ? (vI + 1) : vI - 1;
//                         if (vpair == meshEI[0] || vpair == meshEI[1]) {
//                             continue;
//                         }
//                     }
//                     const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(meshEI[0]));
//                     const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(meshEI[1]));
//                     Eigen::Matrix<T, 3, 1> e0(Xe0.data), e1(Xe1.data);
//                     Eigen::Matrix<T, 3, 1> de0(searchDir.data() + meshEI[0] * dim), 
//                         de1(searchDir.data() + meshEI[1] * dim);

//                     if (!Point_Edge_CCD_Broadphase(p, e0, e1, dp, de0, de1, thickness)) {
//                         continue;
//                     }

//                     T largestAlpha = 1.0;
//                     if (Point_Edge_CCD(p, e0, e1, dp, de0, de1, 0.1, thickness, largestAlpha)) {
//                         if (largestAlphasPT[svI] > largestAlpha) {
//                             largestAlphasPT[svI] = largestAlpha;
//                         }
//                     }
//                 }
//             }

//             // PP
// #ifdef USE_SH_LFSS
//             //NOTE: results may differ when computing step size with large eta as long-distance pairs are dropped
//             for (const auto& nJ : sPointInds) {
//                 int vJ = boundaryNode[nJ];
// #else
//             for (const auto& vJ : boundaryNode) {
// #endif
//                 if (vI < vJ && !(DBCb[vI] && DBCb[vJ])) {
//                     const VECTOR<T, 3>& X1 = std::get<0>(X.Get_Unchecked(vJ));
//                     Eigen::Matrix<T, 3, 1> p1(X1.data), dp1(searchDir.data() + vJ * dim);

//                     if (!Point_Point_CCD_Broadphase(p, p1, dp, dp1, thickness)) {
//                         continue;
//                     }

//                     T largestAlpha = 1.0;
//                     if (Point_Point_CCD(p, p1, dp, dp1, 0.1, thickness, largestAlpha)) {
//                         if (largestAlphasPT[svI] > largestAlpha) {
//                             largestAlphasPT[svI] = largestAlpha;
//                         }
//                     }
//                 }
//             }

            // particle P - rod E and particle P
            if (svI >= codimBNStartInd[1]) {
                // PE
#ifdef USE_SH_LFSS
                for (const auto& eI : sEdgeInds)
#else
                for (int eI = 0; eI < boundaryEdge.size(); ++eI)
#endif
                {
                    if (eI >= boundaryEdge.size() - rod.size()) {
                        const VECTOR<int, 2>& meshEI = boundaryEdge[eI];
                        if (!(vI == meshEI[0] || vI == meshEI[1]) &&
                            !(DBCb[vI] && DBCb[meshEI[0]] && DBCb[meshEI[1]])) 
                        {
                            if constexpr (shell) {
                                int vpair = (vI % 2 == 0) ? (vI + 1) : vI - 1;
                                if (vpair == meshEI[0] || vpair == meshEI[1]) {
                                    continue;
                                }
                            }
                            const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(meshEI[0]));
                            const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(meshEI[1]));
                            Eigen::Matrix<T, 3, 1> e0(Xe0.data), e1(Xe1.data);
                            Eigen::Matrix<T, 3, 1> de0(searchDir.data() + meshEI[0] * dim), 
                                de1(searchDir.data() + meshEI[1] * dim);

                            if (!Point_Edge_CCD_Broadphase(p, e0, e1, dp, de0, de1, thickness)) {
                                continue;
                            }

                            T largestAlpha = largestAlphasPT[svI];
                            if (Point_Edge_CCD(p, e0, e1, dp, de0, de1, 0.1, thickness, largestAlpha)) {
                                if (largestAlphasPT[svI] > largestAlpha) {
                                    largestAlphasPT[svI] = largestAlpha;
                                }
                            }
                        }
                    }
                }

                // PP
#ifdef USE_SH_LFSS
                for (const auto& svJ : sPointInds)
#else
                for (int svJ = 0; svJ < boundaryNode.size(); ++svJ)
#endif
                {
                    if (svJ > svI) { // svJ is particle point and this won't double count PP CCD
                        int vJ = boundaryNode[svJ];
                        if (!(DBCb[vI] && DBCb[vJ])) {
                            const VECTOR<T, 3>& XpJ = std::get<0>(X.Get_Unchecked(vJ));
                            Eigen::Matrix<T, 3, 1> pJ(XpJ.data);
                            Eigen::Matrix<T, 3, 1> dpJ(searchDir.data() + vJ * dim);

                            if (!Point_Point_CCD_Broadphase(p, pJ, dp, dpJ, thickness)) {
                                continue;
                            }

                            T largestAlpha = largestAlphasPT[svI];
                            if (Point_Point_CCD(p, pJ, dp, dpJ, 0.1, thickness, largestAlpha)) {
                                if (largestAlphasPT[svI] > largestAlpha) {
                                    largestAlphasPT[svI] = largestAlpha;
                                }
                            }
                        }
                    }
                }
            }
        });
        stepSize = std::min(stepSize, *std::min_element(largestAlphasPT.begin(), largestAlphasPT.end()));
        } //TIMER_FLAG

        // edge-edge
        { TIMER_FLAG("Compute_Intersection_Free_StepSize_EE");
        BASE_STORAGE<int> threadsEE(boundaryEdge.size());
        for (int i = 0; i < boundaryEdge.size(); ++i) {
            threadsEE.Append(i);
        }

        std::vector<T> largestAlphasEE(boundaryEdge.size());
        threadsEE.Par_Each([&](int eI, auto data) {
            const VECTOR<int, 2>& meshEI = boundaryEdge[eI];
            const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(meshEI[0]));
            const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(meshEI[1]));
            Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data);
            Eigen::Matrix<T, 3, 1> dea0(searchDir.data() + meshEI[0] * dim), dea1(searchDir.data() + meshEI[1] * dim);
            largestAlphasEE[eI] = stepSize;
#ifdef USE_SH_LFSS
            std::unordered_set<int> sEdgeInds;
            sh.Query_Edge_For_Edges(eI, sEdgeInds);
            //NOTE: results may differ when computing step size with large eta as long-distance pairs are dropped
            for (const auto& eJ : sEdgeInds)
#else
            for (int eJ = eI + 1; eJ < boundaryEdge.size(); ++eJ)
#endif
            {
                const VECTOR<int, 2>& meshEJ = boundaryEdge[eJ];
                if (!(meshEI[0] == meshEJ[0] || meshEI[0] == meshEJ[1] || meshEI[1] == meshEJ[0] || meshEI[1] == meshEJ[1] || eI > eJ) &&
                    !(DBCb[meshEI[0]] && DBCb[meshEI[1]] && DBCb[meshEJ[0]] && DBCb[meshEJ[1]]))
                {
                    auto NNEFinder0 = NNExclusion.find(meshEI[0]);
                    auto NNEFinder1 = NNExclusion.find(meshEI[1]);
                    if ((NNEFinder0 != NNExclusion.end() &&
                        (NNEFinder0->second.find(meshEJ[0]) != NNEFinder0->second.end() ||
                        NNEFinder0->second.find(meshEJ[1]) != NNEFinder0->second.end())) ||
                        (NNEFinder1 != NNExclusion.end() && 
                        (NNEFinder1->second.find(meshEJ[0]) != NNEFinder1->second.end() ||
                        NNEFinder1->second.find(meshEJ[1]) != NNEFinder1->second.end())))
                    {
                        continue;
                    }
                    if constexpr (shell) {
                        if (meshEI[0] % 2 == 1) {
                            if (meshEI[0] - 1 == meshEJ[0] || meshEI[0] - 1 == meshEJ[1]) { continue; }
                        }
                        else if (meshEI[0] % 2 == 0) {
                            if (meshEI[0] + 1 == meshEJ[0] || meshEI[0] + 1 == meshEJ[1]) { continue; }
                        }

                        if (meshEI[1] % 2 == 1) {
                            if (meshEI[1] - 1 == meshEJ[0] || meshEI[1] - 1 == meshEJ[1]) { continue; }
                        }
                        else if (meshEI[1] % 2 == 0) {
                            if (meshEI[1] + 1 == meshEJ[0] || meshEI[1] + 1 == meshEJ[1]) { continue; }
                        }
                    }
                    const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(meshEJ[0]));
                    const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(meshEJ[1]));
                    Eigen::Matrix<T, 3, 1> eb0(Xeb0.data), eb1(Xeb1.data);
                    Eigen::Matrix<T, 3, 1> deb0(searchDir.data() + meshEJ[0] * dim), deb1(searchDir.data() + meshEJ[1] * dim);

                    if (!Edge_Edge_CCD_Broadphase(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, thickness)) {
                        continue;
                    }

                    T largestAlpha = largestAlphasEE[eI];
                    if (Edge_Edge_CCD(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, 0.1, thickness, largestAlpha)) {
                        if (largestAlphasEE[eI] > largestAlpha) {
                            largestAlphasEE[eI] = largestAlpha;
                        }
                    }
                }
            }
        });
        stepSize = std::min(stepSize, *std::min_element(largestAlphasEE.begin(), largestAlphasEE.end()));
        } //TIMER_FLAG
    }
}

template <class T, int dim, bool elasticIPC = false>
void Compute_Min_Dist2(MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, dim + 1>>& constraintSet,
    T thickness, std::vector<T>& dist2, T& minDist2)
{
    TIMER_FLAG("Compute_Min_Dist");

    if (constraintSet.empty()) {
        return;
    }

    if constexpr (elasticIPC) {
        thickness = 0;
    }

    dist2.resize(constraintSet.size());
    if constexpr (dim == 2) {
        //TODO: parallelize
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            const VECTOR<int, 3>& cIVInd = constraintSet[cI];
            if (cIVInd[2] < 0) {
                // PP
                const VECTOR<T, 2>& Xp0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                const VECTOR<T, 2>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                Eigen::Matrix<T, 2, 1> p0(Xp0[0], Xp0[1]), p1(Xp1[0], Xp1[1]);
                Point_Point_Distance(p0, p1, dist2[cI]);
            }
            else {
                // PE
                const VECTOR<T, 2>& Xp = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                const VECTOR<T, 2>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                const VECTOR<T, 2>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                Eigen::Matrix<T, 2, 1> p(Xp[0], Xp[1]), e0(Xe0[0], Xe0[1]), e1(Xe1[0], Xe1[1]);
                Point_Edge_Distance(p, e0, e1, dist2[cI]);
            }

            if (dist2[cI] <= 0) {
                std::cout << "0 distance detected during barrier evaluation!" << std::endl;
                exit(-1);
            }
        }
    }
    else {
        //TODO: parallelize
        for (int cI = 0; cI < constraintSet.size(); ++cI) {
            const VECTOR<int, 4>& cIVInd = constraintSet[cI];
            assert(cIVInd[1] >= 0);
            if (cIVInd[0] >= 0) {
                // EE
                if (cIVInd[3] >= 0 && cIVInd[2] >= 0) {
                    // ++++ EE, no mollification
                    const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                    const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                    Eigen::Matrix<T, 3, 1> ea0(Xea0.data), ea1(Xea1.data), eb0(Xeb0.data), eb1(Xeb1.data);
                    
                    Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2[cI]);
                }
                else {
                    // EE, PE, or PP with mollification
                    std::array<int, 4> edgeVInd;
                    Eigen::Matrix<T, 3, 1> ea0, ea1, eb0, eb1;
                    if (cIVInd[3] >= 0) {
                        // ++-+ EE with mollification
                        edgeVInd = {cIVInd[0], cIVInd[1], -cIVInd[2] - 1, cIVInd[3]};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        ea0 = std::move(Eigen::Matrix<T, 3, 1>(Xea0.data));
                        ea1 = std::move(Eigen::Matrix<T, 3, 1>(Xea1.data));
                        eb0 = std::move(Eigen::Matrix<T, 3, 1>(Xeb0.data));
                        eb1 = std::move(Eigen::Matrix<T, 3, 1>(Xeb1.data));
                        
                        Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2[cI]);
                    }
                    else if (cIVInd[2] >= 0) {
                        // +++- PE with mollification, multiplicity 1
                        edgeVInd = {cIVInd[0], -cIVInd[3] - 1, cIVInd[1], cIVInd[2]};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        ea0 = std::move(Eigen::Matrix<T, 3, 1>(Xea0.data));
                        ea1 = std::move(Eigen::Matrix<T, 3, 1>(Xea1.data));
                        eb0 = std::move(Eigen::Matrix<T, 3, 1>(Xeb0.data));
                        eb1 = std::move(Eigen::Matrix<T, 3, 1>(Xeb1.data));

                        Point_Edge_Distance(ea0, eb0, eb1, dist2[cI]);
                    }
                    else {
                        // ++-- PP with mollification, multiplicity 1
                        edgeVInd = {cIVInd[0], -cIVInd[2] - 1, cIVInd[1], -cIVInd[3] - 1};
                        const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                        const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                        const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                        const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                        ea0 = std::move(Eigen::Matrix<T, 3, 1>(Xea0.data));
                        ea1 = std::move(Eigen::Matrix<T, 3, 1>(Xea1.data));
                        eb0 = std::move(Eigen::Matrix<T, 3, 1>(Xeb0.data));
                        eb1 = std::move(Eigen::Matrix<T, 3, 1>(Xeb1.data));

                        Point_Point_Distance(ea0, eb0, dist2[cI]);
                    }
                }
            }
            else {
                // PT, PE, and PP
                if (cIVInd[3] >= 0) {
                    // -+++ PT 
                    assert(cIVInd[2] >= 0);
                    const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xt0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xt1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    const VECTOR<T, 3>& Xt2 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                    Eigen::Matrix<T, 3, 1> p(Xp.data), t0(Xt0.data), t1(Xt1.data), t2(Xt2.data);
                    
                    Point_Triangle_Distance(p, t0, t1, t2, dist2[cI]);
                }
                else if (cIVInd[2] >= 0) {
                    // -++[-] PE, last digit stores muliplicity
                    const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                    Eigen::Matrix<T, 3, 1> p(Xp.data), e0(Xe0.data), e1(Xe1.data);
                    
                    Point_Edge_Distance(p, e0, e1, dist2[cI]);
                }
                else {
                    // -+-[-] PP, last digit stores muliplicity
                    const VECTOR<T, 3>& Xp0 = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                    const VECTOR<T, 3>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                    Eigen::Matrix<T, 3, 1> p0(Xp0.data), p1(Xp1.data);
                    
                    Point_Point_Distance(p0, p1, dist2[cI]);
                }
            }
        }
    }
    minDist2 = *std::min_element(dist2.begin(), dist2.end());
    minDist2 -= thickness * thickness;
}

bool segTriIntersect(const Eigen::RowVector3d& ve0, const Eigen::RowVector3d& ve1,
    const Eigen::RowVector3d& vt0, const Eigen::RowVector3d& vt1, const Eigen::RowVector3d& vt2)
{
    Eigen::Matrix3d coefMtr;
    coefMtr.col(0) = vt1 - vt0;
    coefMtr.col(1) = vt2 - vt0;
    coefMtr.col(2) = ve0 - ve1;

#ifdef USE_PREDICATES
    igl::predicates::exactinit();
    const auto ori1 = igl::predicates::orient3d(vt0, vt1, vt2, ve0);
    const auto ori2 = igl::predicates::orient3d(vt0, vt1, vt2, ve1);
    if (ori1 == igl::predicates::Orientation::COPLANAR || ori2 == igl::predicates::Orientation::COPLANAR) {
        // coplanar, we can detect it by d(EE)=0 or d(PT)=0
        return false;
    }

    if (ori1 == ori2) {
        // edge is on one side of the plane that triangle is in
        return false;
    }
#else
    Eigen::RowVector3d n = coefMtr.col(0).cross(coefMtr.col(1));
    if (n.dot(ve0 - vt0) * n.dot(ve1 - vt0) > 0.0) {
        return false; // edge is on one side of the plane that triangle is in
    }

    if (coefMtr.determinant() == 0.0) {
        return false; // coplanar, we can detect it by d(EE)=0 or d(PT)=0
    }
#endif

#ifdef USE_PREDICATES
    // int res = eccd::segment_triangle_inter(ve0, ve1, vt0, vt1, vt2);
    // if(res == 1){
    // std::cout << ve0 << std::endl;
    // std::cout << ve1 << std::endl;
    // std::cout << vt0 << std::endl;
    // std::cout << vt1 << std::endl;
    // std::cout << vt2 << std::endl;
    // exit(0);
    // }
    // return res == 1 ? true : false;
#endif
    Eigen::Vector3d uvt = coefMtr.fullPivLu().solve((ve0 - vt0).transpose());
    if (uvt[0] >= 0.0 && uvt[1] >= 0.0 && uvt[0] + uvt[1] <= 1.0 && uvt[2] >= 0.0 && uvt[2] <= 1.0) {
        return true;
    }
    else {
        return false;
    }
}

template <class T, int dim>
bool Check_Edge_Tri_Intersect(
    MESH_NODE<T, dim>& X,
    const std::vector<int>& boundaryNode, // tet surf and tri nodes, seg nodes, rod nodes, particle nodes
    const std::vector<VECTOR<int, 2>>& boundaryEdge, // tet surf and tri edges, seg, rod
    const std::vector<VECTOR<int, 3>>& boundaryTri, // tet surf tris, tris
    const std::vector<int>& particle,
    const std::vector<VECTOR<int, 2>>& rod,
    const VECTOR<int, 2>& codimBNStartInd,
    const std::vector<bool>& DBCb) // weight, dHat2
{
    TIMER_FLAG("Check_Edge_Tri_Intersect");

#define USE_SH_CCS
#ifdef USE_SH_CCS
    SPATIAL_HASH<T, dim> sh;
    {
        TIMER_FLAG("Check_Edge_Tri_Intersect_Build_Hash");
        sh.Build(X, boundaryNode, boundaryEdge, boundaryTri, 1.0);
    }
#endif

    if constexpr (dim == 2) {
    }
    else {
        BASE_STORAGE<int> threadsT(boundaryTri.size());
        for (int i = 0; i < boundaryTri.size(); ++i) {
            threadsT.Append(i);
        }

        std::vector<std::vector<int>> intersectedEdges(boundaryTri.size());
        threadsT.Par_Each([&](int sfI, auto data) {
            const VECTOR<int, 3>& sfVInd = boundaryTri[sfI];
            Eigen::Matrix<T, dim, 1> t0(std::get<0>(X.Get_Unchecked(sfVInd[0])).data);
            Eigen::Matrix<T, dim, 1> t1(std::get<0>(X.Get_Unchecked(sfVInd[1])).data);
            Eigen::Matrix<T, dim, 1> t2(std::get<0>(X.Get_Unchecked(sfVInd[2])).data);
#ifdef USE_SH_CCS
            std::unordered_set<int> sEdgeInds;
            sh.Query_Triangle_For_Edges(t0, t1, t2, 0.0, sEdgeInds);
            for (const auto& eI : sEdgeInds)
#else
            for (int eI = 0; eI < boundaryEdge.size(); ++eI)
#endif
            {
                const VECTOR<int, 2>& meshEI = boundaryEdge[eI];
                if (meshEI[0] == sfVInd[0] || meshEI[0] == sfVInd[1] || meshEI[0] == sfVInd[2] || 
                    meshEI[1] == sfVInd[0] || meshEI[1] == sfVInd[1] || meshEI[1] == sfVInd[2] ||
                    (DBCb[meshEI[0]] && DBCb[meshEI[1]] && DBCb[sfVInd[0]] && DBCb[sfVInd[1]] && DBCb[sfVInd[2]])) {
                    continue;
                }
                Eigen::Matrix<T, dim, 1> e0(std::get<0>(X.Get_Unchecked(meshEI[0])).data);
                Eigen::Matrix<T, dim, 1> e1(std::get<0>(X.Get_Unchecked(meshEI[1])).data);
                if (segTriIntersect(e0, e1, t0, t1, t2)) {
                    intersectedEdges[sfI].emplace_back(eI);
                }
            }
        });

        bool intersected = false;
        for (int sfI = 0; sfI < intersectedEdges.size(); ++sfI) {
            for (const auto& eI : intersectedEdges[sfI]) {
                intersected = true;
                std::cout << "self edge - triangle intersection detected" << std::endl;
                std::cout << boundaryEdge[eI][0] << " " << boundaryEdge[eI][1] << std::endl;
                std::cout << boundaryTri[sfI][0] << " " << boundaryTri[sfI][1] << " " << boundaryTri[sfI][2] << std::endl;
                std::cout << Eigen::RowVector3d(std::get<0>(X.Get_Unchecked(boundaryEdge[eI][0])).data) << std::endl;
                std::cout << Eigen::RowVector3d(std::get<0>(X.Get_Unchecked(boundaryEdge[eI][1])).data) << std::endl;
                std::cout << Eigen::RowVector3d(std::get<0>(X.Get_Unchecked(boundaryTri[sfI][0])).data) << std::endl;
                std::cout << Eigen::RowVector3d(std::get<0>(X.Get_Unchecked(boundaryTri[sfI][1])).data) << std::endl;
                std::cout << Eigen::RowVector3d(std::get<0>(X.Get_Unchecked(boundaryTri[sfI][2])).data) << std::endl;
            }
        }
        return !intersected;
    }
}

template <class T>
void Point_Triangle_Distance_Vector_Unclassified(
    const Eigen::Matrix<T, 3, 1>& p, 
    const Eigen::Matrix<T, 3, 1>& t0, 
    const Eigen::Matrix<T, 3, 1>& t1,
    const Eigen::Matrix<T, 3, 1>& t2,
    const Eigen::Matrix<T, 3, 1>& dp, 
    const Eigen::Matrix<T, 3, 1>& dt0, 
    const Eigen::Matrix<T, 3, 1>& dt1,
    const Eigen::Matrix<T, 3, 1>& dt2,
    T t, T lambda, T beta,
    Eigen::Matrix<T, 3, 1>& distVec)
{
    const Eigen::Matrix<T, 3, 1> tp = (1 - lambda - beta) * t0 + lambda * t1 + beta * t2;
    const Eigen::Matrix<T, 3, 1> dtp = (1 - lambda - beta) * dt0 + lambda * dt1 + beta * dt2;
    distVec = p + t * dp - (tp + t * dtp);
}

template <class T>
bool Point_Triangle_CheckInterval_Unclassified(
    const Eigen::Matrix<T, 3, 1>& p, 
    const Eigen::Matrix<T, 3, 1>& t0, 
    const Eigen::Matrix<T, 3, 1>& t1,
    const Eigen::Matrix<T, 3, 1>& t2,
    const Eigen::Matrix<T, 3, 1>& dp, 
    const Eigen::Matrix<T, 3, 1>& dt0, 
    const Eigen::Matrix<T, 3, 1>& dt1,
    const Eigen::Matrix<T, 3, 1>& dt2,
    const std::array<T, 6>& interval,
    T gap)
{
    Eigen::Matrix<T, 3, 1> distVecMax, distVecMin;
    distVecMax.setConstant(-2 * gap - 1);
    distVecMin.setConstant(2 * gap + 1);
    for (int t = 0; t < 2; ++t) {
        for (int lambda = 0; lambda < 2; ++lambda) {
            for (int beta = 0; beta < 2; ++beta) {
                if (lambda == 1 && beta == 1) {
                    continue;
                }
                Eigen::Matrix<T, 3, 1> distVec;
                Point_Triangle_Distance_Vector_Unclassified(p, t0, t1, t2, dp, dt0, dt1, dt2, 
                    interval[t], interval[2 + lambda], interval[4 + beta], distVec);
                distVecMax = distVecMax.array().max(distVec.array());
                distVecMin = distVecMin.array().min(distVec.array());
            }
        }
    }
    return (distVecMax.array() >= -gap).all() && (distVecMin.array() <= gap).all();
}

template <class T>
bool Point_Triangle_TICCD(
    const Eigen::Matrix<T, 3, 1>& p, 
    const Eigen::Matrix<T, 3, 1>& t0, 
    const Eigen::Matrix<T, 3, 1>& t1,
    const Eigen::Matrix<T, 3, 1>& t2,
    Eigen::Matrix<T, 3, 1> dp, 
    Eigen::Matrix<T, 3, 1> dt0, 
    Eigen::Matrix<T, 3, 1> dt1,
    Eigen::Matrix<T, 3, 1> dt2,
    T eta, T thickness, T& toc)
{
    // can run ACCD in the same loop, and compensate each other on worst case

    // Eigen::Matrix<T, 3, 1> mov = (dt0 + dt1 + dt2 + dp) / 4;
    // dt0 -= mov;
    // dt1 -= mov;
    // dt2 -= mov;
    // dp -= mov;
    // std::vector<T> dispMag2Vec{dt0.squaredNorm(), dt1.squaredNorm(), dt2.squaredNorm()};
    // T maxDispMag = dp.norm() + std::sqrt(*std::max_element(dispMag2Vec.begin(), dispMag2Vec.end()));
    // if (maxDispMag == 0) {
    //     return false;
    // }

    T dist2_cur;
    Point_Triangle_Distance_Unclassified(p, t0, t1, t2, dist2_cur);
    T dist_cur = std::sqrt(dist2_cur);
    T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);

    T tTol = 1e-3;

    std::vector<std::array<T, 6>> roots;
    std::deque<std::array<T, 6>> intervals;
    intervals.push_back({0, toc, 0, 1, 0, 1});
    int iterAmt = 0;
    while (!intervals.empty()) {
        ++iterAmt;

        std::array<T, 6> curIV = intervals.front();
        intervals.pop_front();

        // Point_Triangle_Distance_Unclassified<T>(p + curIV[0] * dp, 
        //     t0 + curIV[0] * dt0, t1 + curIV[0] * dt1, t2 + curIV[0] * dt2, dist2_cur);
        // dist_cur = std::sqrt(dist2_cur);
        // T dist2target = (dist2_cur - thickness * thickness) / (dist_cur + thickness) - gap;
        // if (dist2target > 0) {
        //     T tocLowerBound = dist2target / maxDispMag;
        //     if (curIV[0] + tocLowerBound >= curIV[1]) {
        //         continue;
        //     }
        // }

        // Point_Triangle_Distance_Unclassified<T>(p + curIV[1] * dp, 
        //     t0 + curIV[1] * dt0, t1 + curIV[1] * dt1, t2 + curIV[1] * dt2, dist2_cur);
        // dist_cur = std::sqrt(dist2_cur);
        // dist2target = (dist2_cur - thickness * thickness) / (dist_cur + thickness) - gap;
        // if (dist2target > 0) {
        //     T tocLowerBound = dist2target / maxDispMag;
        //     if (curIV[0] >= curIV[1] - tocLowerBound) {
        //         continue;
        //     }
        // }

        // check
        if (Point_Triangle_CheckInterval_Unclassified(p, t0, t1, t2, dp, dt0, dt1, dt2, curIV, gap)) {
            if (curIV[0] && curIV[1] - curIV[0] < tTol) {
                // root found within tTol
                roots.emplace_back(curIV);
            }
            else {
                // split interval and push back
                std::vector<T> intervalLen({curIV[1] - curIV[0], curIV[3] - curIV[2], curIV[5] - curIV[4]});
                switch (std::max_element(intervalLen.begin(), intervalLen.end()) - intervalLen.begin()) {
                case 0:
                    intervals.push_back({curIV[0], (curIV[1] + curIV[0]) / 2, curIV[2], curIV[3], curIV[4], curIV[5]});
                    intervals.push_back({(curIV[1] + curIV[0]) / 2, curIV[1], curIV[2], curIV[3], curIV[4], curIV[5]});
                    break;

                case 1:
                    intervals.push_back({curIV[0], curIV[1], curIV[2], (curIV[2] + curIV[3]) / 2, curIV[4], curIV[5]});
                    intervals.push_back({curIV[0], curIV[1], (curIV[2] + curIV[3]) / 2, curIV[3], curIV[4], curIV[5]});
                    break;

                case 2:
                    intervals.push_back({curIV[0], curIV[1], curIV[2], curIV[3], curIV[4], (curIV[4] + curIV[5]) / 2});
                    intervals.push_back({curIV[0], curIV[1], curIV[2], curIV[3], (curIV[4] + curIV[5]) / 2, curIV[5]});
                    break;
                }
            }
        }
    }
    
    if (roots.empty()) {
        printf("TICCD PT converged with %d iters\n", iterAmt);
        return false;
    }
    else {
        for (const auto& rI : roots) {
            if (toc > rI[0]) {
                toc = rI[0];
            }
        }
        printf("TICCD PT converged with %d iters\n", iterAmt);
        return true;
    }
}

template <class T>
bool Point_Triangle_ACCD(
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
    Eigen::Matrix<T, 3, 1> mov = (dt0 + dt1 + dt2 + dp) / 4;
    dt0 -= mov;
    dt1 -= mov;
    dt2 -= mov;
    dp -= mov;
    std::vector<T> dispMag2Vec{dt0.squaredNorm(), dt1.squaredNorm(), dt2.squaredNorm()};
    T maxDispMag = dp.norm() + std::sqrt(*std::max_element(dispMag2Vec.begin(), dispMag2Vec.end()));
    if (maxDispMag == 0) {
        return false;
    }

    T dist2_cur;
    Point_Triangle_Distance_Unclassified(p, t0, t1, t2, dist2_cur);
    T dist_cur = std::sqrt(dist2_cur);
    T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    T toc_prev = toc;
    toc = 0;
    int iterAmt = 0;
    while (true) {
        ++iterAmt;
        T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness) / ((dist_cur + thickness) * maxDispMag);
        // T tcheck_ub = (toc_prev - toc) / 2;
        // if (!Point_Triangle_CheckInterval_Unclassified(p, t0, t1, t2, dp, dt0, dt1, dt2, {0, tcheck_ub, 0, 1, 0, 1}, gap)) {
        //     printf("helped? %le %le\n", tocLowerBound, tcheck_ub);
        //     tocLowerBound = std::max(tocLowerBound, tcheck_ub);
        // }

        p += tocLowerBound * dp;
        t0 += tocLowerBound * dt0;
        t1 += tocLowerBound * dt1;
        t2 += tocLowerBound * dt2;
        Point_Triangle_Distance_Unclassified(p, t0, t1, t2, dist2_cur);
        dist_cur = std::sqrt(dist2_cur);
        if (toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)) {
            break;
        }
        
        toc += tocLowerBound;
        if (toc > toc_prev) {
            printf("ACCD PT converged with %d iters\n", iterAmt);
            return false;
        }
    }

    printf("ACCD PT converged with %d iters\n", iterAmt);
    return true;
}

}

