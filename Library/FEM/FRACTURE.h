#pragma once

#include <Math/VECTOR.h>
#include <Utils/MESHIO.h>
#include <Math/UTILS.h>

namespace py = pybind11;
namespace JGSL {

bool findTetByTri(std::map<VECTOR<int, 3>, int>& tri2Tet,
    const VECTOR<int, 3>& triVInd,
    std::map<VECTOR<int, 3>, int>::iterator& finder)
{
    finder = tri2Tet.find(triVInd);
    if (finder == tri2Tet.end()) {
        finder = tri2Tet.find(VECTOR<int, 3>(triVInd[1], triVInd[2], triVInd[0]));
        if (finder == tri2Tet.end()) {
            finder = tri2Tet.find(VECTOR<int, 3>(triVInd[2], triVInd[0], triVInd[1]));
            if (finder == tri2Tet.end()) {
                return false;
            }
        }
    }
    return true;
}

int faceIDByVInd(const std::map<VECTOR<int, 3>, int> faceID,
    const VECTOR<int, 3>& triVInd)
{
    auto finder = faceID.find(triVInd);
    if (finder == faceID.end()) {
        finder = faceID.find(VECTOR<int, 3>(triVInd[1], triVInd[2], triVInd[0]));
        if (finder == faceID.end()) {
            finder = faceID.find(VECTOR<int, 3>(triVInd[2], triVInd[0], triVInd[1]));
            if (finder == faceID.end()) {
                std::cout << "face not found!" << std::endl;
                exit(-1);
            }
        }
    }
    return finder->second;
}

template <class T, int dim>
void Initialize_Fracture(
    MESH_NODE<T, dim>& X, MESH_ELEM<dim>& Elem,
    int debrisAmt, T fractureRatio2,
    T strengthenFactor,
    std::vector<std::array<int, dim * 2>>& edge_dupV,
    std::vector<int>& isFracture_edge,
    std::vector<VECTOR<int, 4>>& incTriV_edge,
    std::vector<T>& incTriRestDist2_edge)
{
    TIMER_FLAG("Initialize_Fracture");

    std::map<VECTOR<int, dim>, int> edge2tri;
    std::vector<VECTOR<int, dim>> edge;
    std::map<VECTOR<int, dim>, int> edgeID;
    Elem.Each([&](int id, auto data) {
        auto &[triVInd] = data;
        if constexpr (dim == 2) {
            edge2tri[VECTOR<int, dim>(triVInd[0], triVInd[1])] = id;
            edge2tri[VECTOR<int, dim>(triVInd[1], triVInd[2])] = id;
            edge2tri[VECTOR<int, dim>(triVInd[2], triVInd[0])] = id;

            auto finder = edge2tri.find(VECTOR<int, dim>(triVInd[1], triVInd[0]));
            if (finder == edge2tri.end()) {
                edgeID[VECTOR<int, dim>(triVInd[0], triVInd[1])] = edge.size();
                edgeID[VECTOR<int, dim>(triVInd[1], triVInd[0])] = edge.size();
                edge.emplace_back(triVInd[0], triVInd[1]);
            }
            finder = edge2tri.find(VECTOR<int, dim>(triVInd[2], triVInd[1]));
            if (finder == edge2tri.end()) {
                edgeID[VECTOR<int, dim>(triVInd[1], triVInd[2])] = edge.size();
                edgeID[VECTOR<int, dim>(triVInd[2], triVInd[1])] = edge.size();
                edge.emplace_back(triVInd[1], triVInd[2]);
            }
            finder = edge2tri.find(VECTOR<int, dim>(triVInd[0], triVInd[2]));
            if (finder == edge2tri.end()) {
                edgeID[VECTOR<int, dim>(triVInd[2], triVInd[0])] = edge.size();
                edgeID[VECTOR<int, dim>(triVInd[0], triVInd[2])] = edge.size();
                edge.emplace_back(triVInd[2], triVInd[0]);
            }
        }
        else {
            edge2tri[VECTOR<int, dim>(triVInd[0], triVInd[2], triVInd[1])] = id;
            edge2tri[VECTOR<int, dim>(triVInd[0], triVInd[3], triVInd[2])] = id;
            edge2tri[VECTOR<int, dim>(triVInd[0], triVInd[1], triVInd[3])] = id;
            edge2tri[VECTOR<int, dim>(triVInd[1], triVInd[2], triVInd[3])] = id;

            std::map<VECTOR<int, 3>, int>::iterator finder;
            if(!findTetByTri(edge2tri, VECTOR<int, dim>(triVInd[0], triVInd[1], triVInd[2]), finder)) {
                edgeID[VECTOR<int, dim>(triVInd[0], triVInd[2], triVInd[1])] = edge.size();
                edgeID[VECTOR<int, dim>(triVInd[0], triVInd[1], triVInd[2])] = edge.size();
                edge.emplace_back(triVInd[0], triVInd[2], triVInd[1]);
            }
            if(!findTetByTri(edge2tri, VECTOR<int, dim>(triVInd[0], triVInd[2], triVInd[3]), finder)) {
                edgeID[VECTOR<int, dim>(triVInd[0], triVInd[3], triVInd[2])] = edge.size();
                edgeID[VECTOR<int, dim>(triVInd[0], triVInd[2], triVInd[3])] = edge.size();
                edge.emplace_back(triVInd[0], triVInd[3], triVInd[2]);
            }
            if(!findTetByTri(edge2tri, VECTOR<int, dim>(triVInd[0], triVInd[3], triVInd[1]), finder)) {
                edgeID[VECTOR<int, dim>(triVInd[0], triVInd[1], triVInd[3])] = edge.size();
                edgeID[VECTOR<int, dim>(triVInd[0], triVInd[3], triVInd[1])] = edge.size();
                edge.emplace_back(triVInd[0], triVInd[1], triVInd[3]);
            }
            if(!findTetByTri(edge2tri, VECTOR<int, dim>(triVInd[1], triVInd[3], triVInd[2]), finder)) {
                edgeID[VECTOR<int, dim>(triVInd[1], triVInd[2], triVInd[3])] = edge.size();
                edgeID[VECTOR<int, dim>(triVInd[1], triVInd[3], triVInd[2])] = edge.size();
                edge.emplace_back(triVInd[1], triVInd[2], triVInd[3]);
            }
        }
    });

    edge_dupV.resize(edge.size());
    incTriV_edge.resize(edge.size());
    incTriRestDist2_edge.resize(edge.size());
    isFracture_edge.resize(0);
    isFracture_edge.resize(edge.size(), 0);
    //TODO: parallelize
    if constexpr (dim == 2) {
        for (int eI = 0; eI < edge.size(); ++eI) {
            const auto tri0 = edge2tri.find(edge[eI]);
            const auto tri1 = edge2tri.find(VECTOR<int, dim>(edge[eI][1], edge[eI][0]));

            const VECTOR<int, 3>& triVInd0 = std::get<0>(Elem.Get_Unchecked(tri0->second));
            int i = 0;
            for ( ; i < 3; ++i) {
                if (triVInd0[i] == edge[eI][0]) {
                    break;
                }
            }
            edge_dupV[eI][0] = tri0->second * 3 + i;
            edge_dupV[eI][1] = tri0->second * 3 + (i + 1) % 3;
            incTriV_edge[eI][0] = tri0->second;
            incTriV_edge[eI][1] = (i + 2) % 3;
            int v0I = triVInd0[(i + 2) % 3];

            if (tri1 == edge2tri.end()) {
                // boundary edge
                edge_dupV[eI][2] = edge_dupV[eI][3] = -1;
                isFracture_edge[eI] = 1;
                incTriV_edge[eI][0] = -1;
                incTriRestDist2_edge[eI] = -1;
            }
            else {
                // interior edge
                const VECTOR<int, 3>& triVInd1 = std::get<0>(Elem.Get_Unchecked(tri1->second));
                int i = 0;
                for ( ; i < 3; ++i) {
                    if (triVInd1[i] == edge[eI][1]) {
                        break;
                    }
                }
                edge_dupV[eI][3] = tri1->second * 3 + i;
                edge_dupV[eI][2] = tri1->second * 3 + (i + 1) % 3;
                incTriV_edge[eI][2] = tri1->second;
                incTriV_edge[eI][3] = (i + 2) % 3;
                int v1I = triVInd1[(i + 2) % 3];

                // dist2 = ((v0 + ve0 + ve1) / 3 - (v1 + ve0 + ve1) / 3).length2()
                const VECTOR<T, 2>& v0 = std::get<0>(X.Get_Unchecked(v0I));
                const VECTOR<T, 2>& v1 = std::get<0>(X.Get_Unchecked(v1I));
                incTriRestDist2_edge[eI] = (v0 - v1).length2() / 9;
            }
        }
    }
    else {
        for (int eI = 0; eI < edge.size(); ++eI) {
            std::map<VECTOR<int, 3>, int>::iterator tri0, tri1;
            findTetByTri(edge2tri, edge[eI], tri0);
            findTetByTri(edge2tri, VECTOR<int, 3>(edge[eI][0], edge[eI][2], edge[eI][1]), tri1);

            const VECTOR<int, 4>& triVInd0 = std::get<0>(Elem.Get_Unchecked(tri0->second));
            std::map<int, int> vInd2Local;
            for (int i = 0; i < 4; ++i) {
                vInd2Local[triVInd0[i]] = i;
            }
            for (int i = 0; i < 3; ++i) {
                edge_dupV[eI][i] = tri0->second * 4 + vInd2Local[edge[eI][i]];
                vInd2Local.erase(edge[eI][i]);
            }
            incTriV_edge[eI][0] = tri0->second;
            incTriV_edge[eI][1] = vInd2Local.begin()->second;
            int v0I = triVInd0[incTriV_edge[eI][1]];

            if (tri1 == edge2tri.end()) {
                // boundary edge
                edge_dupV[eI][3] = edge_dupV[eI][4] = edge_dupV[eI][5] = -1;
                isFracture_edge[eI] = 1;
                incTriV_edge[eI][0] = -1;
                incTriRestDist2_edge[eI] = -1;
            }
            else {
                // interior edge
                const VECTOR<int, 4>& triVInd1 = std::get<0>(Elem.Get_Unchecked(tri1->second));
                std::map<int, int> vInd2Local;
                for (int i = 0; i < 4; ++i) {
                    vInd2Local[triVInd1[i]] = i;
                }
                for (int i = 0; i < 3; ++i) {
                    edge_dupV[eI][dim + i] = tri1->second * 4 + vInd2Local[edge[eI][i]];
                    vInd2Local.erase(edge[eI][i]);
                }
                incTriV_edge[eI][2] = tri1->second;
                incTriV_edge[eI][3] = vInd2Local.begin()->second;
                int v1I = triVInd1[incTriV_edge[eI][3]];

                // dist2 = ((v0 + vt0 + vt1 + vt2) / 4 - (v1 + vt0 + vt1 + vt2) / 4).length2()
                const VECTOR<T, 3>& v0 = std::get<0>(X.Get_Unchecked(v0I));
                const VECTOR<T, 3>& v1 = std::get<0>(X.Get_Unchecked(v1I));
                incTriRestDist2_edge[eI] = (v0 - v1).length2() / 16;
            }
        }
    }

    if (debrisAmt > 1 && debrisAmt < Elem.size) {
        std::deque<VECTOR<int, 2>> toSpread;
        std::vector<int> elemLabel(Elem.size, -1);
        std::set<int> seedElem;
        while (toSpread.size() < debrisAmt) {
            int randElem = rand() % Elem.size;
            if (seedElem.insert(randElem).second) {
                elemLabel[randElem] = toSpread.size();
                toSpread.push_back(std::move(VECTOR<int, 2>(randElem, elemLabel[randElem])));
            }
        }
        std::vector<int> isFracturable(edge.size(), 0);
        while (!toSpread.empty()) {
            const VECTOR<int, 2> curElemI = toSpread.front();
            toSpread.pop_front();

            const VECTOR<int, dim + 1>& elemVInd = std::get<0>(Elem.Get_Unchecked(curElemI[0]));
            if constexpr (dim == 2) {
                for (int i = 0; i < 3; ++i) {
                    const auto nbFinder = edge2tri.find(VECTOR<int, dim>(elemVInd[(i + 1) % 3], elemVInd[i]));
                    if (nbFinder != edge2tri.end()) { 
                        if (elemLabel[nbFinder->second] == -1) {
                            elemLabel[nbFinder->second] = curElemI[1];
                            toSpread.push_back(std::move(VECTOR<int, 2>(nbFinder->second, curElemI[1])));
                        }
                        else if (elemLabel[nbFinder->second] != curElemI[1]) {
                            int eI = edgeID[VECTOR<int, dim>(elemVInd[(i + 1) % 3], elemVInd[i])];
                            isFracturable[eI] = 1;
                        }
                    }
                }
            }
            else {
                VECTOR<int, 3> faceVInd[4];
                faceVInd[0] = std::move(VECTOR<int, 3>(elemVInd[0], elemVInd[1], elemVInd[2]));
                faceVInd[1] = std::move(VECTOR<int, 3>(elemVInd[0], elemVInd[2], elemVInd[3]));
                faceVInd[2] = std::move(VECTOR<int, 3>(elemVInd[0], elemVInd[3], elemVInd[1]));
                faceVInd[3] = std::move(VECTOR<int, 3>(elemVInd[1], elemVInd[3], elemVInd[2]));
                for (int i = 0; i < 4; ++i) {
                    std::map<VECTOR<int, 3>, int>::iterator tetFinder;
                    findTetByTri(edge2tri, faceVInd[i], tetFinder);
                    if (tetFinder != edge2tri.end()) { 
                        if (elemLabel[tetFinder->second] == -1) {
                            elemLabel[tetFinder->second] = curElemI[1];
                            toSpread.push_back(std::move(VECTOR<int, 2>(tetFinder->second, curElemI[1])));
                        }
                        else if (elemLabel[tetFinder->second] != curElemI[1]) {
                            int eI = faceIDByVInd(edgeID, faceVInd[i]);
                            isFracturable[eI] = 1;
                        }
                    }
                }
            }
        }

        MESH_NODE<T, dim> X_vis;
        X.deep_copy_to(X_vis);
        MESH_ELEM<dim> Elem_vis;
        Elem.deep_copy_to(Elem_vis);
        std::vector<int> finalV2old;
        Node_Fracture(edge_dupV, isFracturable, X_vis, Elem_vis, finalV2old);
        if constexpr (dim == 2) {
            Write_TriMesh_Obj(X_vis, Elem_vis, "fracturable.obj");
        }
        else {
            BASE_STORAGE<int> TriVI2TetVI;
            BASE_STORAGE<VECTOR<int, 3>> Tri;
            Find_Surface_TriMesh(X_vis, Elem_vis, TriVI2TetVI, Tri);
            Write_Surface_TriMesh_Obj(X_vis, TriVI2TetVI, Tri, "fracturable.obj");
        }

        T allowedStretch = std::sqrt(fractureRatio2) - 1;
        T strengthen = std::pow(1 + strengthenFactor * allowedStretch, 2) / fractureRatio2;
        for (int eI = 0; eI < incTriRestDist2_edge.size(); ++eI) {
            if (!isFracturable[eI]) {
                incTriRestDist2_edge[eI] *= strengthen;
            }
        }
    }
}

template <class T, int dim>
bool Edge_Fracture(
    MESH_NODE<T, dim>& X, MESH_ELEM<dim>& Elem,
    const std::vector<VECTOR<int, 4>>& incTriV_edge,
    const std::vector<T>& incTriRestDist2_edge,
    T fractureRatio2,
    std::vector<int>& isFracture_edge)
{
    TIMER_FLAG("Edge_Fracture");

    //TODO: parallelize
    std::vector<T> lenRatio2(isFracture_edge.size(), 0);
    bool fractured = false;
    for (int eI = 0; eI < isFracture_edge.size(); ++eI) {
        if (isFracture_edge[eI] == 0 && incTriV_edge[eI][0] >= 0) {
            int v0I = std::get<0>(Elem.Get_Unchecked(incTriV_edge[eI][0]))[incTriV_edge[eI][1]];
            int v1I = std::get<0>(Elem.Get_Unchecked(incTriV_edge[eI][2]))[incTriV_edge[eI][3]];
            const VECTOR<T, dim>& v0 = std::get<0>(X.Get_Unchecked(v0I));
            const VECTOR<T, dim>& v1 = std::get<0>(X.Get_Unchecked(v1I));

            // curDist2 = ((v0 + ve0 + ve1) / 3 - (v1 + ve0 + ve1) / 3).length2()
            lenRatio2[eI] = (v0 - v1).length2() / std::pow(dim + 1, 2) / incTriRestDist2_edge[eI];
            if (lenRatio2[eI] > fractureRatio2) {
                fractured = true;
                isFracture_edge[eI] = 1;
            }
        }
    }
    return fractured;
}

template <class T, int dim>
void Node_Fracture(
    const std::vector<std::array<int, dim * 2>>& edge_dupV,
    const std::vector<int>& isFracture_edge,
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim>& Elem,
    std::vector<int>& finalV2old)
{
    TIMER_FLAG("Node_Fracture");
    //TODO: does vertex order matter?

    std::vector<int> dupV2new(Elem.size * (dim + 1), -1);
    std::vector<std::vector<int>> newV2dup;
    for (int eI = 0; eI < isFracture_edge.size(); ++eI) {
        if (edge_dupV[eI][dim] < 0) {
            // boundary edge
            for (int i = 0; i < dim; ++i) {
                if (dupV2new[edge_dupV[eI][i]] == -1) {
                    dupV2new[edge_dupV[eI][i]] = newV2dup.size();
                    newV2dup.emplace_back(std::move(std::vector<int>{edge_dupV[eI][i]}));
                }
            }
        }
        else {
            // interior edge
            if (isFracture_edge[eI]) {
                for (int i = 0; i < dim * 2; ++i) {
                    if (dupV2new[edge_dupV[eI][i]] == -1) {
                        dupV2new[edge_dupV[eI][i]] = newV2dup.size();
                        newV2dup.emplace_back(std::move(std::vector<int>{edge_dupV[eI][i]}));
                    }
                }
            }
            else {
                // merge
                for (int endI = 0; endI < dim; ++endI) {
                    int dupV0I = edge_dupV[eI][endI];
                    int dupV1I = edge_dupV[eI][dim + endI];
                    if (dupV2new[dupV0I] == -1) {
                        if (dupV2new[dupV1I] == -1) {
                            dupV2new[dupV0I] = dupV2new[dupV1I] = newV2dup.size();
                            newV2dup.emplace_back(std::move(std::vector<int>{dupV0I, dupV1I}));
                        }
                        else {
                            dupV2new[dupV0I] = dupV2new[dupV1I];
                            newV2dup[dupV2new[dupV1I]].emplace_back(dupV0I);
                        }
                    }
                    else {
                        if (dupV2new[dupV1I] == -1) {
                            dupV2new[dupV1I] = dupV2new[dupV0I];
                            newV2dup[dupV2new[dupV0I]].emplace_back(dupV1I);
                        }
                        else if (dupV2new[dupV0I] != dupV2new[dupV1I]) {
                            const int newV1I = dupV2new[dupV1I];
                            for (const auto& dupVI : newV2dup[newV1I]) {
                                dupV2new[dupVI] = dupV2new[dupV0I];
                            }
                            newV2dup[dupV2new[dupV0I]].insert(newV2dup[dupV2new[dupV0I]].end(),
                                newV2dup[newV1I].begin(), newV2dup[newV1I].end());
                            newV2dup[newV1I].resize(0);
                        }
                    }
                }
            }
        }
    }

    std::vector<int> newV2final(newV2dup.size());
    int curFinalVI = 0;
    for (int newVI = 0; newVI < newV2dup.size(); ++newVI) {
        if (newV2dup[newVI].empty()) {
            newV2final[newVI] = -1;
        }
        else {
            newV2final[newVI] = curFinalVI++;
        }
    }

    MESH_NODE<T, dim> X_final(curFinalVI);
    finalV2old.resize(0);
    finalV2old.reserve(curFinalVI);
    for (int newVI = 0; newVI < newV2dup.size(); ++newVI) {
        if (!newV2dup[newVI].empty()) {
            int dupVI = newV2dup[newVI][0]; // any source/old vertex will work
            int oldVI = std::get<0>(Elem.Get_Unchecked(dupVI / (dim + 1)))[dupVI % (dim + 1)];
            X_final.Append(std::get<0>(X.Get_Unchecked(oldVI)));
            finalV2old.emplace_back(oldVI);
        }
    }
    X_final.deep_copy_to(X);

    Elem.Par_Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        for (int i = 0; i < dim + 1; ++i) {
            int dupVI = id * (dim + 1) + i;
            elemVInd[i] = newV2final[dupV2new[dupVI]];
        }
    });
}

template <class T, int dim>
void Update_Fracture(
    MESH_ELEM<dim>& Elem, T rho0,
    const std::vector<int>& finalV2old,
    MESH_NODE<T, dim>& X0,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    DBC_MOTION<T, dim>& DBCMotion,
    bool withCollision, T dHat2,
    MESH_NODE<T, dim>& X)
{
    TIMER_FLAG("Update_Fracture");

    MESH_NODE<T, dim> newX0(finalV2old.size());
    MESH_NODE_ATTR<T, dim> newNodeAttr(finalV2old.size());
    std::vector<std::vector<int>> oldV2final(X0.size);
    for (int i = 0; i < finalV2old.size(); ++i) {
        newX0.Insert(i, std::get<0>(X0.Get_Unchecked(finalV2old[i])));
        const auto& [x0, v, g, m] = nodeAttr.Get_Unchecked(finalV2old[i]);
        newNodeAttr.Insert(i, x0, v, g, T(0));
        oldV2final[finalV2old[i]].emplace_back(i);
    }
    newX0.deep_copy_to(X0);
    newNodeAttr.deep_copy_to(nodeAttr);

    // recompute mass
    Elem.Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        const VECTOR<T, dim>& V0 = std::get<0>(X0.Get_Unchecked(elemVInd[0]));
        const VECTOR<T, dim>& V1 = std::get<0>(X0.Get_Unchecked(elemVInd[1]));
        const VECTOR<T, dim>& V2 = std::get<0>(X0.Get_Unchecked(elemVInd[2]));
        const VECTOR<T, dim> E01 = V1 - V0;
        const VECTOR<T, dim> E02 = V2 - V0;
        T massPortion;
        if constexpr (dim == 2) {
            massPortion = (E01[0] * E02[1] - E01[1] * E02[0]) / 2 * rho0 / 3;
        }
        else {
            const VECTOR<T, dim>& V3 = std::get<0>(X0.Get_Unchecked(elemVInd[3]));
            const VECTOR<T, dim> E03 = V3 - V0;
            massPortion = E01.dot(cross(E02, E03)) / 6 * rho0 / 4;
            std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(elemVInd[3])) += massPortion;
        }
        std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(elemVInd[0])) += massPortion;
        std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(elemVInd[1])) += massPortion;
        std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(elemVInd[2])) += massPortion;
    });

    VECTOR_STORAGE<T, dim + 1> newDBC(DBC.size);
    DBCMotion.Each([&](int id, auto data) {
        auto &[range, v, rotCenter, rotAxis, angVelDeg] = data;
        int newRange0 = newDBC.size;
        for (int i = range[0]; i < range[1]; ++i) {
            VECTOR<T, dim + 1> dbcI = std::get<0>(DBC.Get_Unchecked(i));
            const auto& finalVInd = oldV2final[dbcI[0]];
            for (const auto& i : finalVInd) {
                dbcI[0] = i;
                newDBC.Append(dbcI);
            }
        }
        range[0] = newRange0;
        range[1] = newDBC.size;
    });
    newDBC.deep_copy_to(DBC);

    if (withCollision) {
        std::vector<std::vector<T>> vIncELen2(X.size);
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            for (int i = 0; i < dim + 1; ++i) {
                for (int j = i + 1; j < dim + 1; ++j) {
                    const int vaI = elemVInd[i];
                    const int vbI = elemVInd[j];
                    const VECTOR<T, dim>& va = std::get<0>(X.Get_Unchecked(vaI));
                    const VECTOR<T, dim>& vb = std::get<0>(X.Get_Unchecked(vbI));
                    const T eLen2 = (va - vb).length2();
                    vIncELen2[vaI].emplace_back(eLen2);
                    vIncELen2[vbI].emplace_back(eLen2);
                }
            }
        });

        //TODO: avoid redundant contact primitive computation
        std::vector<int> boundaryNode;
        std::vector<VECTOR<int, 2>> boundaryEdge;
        std::vector<VECTOR<int, 3>> boundaryTri;
        if constexpr (dim == 2) {
            Find_Boundary_Edge_And_Node(X.size, Elem, boundaryNode, boundaryEdge);
        }
        else {
            BASE_STORAGE<int> TriVI2TetVI;
            BASE_STORAGE<VECTOR<int, 3>> Tri;
            Find_Surface_TriMesh<T, false>(X, Elem, TriVI2TetVI, Tri);
            Find_Surface_Primitives(X.size, Tri, boundaryNode, boundaryEdge, boundaryTri);
        }

        std::map<int, std::vector<int>> bNbE;
        if constexpr (dim == 2) {
            for (int beI = 0; beI < boundaryEdge.size(); ++beI) {
                bNbE[boundaryEdge[beI][0]].emplace_back(beI);
                bNbE[boundaryEdge[beI][1]].emplace_back(beI);
            }
        }
        else {
            for (int btI = 0; btI < boundaryTri.size(); ++btI) {
                bNbE[boundaryTri[btI][0]].emplace_back(btI);
                bNbE[boundaryTri[btI][1]].emplace_back(btI);
                bNbE[boundaryTri[btI][2]].emplace_back(btI);
            }
        }

        //TODO: parallelize
        std::set<int> movedV;
        for (const auto& oldVIfinalVerts : oldV2final) {
            if (oldVIfinalVerts.size() > 1) {
                // push inwards a little bit for each vertex split from an old vertex
                for (const int vI : oldVIfinalVerts) {
                    const auto finder = bNbE.find(vI);
                    if (finder == bNbE.end() || finder->second.size() < 2) {
                        std::cout << "boundary node record inconsistent!" << std::endl;
                        exit(-1);
                    }
                    
                    VECTOR<T, dim> movDir;
                    if constexpr (dim == 2) {
                        const VECTOR<T, dim>& e0v0 = std::get<0>(X.Get_Unchecked(boundaryEdge[finder->second[0]][0]));
                        const VECTOR<T, dim>& e0v1 = std::get<0>(X.Get_Unchecked(boundaryEdge[finder->second[0]][1]));
                        const VECTOR<T, dim>& e1v0 = std::get<0>(X.Get_Unchecked(boundaryEdge[finder->second[1]][0]));
                        const VECTOR<T, dim>& e1v1 = std::get<0>(X.Get_Unchecked(boundaryEdge[finder->second[1]][1]));

                        const VECTOR<T, dim> e0dir = (e0v1 - e0v0).Normalized();
                        const VECTOR<T, dim> e1dir = (e1v1 - e1v0).Normalized();
                        movDir = std::move(VECTOR<T, dim>(-e0dir[1] - e1dir[1], e0dir[0] + e1dir[0]));
                    }
                    else {
                        movDir.setZero();
                        for (const auto& tI : finder->second) {
                            const VECTOR<int, 3>& triVInd = boundaryTri[tI];
                            const VECTOR<T, 3>& v0 = std::get<0>(X.Get_Unchecked(triVInd[0]));
                            const VECTOR<T, 3>& v1 = std::get<0>(X.Get_Unchecked(triVInd[1]));
                            const VECTOR<T, 3>& v2 = std::get<0>(X.Get_Unchecked(triVInd[2]));
                            movDir -= cross(v1 - v0, v2 - v0); //TODO: normalize? intersection? edge-edge?
                        }
                    }
                    const T minELen = std::sqrt(*std::min_element(vIncELen2[vI].begin(), vIncELen2[vI].end()));
                    const T pushLen = std::min(0.1 * std::sqrt(dHat2), 1.0e-2 * minELen); //TODO: adapt with dHat
                    VECTOR<T, dim>& x = std::get<0>(X.Get_Unchecked(vI));
                    x += movDir / movDir.norm() * pushLen;
                    movedV.insert(vI);
                }
            }
        }

        DBC.Par_Each([&](int id, auto data) {
            auto &[dbcI] = data;
            const auto finder = movedV.find(int(dbcI[0]));
            if (finder != movedV.end()) {
                const VECTOR<T, dim>& x = std::get<0>(X.Get_Unchecked(int(dbcI[0])));
                dbcI[1] = x[0];
                dbcI[2] = x[1];
                if constexpr (dim == 3) {
                    dbcI[3] = x[2];
                }
            }
        });
    }
}

template <class T>
void Test_Fracture(void)
{
    {
        MESH_NODE<T, 2> X;
        MESH_ELEM<2> Elem;
        Read_TriMesh_Obj("sheet100.obj", X, Elem);
        Write_TriMesh_Obj(X, Elem, "before.obj");

        std::vector<std::array<int, 4>> edge_dupV;
        std::vector<int> isFracture_edge;
        std::vector<VECTOR<int, 4>> incTriV_edge;
        std::vector<T> incTriRestDist2_edge;
        Initialize_Fracture<T, 2>(X, Elem, -1, T(2), T(2), edge_dupV, isFracture_edge, 
            incTriV_edge, incTriRestDist2_edge);

        std::vector<int> finalV2old;
        Node_Fracture<T, 2>(edge_dupV, isFracture_edge, X, Elem, finalV2old);
        Write_TriMesh_Obj(X, Elem, "after_0.obj");

        isFracture_edge.resize(0);
        isFracture_edge.resize(edge_dupV.size(), 1);
        Node_Fracture<T, 2>(edge_dupV, isFracture_edge, X, Elem, finalV2old);
        Write_TriMesh_Obj(X, Elem, "after_1.obj");
    }

    {
        MESH_NODE<T, 3> X;
        MESH_ELEM<3> Elem;
        Read_TetMesh_Vtk("mat20x20.vtk", X, Elem);
        BASE_STORAGE<int> TriVI2TetVI;
        BASE_STORAGE<VECTOR<int, 3>> Tri;
        Find_Surface_TriMesh(X, Elem, TriVI2TetVI, Tri);
        Write_Surface_TriMesh_Obj(X, TriVI2TetVI, Tri, "3D_before.obj");

        std::vector<std::array<int, 6>> edge_dupV;
        std::vector<int> isFracture_edge;
        std::vector<VECTOR<int, 4>> incTriV_edge;
        std::vector<T> incTriRestDist2_edge;
        Initialize_Fracture<T, 3>(X, Elem, 2, T(2), T(2), edge_dupV, isFracture_edge, 
            incTriV_edge, incTriRestDist2_edge);

        std::vector<int> finalV2old;
        Node_Fracture<T, 3>(edge_dupV, isFracture_edge, X, Elem, finalV2old);
        Find_Surface_TriMesh(X, Elem, TriVI2TetVI, Tri);
        Write_Surface_TriMesh_Obj(X, TriVI2TetVI, Tri, "3D_after_0.obj");

        isFracture_edge.resize(0);
        isFracture_edge.resize(edge_dupV.size(), 1);
        Node_Fracture<T, 3>(edge_dupV, isFracture_edge, X, Elem, finalV2old);
        Find_Surface_TriMesh(X, Elem, TriVI2TetVI, Tri);
        Write_Surface_TriMesh_Obj(X, TriVI2TetVI, Tri, "3D_after_1.obj");
    }
}

// export python
void Export_Fracture(py::module& m) {
    m.def("Initialize_Fracture", &Initialize_Fracture<double, 2>, "initialize FEM fracture data structures");
    m.def("Initialize_Fracture", &Initialize_Fracture<double, 3>, "initialize FEM fracture data structures");
    m.def("Edge_Fracture", &Edge_Fracture<double, 2>, "decide which edge to fracture");
    m.def("Edge_Fracture", &Edge_Fracture<double, 3>, "decide which edge to fracture");
    m.def("Node_Fracture", &Node_Fracture<double, 2>, "decide which node to fracture according to edge fracture");
    m.def("Node_Fracture", &Node_Fracture<double, 3>, "decide which node to fracture according to edge fracture");
    m.def("Update_Fracture", &Update_Fracture<double, 2>, "initialize FEM fracture data structures");
    m.def("Update_Fracture", &Update_Fracture<double, 3>, "initialize FEM fracture data structures");
    m.def("Test_Fracture", &Test_Fracture<double>, "test FEM fracture functionality");
}

}