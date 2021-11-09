#pragma once

namespace JGSL {

template <class T, int dim>
void Compute_DBC_Dist2(
    MESH_NODE<T, dim>& X,
    VECTOR_STORAGE<T, dim + 1>& DBCBackup,
    T& dist2)
{
    std::vector<T> penalty(DBCBackup.size, T(0));
    DBCBackup.Par_Each([&](int id, auto data) {
        auto &[dbcI] = data;

        const VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
        penalty[id] += (dbcI(1) - x(0)) * (dbcI(1) - x(0));
        penalty[id] += (dbcI(2) - x(1)) * (dbcI(2) - x(1));
        if constexpr (dim == 3) {
            penalty[id] += (dbcI(3) - x(2)) * (dbcI(3) - x(2));
        }
    });
    dist2 = std::accumulate(penalty.begin(), penalty.end(), T(0));
}

template <class T, int dim>
void Compute_DBC_Energy(
    MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    T DBCStiff, T& E)
{
    std::vector<T> penalty(DBC.size, T(0));
    DBC.Par_Each([&](int id, auto data) {
        auto &[dbcI] = data;

        const VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
        penalty[id] += (dbcI(1) - x(0)) * (dbcI(1) - x(0));
        penalty[id] += (dbcI(2) - x(1)) * (dbcI(2) - x(1));
        if constexpr (dim == 3) {
            penalty[id] += (dbcI(3) - x(2)) * (dbcI(3) - x(2));
        }

        const T& m = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(dbcI(0)));
        penalty[id] *= m;
    });
    E += 0.5 * DBCStiff * std::accumulate(penalty.begin(), penalty.end(), T(0));
}

template <class T, int dim>
void Compute_DBC_Gradient(
    MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    T DBCStiff)
{
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;

        const VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
        const T& m = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(dbcI(0)));
        VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(dbcI(0)));

        g[0] += DBCStiff * m * (x(0) - dbcI(1));
        g[1] += DBCStiff * m * (x(1) - dbcI(2));
        if constexpr (dim == 3) {
            g[2] += DBCStiff * m * (x(2) - dbcI(3));
        }
    });
}

template <class T, int dim>
void Compute_DBC_Hessian(
    MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    T DBCStiff, std::vector<Eigen::Triplet<T>>& triplets)
{
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        int vI = dbcI(0);

        const VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(vI));
        const T& m = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(vI));

        int startInd = vI * dim;
        T entryVal = DBCStiff * m;
        triplets.emplace_back(startInd, startInd, entryVal);
        triplets.emplace_back(startInd + 1, startInd + 1, entryVal);
        if constexpr (dim == 3) {
            triplets.emplace_back(startInd + 2, startInd + 2, entryVal);
        }
    });
}

}