#pragma once

namespace JGSL {

template <class T, int dim = 3>
VECTOR<int, 4> Make_Rod(T len, int nSeg,
    MESH_NODE<T, dim>& X,
    std::vector<VECTOR<int, 2>>& rod)
{
    VECTOR<int, 4> counter;
    counter[0] = X.size;
    counter[1] = rod.size();
    
    X.Reserve(X.size + nSeg + 1);
    rod.reserve(rod.size() + nSeg);

    X.Append(VECTOR<T, dim>(-len / 2, 0, 0));
    const T segLen = len / nSeg;
    for (int i = 0; i < nSeg; ++i) {
        X.Append(VECTOR<T, dim>(-len / 2 + (i + 1) * segLen, 0, 0));
        rod.emplace_back(counter[0] + i, counter[0] + i + 1);
    }

    counter[2] = X.size;
    counter[3] = rod.size();
    return counter;
}

template <class T, int dim = 3>
VECTOR<int, 4> Make_Rod_Net(T len, int nSeg,
    int midPointAmt,
    MESH_NODE<T, dim>& X,
    std::vector<VECTOR<int, 2>>& rod)
{
    VECTOR<int, 4> counter;
    counter[0] = X.size;
    counter[1] = rod.size();
    
    X.Reserve(X.size + (nSeg + 1) * (nSeg + 1));
    rod.reserve(rod.size() + nSeg * (nSeg + 1) * 2);

    const T segLen = len / nSeg;
    for (int i = 0; i < nSeg + 1; ++i) {
        for (int j = 0; j < nSeg + 1; ++j) {
            X.Append(VECTOR<T, dim>(-len / 2 + i * segLen, 0, -len / 2 + j * segLen));
        }
    }

    for (int i = 0; i < nSeg; ++i) {
        for (int j = 0; j < nSeg; ++j) {
            rod.emplace_back(counter[0] + i * (nSeg + 1) + j, counter[0] + i * (nSeg + 1) + j + 1);
            rod.emplace_back(counter[0] + i * (nSeg + 1) + j, counter[0] + (i + 1) * (nSeg + 1) + j);
        }
    }
    for (int i = 0; i < nSeg; ++i) {
        rod.emplace_back(counter[0] + i * (nSeg + 1) + nSeg, counter[0] + (i + 1) * (nSeg + 1) + nSeg);
    }
    for (int j = 0; j < nSeg; ++j) {
        rod.emplace_back(counter[0] + nSeg * (nSeg + 1) + j, counter[0] + nSeg * (nSeg + 1) + j + 1);
    }

    if (midPointAmt > 0) {
        std::vector<VECTOR<int, 2>> subRods;
        for (int rodI = counter[1]; rodI < rod.size(); ++rodI) {
            const VECTOR<T, dim>& v0 = std::get<0>(X.Get_Unchecked(rod[rodI][0]));
            const VECTOR<T, dim>& v1 = std::get<0>(X.Get_Unchecked(rod[rodI][1]));
            T ratio = 1.0 / (midPointAmt + 1);
            X.Append(v0 * ratio + v1 * (1 - ratio));
            int rodEndVI = rod[rodI][1];
            rod[rodI][1] = X.size - 1;
            for (int mpI = 1; mpI < midPointAmt; ++mpI) {
                T ratio = (mpI + 1) / (midPointAmt + 1);
                X.Append(v0 * ratio + v1 * (1 - ratio));
                subRods.emplace_back(X.size - 2, X.size - 1);
            }
            subRods.emplace_back(X.size - 1, rodEndVI);
        }
        rod.insert(rod.end(), subRods.begin(), subRods.end());
    }

    counter[2] = X.size;
    counter[3] = rod.size();
    return counter;
}

template <class T, int dim = 3>
T Initialize_Discrete_Rod(
    MESH_NODE<T, dim>& X,
    const std::vector<VECTOR<int, 2>>& rod,
    T rho, T E, T thickness, 
    const VECTOR<T, dim>& gravity,
    std::vector<T>& b,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix
    std::vector<VECTOR<T, 3>>& rodInfo, // rodInfo: k_i, l_rest_i
    std::vector<VECTOR<int, 3>>& rodHinge,
    std::vector<VECTOR<T, 3>>& rodHingeInfo,
    T rodBendStiffMult,
    T h, T dHat2, VECTOR<T, 3>& kappa)
{
    std::map<int, std::vector<int>> rodVNeighbor;
    rodInfo.resize(rod.size());
    std::vector<Eigen::Triplet<T>> triplets;
    int i = 0;
    for (const auto& segI : rod) {
        rodVNeighbor[segI[0]].emplace_back(segI[1]);
        rodVNeighbor[segI[1]].emplace_back(segI[0]);

        const VECTOR<T, dim>& X0 = std::get<0>(X.Get_Unchecked(segI[0]));
        const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(segI[1]));

        rodInfo[i][0] = E;
        rodInfo[i][1] = (X0 - X1).length();
        rodInfo[i][2] = thickness;

        const T massPortion = rodInfo[i][1] * M_PI * thickness * thickness / 4 * rho / 2;
        for (int endI = 0; endI < 2; ++endI) {
            std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(segI[endI])) += massPortion;
            for (int dimI = 0; dimI < dim; ++dimI) {
                triplets.emplace_back(segI[endI] * dim + dimI, segI[endI] * dim + dimI, massPortion);
                b[segI[endI] * dim + dimI] += massPortion * gravity[dimI];
            }
        }

        ++i;
    }
    CSR_MATRIX<T> M_add;
    M_add.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
    M.Get_Matrix() += M_add.Get_Matrix();

    for (const auto& RVNbI : rodVNeighbor) {
        if (RVNbI.second.size() >= 2) {
            // > 2 means non-manifold connection
            for (int i = 0; i < RVNbI.second.size(); ++i) {
                for (int j = i + 1; j < RVNbI.second.size(); ++j) {
                    rodHinge.emplace_back(RVNbI.second[i], RVNbI.first, RVNbI.second[j]);
                }
            }
        }
    }
    rodHingeInfo.resize(rodHinge.size());
    for (int rhI = 0; rhI < rodHinge.size(); ++rhI) {
        const VECTOR<T, dim>& X0 = std::get<0>(X.Get_Unchecked(rodHinge[rhI][0]));
        const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked(rodHinge[rhI][1]));
        const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked(rodHinge[rhI][2]));

        rodHingeInfo[rhI][0] = E * rodBendStiffMult;
        rodHingeInfo[rhI][1] = (X1 - X0).length() + (X2 - X1).length();
        rodHingeInfo[rhI][2] = thickness;
    }

    //TODO: EIPC volume
    if (kappa[0] == 0 && rodInfo.size()) {
        const T nu = 0.4;
        const T h2vol = h * h * rodInfo[0][1] * thickness * thickness / 10000;
        const T lambda = E * nu / ((T)1 - nu * nu);
        const T mu = E / ((T)2 * ((T)1 + nu));
        kappa[0] = h2vol * mu;
        kappa[1] = h2vol * lambda;
        kappa[2] = nu;
        dHat2 = thickness * thickness;
    }
    return dHat2;
}

template<class T, int dim>
void Compute_Max_And_Avg_Stretch_Rod(
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo, // rodInfo: k_i, l_rest_i
    T& maxs, T& avgs)
{
    TIMER_FLAG("Compute_Max_And_Avg_Stretch_Rod");

    maxs = 1.0, avgs = 0.0;
    int stretchedAmt = 0;
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        for (int rodI = 0; rodI < rod.size(); ++rodI) {
            const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(rod[rodI][0]));
            const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(rod[rodI][1]));

            T maxsI = (x1 - x2).length() / rodInfo[rodI][1];

            if (maxsI > maxs) {
                maxs = maxsI;
            }
            if (maxsI > 1) {
                ++stretchedAmt;
                avgs += maxsI;
            }
        }
    }

    if (stretchedAmt) {
        avgs /= stretchedAmt;
    }
}

}