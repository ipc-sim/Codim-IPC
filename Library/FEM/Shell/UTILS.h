#pragma once

namespace JGSL {

template <class T>
Eigen::Matrix<T, 3, 3> crossMatrix(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> ret;
    ret << 0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
    return ret;
}

template <class T, int dim = 3>
void Compute_SFF(
    MESH_NODE<T, dim>& X, 
    MESH_ELEM<dim - 1>& Elem,
    const std::map<std::pair<int, int>, int>& edge2tri,
    int elemI,
    Eigen::Matrix<T, dim, 1>& cNormal,
    Eigen::Matrix<T, dim, 1> oppNormals[],
    T mnorms[],
    MATRIX<T, dim - 1>& D,
    int *oppvI = NULL,
    Eigen::Matrix<T, 3, 9> *dn = NULL)
{
    const VECTOR<int, 3>& elemVInd = std::get<0>(Elem.Get_Unchecked(elemI));
    const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
    const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
    const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));

    cNormal = std::move(Eigen::Matrix<T, dim, 1>(cross(x2 - x1, x3 - x1).data));
    for (int i = 0; i < 3; ++i) {
        int v2lI = (i + 2) % 3;
        int v1lI = (i + 1) % 3;
        auto finder = edge2tri.find(std::pair<int, int>(elemVInd[v2lI], elemVInd[v1lI]));
        if (finder == edge2tri.end()) {
            oppNormals[i].setZero();
            if (dn) {
                dn[i].setZero();
            }
            if (oppvI) {
                oppvI[i] = -1;
            }
        }
        else {
            const VECTOR<int, 3>& oppElemVInd = std::get<0>(Elem.Get_Unchecked(finder->second));
            const VECTOR<T, dim>& oppx1 = std::get<0>(X.Get_Unchecked(oppElemVInd[0]));
            const VECTOR<T, dim>& oppx2 = std::get<0>(X.Get_Unchecked(oppElemVInd[1]));
            const VECTOR<T, dim>& oppx3 = std::get<0>(X.Get_Unchecked(oppElemVInd[2]));
            Eigen::Matrix<T, dim, 1> oppxe[3] = { 
                std::move(Eigen::Matrix<T, dim, 1>(oppx1.data)), 
                std::move(Eigen::Matrix<T, dim, 1>(oppx2.data)), 
                std::move(Eigen::Matrix<T, dim, 1>(oppx3.data))
            };
            oppNormals[i] = (oppxe[1] - oppxe[0]).cross(oppxe[2] - oppxe[0]);

            if (dn || oppvI) {
                int oppv0lI = 0;
                for (int j = 0; j < 3; ++j) {
                    if (oppElemVInd[j] == elemVInd[v1lI]) {
                        oppv0lI = (j + 1) % 3;
                        break;
                    }
                }

                if (oppvI) {
                    oppvI[i] = oppElemVInd[oppv0lI];
                }

                if (dn) {
                    int oppv1lI = (oppv0lI + 1) % 3;
                    int oppv2lI = (oppv0lI + 2) % 3;

                    dn[i].template block<3, 3>(0, 0) = crossMatrix<T>(oppxe[oppv2lI] - oppxe[oppv1lI]);
                    dn[i].template block<3, 3>(0, 3) = crossMatrix<T>(oppxe[oppv0lI] - oppxe[oppv2lI]);
                    dn[i].template block<3, 3>(0, 6) = crossMatrix<T>(oppxe[oppv1lI] - oppxe[oppv0lI]);
                }
            }
        }
        mnorms[i] = (oppNormals[i] + cNormal).norm();
    }

    T II[3];
    II[0] = Eigen::Matrix<T, dim, 1>((x2 + x3 - x1 * 2).data).dot(oppNormals[0]) / mnorms[0];
    II[1] = Eigen::Matrix<T, dim, 1>((x3 + x1 - x2 * 2).data).dot(oppNormals[1]) / mnorms[1];
    II[2] = Eigen::Matrix<T, dim, 1>((x1 + x2 - x3 * 2).data).dot(oppNormals[2]) / mnorms[2];
    D(0, 0) = II[0] + II[1];
    D(1, 0) = D(0, 1) = II[0];
    D(1, 1) = II[0] + II[2];
}

}