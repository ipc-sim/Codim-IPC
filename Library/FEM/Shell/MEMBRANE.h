#pragma once

#include <Physics/FIXED_COROTATED.h>

namespace JGSL {

template<class T, int dim, bool useNH = true>
void Compute_Membrane_Energy(
    MESH_ELEM<dim - 1>& Elem, T h,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    T& E)
{
    TIMER_FLAG("Compute_Membrane_Energy");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F, vol, lambda, mu] = data;
            if (!(DBCb[elemVInd[0]] && DBCb[elemVInd[1]] && DBCb[elemVInd[2]])) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                MATRIX<T, dim - 1> IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));
                IB.invert();

                const VECTOR<T, dim> e01 = x2 - x1;
                const VECTOR<T, dim> e02 = x3 - x1;
                MATRIX<T, dim - 1> A;
                A(0, 0) = e01.length2();
                A(1, 0) = A(0, 1) = e01.dot(e02);
                A(1, 1) = e02.length2();

                if constexpr (useNH) {
                    const T lnJ = std::log(A.determinant() * IB.determinant()) / 2;
                    E += h * h * vol * (mu / 2 * ((IB * A).trace() - 2 - 2 * lnJ) + lambda / 2 * lnJ * lnJ);
                }
                else {
                    MATRIX<T, dim - 1> M = IB * A; M(0, 0) -= 1; M(1, 1) -= 1;
                    E += h * h * vol / 4 * (0.5 * lambda * pow(M.trace(), 2) + mu * (M * M).trace());
                }
            }
        });
    }
}

template<class T, int dim, bool useNH = true>
void Compute_Membrane_Gradient(
    MESH_ELEM<dim - 1>& Elem, T h,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr) 
{
    TIMER_FLAG("Compute_Membrane_Gradient");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F, vol, lambda, mu] = data;
            if (!(DBCb[elemVInd[0]] && DBCb[elemVInd[1]] && DBCb[elemVInd[2]])) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                Eigen::Matrix<T, dim, 1> x1e(x1.data), x2e(x2.data), x3e(x3.data);
                MATRIX<T, dim - 1> IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));
                IB.invert();

                const VECTOR<T, dim> e01 = x2 - x1;
                const VECTOR<T, dim> e02 = x3 - x1;
                MATRIX<T, dim - 1> A;
                A(0, 0) = e01.length2();
                A(1, 0) = A(0, 1) = e01.dot(e02);
                A(1, 1) = e02.length2();

                MATRIX<T, dim - 1> temp;
                if constexpr (useNH) {
                    MATRIX<T, dim - 1> IA = A; IA.invert();
                    const T lnJ = std::log(A.determinant() * IB.determinant()) / 2;
                    temp = h * h * vol * (mu / 2 * IB + (-mu + lambda * lnJ) / 2 * IA);
                }
                else {
                    MATRIX<T, dim - 1> M = IB * A; M(0, 0) -= 1; M(1, 1) -= 1;
                    temp = h * h * vol / 4 * (lambda * M.trace() * IB + 2 * mu * M * IB);
                }

                Eigen::Matrix<T, 4, 9> dA_div_dx;
                dA_div_dx.setZero();
                dA_div_dx.template block<1, 3>(0, 3) += 2.0 * (x2e - x1e).transpose();
                dA_div_dx.template block<1, 3>(0, 0) -= 2.0 * (x2e - x1e).transpose();
                dA_div_dx.template block<1, 3>(1, 6) += (x2e - x1e).transpose();
                dA_div_dx.template block<1, 3>(1, 3) += (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(1, 0) += -(x2e - x1e).transpose() - (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(2, 6) += (x2e - x1e).transpose();
                dA_div_dx.template block<1, 3>(2, 3) += (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(2, 0) += -(x2e - x1e).transpose() - (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(3, 6) += 2.0 * (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(3, 0) -= 2.0 * (x3e - x1e).transpose();

                for (int endI = 0; endI < dim; ++endI) {
                    VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[endI]));
                    for (int dimI = 0; dimI < dim; ++dimI) {
                        int i = endI * dim + dimI;
                        g[dimI] += dA_div_dx(0, i) * temp(0, 0) + dA_div_dx(1, i) * temp(1, 0) + 
                            dA_div_dx(2, i) * temp(0, 1) + dA_div_dx(3, i) * temp(1, 1);
                    }
                }
            }
        });
    }
}

template<class T, int dim, bool useNH = true>
void Compute_Membrane_Hessian(
    MESH_ELEM<dim - 1>& Elem, 
    T h, bool projectSPD,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Membrane_Hessian");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        Eigen::Matrix<T, 9, 9> ahess[4];
        for (int i = 0; i < 4; i++) {
            ahess[i].setZero();
        }
        Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();
        ahess[0].template block<3, 3>(0, 0) += 2.0*I;
        ahess[0].template block<3, 3>(3, 3) += 2.0*I;
        ahess[0].template block<3, 3>(0, 3) -= 2.0*I;
        ahess[0].template block<3, 3>(3, 0) -= 2.0*I;

        ahess[1].template block<3, 3>(3, 6) += I;
        ahess[1].template block<3, 3>(6, 3) += I;
        ahess[1].template block<3, 3>(0, 3) -= I;
        ahess[1].template block<3, 3>(0, 6) -= I;
        ahess[1].template block<3, 3>(3, 0) -= I;
        ahess[1].template block<3, 3>(6, 0) -= I;
        ahess[1].template block<3, 3>(0, 0) += 2.0*I;

        ahess[2].template block<3, 3>(3, 6) += I;
        ahess[2].template block<3, 3>(6, 3) += I;
        ahess[2].template block<3, 3>(0, 3) -= I;
        ahess[2].template block<3, 3>(0, 6) -= I;
        ahess[2].template block<3, 3>(3, 0) -= I;
        ahess[2].template block<3, 3>(6, 0) -= I;
        ahess[2].template block<3, 3>(0, 0) += 2.0*I;

        ahess[3].template block<3, 3>(0, 0) += 2.0*I;
        ahess[3].template block<3, 3>(6, 6) += 2.0*I;
        ahess[3].template block<3, 3>(0, 6) -= 2.0*I;
        ahess[3].template block<3, 3>(6, 0) -= 2.0*I;

        std::vector<int> tripletStartInd(Elem.size);
        int nonDBCElemCount = 0;
        Elem.Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            if (!(DBCb[elemVInd[0]] && DBCb[elemVInd[1]] && DBCb[elemVInd[2]])) {
                tripletStartInd[id] = triplets.size() + nonDBCElemCount * 81;
                ++nonDBCElemCount;
            }
            else {
                tripletStartInd[id] = -1;
            }
        });

        triplets.resize(triplets.size() + nonDBCElemCount * 81);
        Elem.Join(elasticityAttr).Par_Each([&](int id, auto data) {
            if (tripletStartInd[id] >= 0) {
                auto &[elemVInd, F, vol, lambda, mu] = data;
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                Eigen::Matrix<T, dim, 1> x1e(x1.data), x2e(x2.data), x3e(x3.data);
                MATRIX<T, dim - 1> IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));
                IB.invert();

                const VECTOR<T, dim> e01 = x2 - x1;
                const VECTOR<T, dim> e02 = x3 - x1;
                MATRIX<T, dim - 1> A;
                A(0, 0) = e01.length2();
                A(1, 0) = A(0, 1) = e01.dot(e02);
                A(1, 1) = e02.length2();

                Eigen::Matrix<T, 4, 9> dA_div_dx;
                dA_div_dx.setZero();
                dA_div_dx.template block<1, 3>(0, 3) += 2.0 * (x2e - x1e).transpose();
                dA_div_dx.template block<1, 3>(0, 0) -= 2.0 * (x2e - x1e).transpose();
                dA_div_dx.template block<1, 3>(1, 6) += (x2e - x1e).transpose();
                dA_div_dx.template block<1, 3>(1, 3) += (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(1, 0) += -(x2e - x1e).transpose() - (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(2, 6) += (x2e - x1e).transpose();
                dA_div_dx.template block<1, 3>(2, 3) += (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(2, 0) += -(x2e - x1e).transpose() - (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(3, 6) += 2.0 * (x3e - x1e).transpose();
                dA_div_dx.template block<1, 3>(3, 0) -= 2.0 * (x3e - x1e).transpose();

                Eigen::Matrix<T, 9, 9> hessian;
                if constexpr (useNH) {
                    MATRIX<T, dim - 1> IA = A; IA.invert();
                    Eigen::Matrix<T, 1, 9> ainvda;
                    for (int endI = 0; endI < dim; ++endI) {
                        for (int dimI = 0; dimI < dim; ++dimI) {
                            int i = endI * dim + dimI;
                            ainvda[i] = dA_div_dx(0, i) * IA(0, 0) + dA_div_dx(1, i) * IA(1, 0) + 
                                dA_div_dx(2, i) * IA(0, 1) + dA_div_dx(3, i) * IA(1, 1);
                        }
                    }
                    const T deta = A.determinant();
                    const T lnJ = std::log(deta * IB.determinant()) / 2;
                    const T term1 = (-mu + lambda * lnJ) / 2;
                    hessian = (-term1 + lambda / 4) * ainvda.transpose() * ainvda;

                    Eigen::Matrix<T, 4, 9> aderivadj;
                    aderivadj << dA_div_dx.row(3), -dA_div_dx.row(1), -dA_div_dx.row(2), dA_div_dx.row(0);
                    hessian += term1 / deta * aderivadj.transpose() * dA_div_dx;

                    for(int i = 0; i < 2; ++i) {
                        for(int j = 0; j < 2; ++j) {
                            hessian += (term1 * IA(i, j) + mu / 2 * IB(i, j)) * ahess[i + j * 2];
                        }
                    }
                    hessian *= h * h * vol;
                }
                else {
                    Eigen::Matrix<T, 1, 9> inner;
                    for (int i = 0; i < 9; ++i) {
                        inner[i] = dA_div_dx(0, i) * IB(0, 0) + dA_div_dx(1, i) * IB(1, 0) + 
                            dA_div_dx(2, i) * IB(0, 1) + dA_div_dx(3, i) * IB(1, 1);
                    }
                    hessian = lambda * inner.transpose() * inner;

                    MATRIX<T, dim - 1> M = IB * A; M(0, 0) -= 1; M(1, 1) -= 1;
                    MATRIX<T, dim - 1> Mainv = M * IB;
                    for(int i = 0; i < 2; ++i) {
                        for(int j = 0; j < 2; ++j) {
                            hessian += (lambda * M.trace() * IB(i, j) + 2 * mu * Mainv(i, j)) * ahess[i + j * 2];
                        }
                    }

                    Eigen::Matrix<T, 1, 9> inner00 = IB(0, 0) * dA_div_dx.row(0) + IB(0, 1) * dA_div_dx.row(2);
                    Eigen::Matrix<T, 1, 9> inner01 = IB(0, 0) * dA_div_dx.row(1) + IB(0, 1) * dA_div_dx.row(3);
                    Eigen::Matrix<T, 1, 9> inner10 = IB(1, 0) * dA_div_dx.row(0) + IB(1, 1) * dA_div_dx.row(2);
                    Eigen::Matrix<T, 1, 9> inner11 = IB(1, 0) * dA_div_dx.row(1) + IB(1, 1) * dA_div_dx.row(3);
                    hessian += 2 * mu * inner00.transpose() * inner00;
                    hessian += 2 * mu * (inner01.transpose() * inner10  + inner10.transpose() * inner01);
                    hessian += 2 * mu * inner11.transpose() * inner11;

                    hessian *= h * h * vol / 4;
                }

                if (projectSPD) {
                    makePD(hessian);
                }

                int indMap[9] = {
                    elemVInd[0] * dim,
                    elemVInd[0] * dim + 1,
                    elemVInd[0] * dim + 2,
                    elemVInd[1] * dim,
                    elemVInd[1] * dim + 1,
                    elemVInd[1] * dim + 2,
                    elemVInd[2] * dim,
                    elemVInd[2] * dim + 1,
                    elemVInd[2] * dim + 2
                };
                for (int i = 0; i < 9; ++i) {
                    int tripletIStart = tripletStartInd[id] + i * 9;
                    for (int j = 0; j < 9; ++j) {
                        triplets[tripletIStart + j] = std::move(Eigen::Triplet<T>(indMap[i], indMap[j], hessian(i, j)));
                    }
                }
            }
        });
    }
}

template <class T, int dim>
void Check_Membrane_Gradient(
    MESH_ELEM<dim - 1>& Elem,
    const VECTOR_STORAGE<T, dim + 1>& DBC,
    T h, MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    T eps = 1.0e-6;

    T E0 = 0;
    Compute_Membrane_Energy(Elem, 1.0, std::vector<bool>(X.size, false), std::vector<bool>(X.size, false), X, nodeAttr, elemAttr, elasticityAttr, E0); 
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Membrane_Gradient(Elem, 1.0, std::vector<bool>(X.size, false), X, nodeAttr, elemAttr, elasticityAttr); 

    std::vector<T> grad_FD(X.size * dim);
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        T E = 0;
        Compute_Membrane_Energy(Elem, 1.0, Xperturb, nodeAttr, elemAttr, elasticityAttr, E);
        grad_FD[i] = (E - E0) / eps;
    }

    T err = 0.0, norm = 0.0;
    nodeAttr.Each([&](int id, auto data) {
        auto &[x0, v, g, m] = data;

        err += std::pow(grad_FD[id * dim] - g[0], 2);
        err += std::pow(grad_FD[id * dim + 1] - g[1], 2);

        norm += std::pow(grad_FD[id * dim], 2);
        norm += std::pow(grad_FD[id * dim + 1], 2);

        if constexpr (dim == 3) {
            err += std::pow(grad_FD[id * dim + 2] - g[2], 2);
            norm += std::pow(grad_FD[id * dim + 2], 2);
        }
    });
    printf("err_abs = %le, err_rel = %le\n", err, err / norm);
}

template <class T, int dim>
void Check_Membrane_Hessian(
    MESH_ELEM<dim - 1>& Elem,
    const VECTOR_STORAGE<T, dim + 1>& DBC, 
    T h, MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    T eps = 1.0e-6;

    MESH_NODE_ATTR<T, dim> nodeAttr0;
    nodeAttr.deep_copy_to(nodeAttr0);
    nodeAttr0.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Membrane_Gradient(Elem, 1.0, std::vector<bool>(X.size, false), X, nodeAttr0, elemAttr, elasticityAttr);
    std::vector<Eigen::Triplet<T>> HStriplets;
    Compute_Membrane_Hessian(Elem, 1.0, false, std::vector<bool>(X.size, false), X, nodeAttr, elemAttr, elasticityAttr, HStriplets); 
    CSR_MATRIX<T> HS;
    HS.Construct_From_Triplet(X.size * dim, X.size * dim, HStriplets);

    std::vector<Eigen::Triplet<T>> HFDtriplets;
    HFDtriplets.reserve(HStriplets.size());
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
        Compute_Membrane_Gradient(Elem, 1.0, std::vector<bool>(X.size, false), Xperturb, nodeAttr, elemAttr, elasticityAttr);
        for (int vI = 0; vI < X.size; ++vI) {
            const VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(vI));
            const VECTOR<T, dim>& g0 = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr0.Get_Unchecked(vI));
            const VECTOR<T, dim> hFD = (g - g0) / eps;
            if (hFD.length2() > eps) {
                HFDtriplets.emplace_back(i, vI * dim, hFD[0]);
                HFDtriplets.emplace_back(i, vI * dim + 1, hFD[1]);
                if constexpr (dim == 3) {
                    HFDtriplets.emplace_back(i, vI * dim + 2, hFD[2]);
                }
            }
        }
    }
    CSR_MATRIX<T> HFD;
    HFD.Construct_From_Triplet(X.size * dim, X.size * dim, HFDtriplets);

    T err = (HS.Get_Matrix() - HFD.Get_Matrix()).squaredNorm(), norm = HFD.Get_Matrix().squaredNorm();
    printf("err_abs = %le, err_rel = %le\n", err, err / norm);
}

}