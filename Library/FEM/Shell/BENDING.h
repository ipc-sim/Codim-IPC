#pragma once

#include <Math/DIHEDRAL_ANGLE.h>

#include <FEM/Shell/UTILS.h>

namespace JGSL {

template<class T, int dim, bool KL>
void Compute_Bending_Energy(
    MESH_ELEM<dim - 1>& Elem, T h, 
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int ,4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    T thickness, T bendingStiffMult,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    T& E)
{
    TIMER_FLAG("Compute_Bending_Energy");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        if constexpr (KL) {
            Elem.Join(elasticityAttr).Each([&](int id, auto data) {
                auto &[elemVInd, F, vol, lambda, mu] = data;
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                MATRIX<T, dim - 1> IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));
                IB.invert();
                const MATRIX<T, dim - 1>& D = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(id));
                
                Eigen::Matrix<T, dim, 1> cNormal;
                Eigen::Matrix<T, dim, 1> oppNormals[3];
                T mnorms[3];
                MATRIX<T, dim - 1> C;
                Compute_SFF(X, Elem, edge2tri, id, cNormal, oppNormals, mnorms, C);
                MATRIX<T, dim - 1> M = IB * (C - D);

                const T coeff = bendingStiffMult * pow(thickness, 3) / 12;
                const T dA = 0.5 / sqrt(IB.determinant());
                const T StVK = 0.5 * lambda * std::pow(M.trace(), 2) + mu * (M * M).trace();
                E += h * h * coeff * dA * StVK;
            });
        }
        else {
            if (elemAttr.size) {
                const T k = bendingStiffMult * std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(0))(0, 0);
                for (int eI = 0; eI < edgeStencil.size(); ++eI) {
                    if (DBCb[edgeStencil[eI][0]] && DBCb[edgeStencil[eI][1]] && 
                        DBCb[edgeStencil[eI][2]] && DBCb[edgeStencil[eI][3]]) 
                    { 
                        continue; 
                    }
                    const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][0]));
                    const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][1]));
                    const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][2]));
                    const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][3]));
                    Eigen::Matrix<T, dim, 1> x0e(x0.data), x1e(x1.data), x2e(x2.data), x3e(x3.data);
                    
                    const T thetabar = edgeInfo[eI][0];
                    const T ebarnorm = edgeInfo[eI][1];
                    const T hbar = edgeInfo[eI][2];

                    T theta;
                    Compute_Dihedral_Angle(x0e, x1e, x2e, x3e, theta);

                    E += h * h * k * (theta - thetabar) * (theta - thetabar) * ebarnorm / hbar;
                }
            }
        }
    }
}

template<class T, int dim, bool KL>
void Compute_Bending_Gradient(
    MESH_ELEM<dim - 1>& Elem, T h,
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    T thickness, T bendingStiffMult,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    TIMER_FLAG("Compute_Bending_Gradient");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        if constexpr (KL) {
            Elem.Join(elasticityAttr).Each([&](int id, auto data) {
                auto &[elemVInd, F, vol, lambda, mu] = data;
                int vInd[6] = { elemVInd[0], elemVInd[1], elemVInd[2] };
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                Eigen::Matrix<T, dim, 1> qs[3] = { 
                    std::move(Eigen::Matrix<T, dim, 1>(x1.data)), 
                    std::move(Eigen::Matrix<T, dim, 1>(x2.data)), 
                    std::move(Eigen::Matrix<T, dim, 1>(x3.data))
                };
                MATRIX<T, dim - 1> IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));
                IB.invert();
                const MATRIX<T, dim - 1>& D = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(id));
                
                Eigen::Matrix<T, dim, 1> cNormal;
                Eigen::Matrix<T, dim, 1> oppNormals[3];
                T mnorms[3];
                MATRIX<T, dim - 1> C;
                Eigen::Matrix<T, 3, 9> dn[3];
                Compute_SFF(X, Elem, edge2tri, id, cNormal, oppNormals, mnorms, C, vInd + 3, dn);
                MATRIX<T, dim - 1> M = IB * (C - D);

                const T coeff = bendingStiffMult * pow(thickness, 3) / 12;
                const T dA = 0.5 / sqrt(IB.determinant());
                MATRIX<T, dim - 1> temp = h * h * coeff * dA * (lambda * M.trace() * IB + 2 * mu * M * IB);

                Eigen::Matrix<T, 3, 9> dcn;
                dcn.template block<3, 3>(0, 0) = crossMatrix<T>(qs[2] - qs[1]);
                dcn.template block<3, 3>(0, 3) = crossMatrix<T>(qs[0] - qs[2]);
                dcn.template block<3, 3>(0, 6) = crossMatrix<T>(qs[1] - qs[0]);

                Eigen::Matrix<T, 3, 18> IIderiv;
                IIderiv.setZero();
                for (int i = 0; i < 3; ++i) {
                    int ip1 = (i + 1) % 3;
                    int ip2 = (i + 2) % 3;
                    IIderiv.template block<1, 3>(i, 3*i) += -2.0 * oppNormals[i].transpose() / mnorms[i];
                    IIderiv.template block<1, 3>(i, 3*ip1) += 1.0 * oppNormals[i].transpose() / mnorms[i];
                    IIderiv.template block<1, 3>(i, 3*ip2) += 1.0 * oppNormals[i].transpose() / mnorms[i];
                    
                    IIderiv.template block<1, 3>(i, 9 + 3*i) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].template block<3, 3>(0, 0);
                    IIderiv.template block<1, 3>(i, 3*ip2) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].template block<3, 3>(0, 3);
                    IIderiv.template block<1, 3>(i, 3*ip1) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].template block<3, 3>(0, 6);

                    IIderiv.template block<1, 3>(i, 9 + 3*i) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].template block<3, 3>(0, 0);
                    IIderiv.template block<1, 3>(i, 3*ip2) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].template block<3, 3>(0, 3);
                    IIderiv.template block<1, 3>(i, 3*ip1) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].template block<3, 3>(0, 6);

                    IIderiv.template block<1, 3>(i, 0) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.template block<3, 3>(0, 0);
                    IIderiv.template block<1, 3>(i, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.template block<3, 3>(0, 3);
                    IIderiv.template block<1, 3>(i, 6) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.template block<3, 3>(0, 6);
                }
                Eigen::Matrix<T, 4, 18> dC_div_dx;
                dC_div_dx.setZero();
                dC_div_dx.row(0) += IIderiv.row(0);
                dC_div_dx.row(0) += IIderiv.row(1);
                dC_div_dx.row(1) += IIderiv.row(0);
                dC_div_dx.row(2) += IIderiv.row(0);
                dC_div_dx.row(3) += IIderiv.row(0);
                dC_div_dx.row(3) += IIderiv.row(2);

                for (int v = 0; v < 6; ++v) {
                    if (vInd[v] < 0) {
                        continue;
                    }
                    VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(vInd[v]));
                    for (int d = 0; d < dim; ++d) {
                        int i = v * dim + d;
                        g[d] += dC_div_dx(0, i) * temp(0, 0) + dC_div_dx(1, i) * temp(1, 0) + 
                            dC_div_dx(2, i) * temp(0, 1) + dC_div_dx(3, i) * temp(1, 1);
                    }
                }
            });
        }
        else {
            if (elemAttr.size) {
                const T k = bendingStiffMult * std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(0))(0, 0);
                for (int eI = 0; eI < edgeStencil.size(); ++eI) {
                    if (DBCb[edgeStencil[eI][0]] && DBCb[edgeStencil[eI][1]] && 
                        DBCb[edgeStencil[eI][2]] && DBCb[edgeStencil[eI][3]]) 
                    { 
                        continue; 
                    }
                    const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][0]));
                    const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][1]));
                    const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][2]));
                    const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][3]));
                    Eigen::Matrix<T, dim, 1> x0e(x0.data), x1e(x1.data), x2e(x2.data), x3e(x3.data);
                    
                    const T thetabar = edgeInfo[eI][0];
                    const T ebarnorm = edgeInfo[eI][1];
                    const T hbar = edgeInfo[eI][2];

                    T theta;
                    Compute_Dihedral_Angle(x0e, x1e, x2e, x3e, theta);
                    Eigen::Matrix<T, 12, 1> grad;
                    Compute_Dihedral_Angle_Gradient(x0e, x1e, x2e, x3e, grad);

                    grad *= h * h * k * 2 * (theta - thetabar) * ebarnorm / hbar;

                    for (int i = 0; i < 4; ++i) {
                        VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(edgeStencil[eI][i]));
                        for (int j = 0; j < 3; ++j) {
                            g[j] += grad[i * 3 + j];
                        }
                    }
                }
            }
        }
    }
}

template<class T, int dim, bool KL>
void Compute_Bending_Hessian(
    MESH_ELEM<dim - 1>& Elem, 
    T h, bool projectSPD,
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    T thickness, T bendingStiffMult,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Bending_Hessian");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        if constexpr (KL) {
            Eigen::Matrix<T, 9, 9> hn[3];
            for (int j = 0; j < 3; ++j) {
                hn[j].setZero();
                Eigen::Matrix<T, dim, 1> ej(0, 0, 0);
                ej[j] = 1.0;
                Eigen::Matrix<T, dim, dim> ejc = crossMatrix<T>(ej);
                hn[j].block(0, 3, 3, 3) -= ejc;
                hn[j].block(0, 6, 3, 3) += ejc;
                hn[j].block(3, 6, 3, 3) -= ejc;
                hn[j].block(3, 0, 3, 3) += ejc;
                hn[j].block(6, 0, 3, 3) -= ejc;
                hn[j].block(6, 3, 3, 3) += ejc;
            }

            int newTripletStartI = triplets.size();
            triplets.resize(newTripletStartI + Elem.size * 18 * 18);
            Elem.Join(elasticityAttr).Par_Each([&](int id, auto data) {
                auto &[elemVInd, F, vol, lambda, mu] = data;
                int vInd[6] = { elemVInd[0], elemVInd[1], elemVInd[2] };
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                Eigen::Matrix<T, dim, 1> qs[3] = { 
                    std::move(Eigen::Matrix<T, dim, 1>(x1.data)), 
                    std::move(Eigen::Matrix<T, dim, 1>(x2.data)), 
                    std::move(Eigen::Matrix<T, dim, 1>(x3.data))
                };
                MATRIX<T, dim - 1> IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));
                IB.invert();
                const MATRIX<T, dim - 1>& D = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(id));
                
                Eigen::Matrix<T, dim, 1> cNormal;
                Eigen::Matrix<T, dim, 1> oppNormals[3];
                T mnorms[3];
                MATRIX<T, dim - 1> C;
                Eigen::Matrix<T, 3, 9> dn[3];
                Compute_SFF(X, Elem, edge2tri, id, cNormal, oppNormals, mnorms, C, vInd + 3, dn);
                MATRIX<T, dim - 1> M = IB * (C - D);

                const T coeff = bendingStiffMult * pow(thickness, 3) / 12;
                const T dA = 0.5 / sqrt(IB.determinant());

                Eigen::Matrix<T, 3, 9> dcn;
                dcn.template block<3, 3>(0, 0) = crossMatrix<T>(qs[2] - qs[1]);
                dcn.template block<3, 3>(0, 3) = crossMatrix<T>(qs[0] - qs[2]);
                dcn.template block<3, 3>(0, 6) = crossMatrix<T>(qs[1] - qs[0]);

                Eigen::Matrix<T, 3, 18> IIderiv;
                IIderiv.setZero();
                Eigen::Matrix<T, 18, 18> IIhess[3];
                for (int i = 0; i < 3; ++i) {
                    int ip1 = (i + 1) % 3;
                    int ip2 = (i + 2) % 3;

                    // gradient
                    IIderiv.template block<1, 3>(i, 3*i) += -2.0 * oppNormals[i].transpose() / mnorms[i];
                    IIderiv.template block<1, 3>(i, 3*ip1) += 1.0 * oppNormals[i].transpose() / mnorms[i];
                    IIderiv.template block<1, 3>(i, 3*ip2) += 1.0 * oppNormals[i].transpose() / mnorms[i];
                    
                    IIderiv.template block<1, 3>(i, 9 + 3*i) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].template block<3, 3>(0, 0);
                    IIderiv.template block<1, 3>(i, 3*ip2) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].template block<3, 3>(0, 3);
                    IIderiv.template block<1, 3>(i, 3*ip1) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].template block<3, 3>(0, 6);

                    IIderiv.template block<1, 3>(i, 9 + 3*i) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].template block<3, 3>(0, 0);
                    IIderiv.template block<1, 3>(i, 3*ip2) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].template block<3, 3>(0, 3);
                    IIderiv.template block<1, 3>(i, 3*ip1) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].template block<3, 3>(0, 6);

                    IIderiv.template block<1, 3>(i, 0) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.template block<3, 3>(0, 0);
                    IIderiv.template block<1, 3>(i, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.template block<3, 3>(0, 3);
                    IIderiv.template block<1, 3>(i, 6) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.template block<3, 3>(0, 6);

                    // hessian
                    IIhess[i].setZero();
                    int miidx[3];
                    miidx[0] = 9 + 3 * i;
                    miidx[1] = 3 * ((i + 2) % 3);
                    miidx[2] = 3 * ((i + 1) % 3);

                    for (int j = 0; j < 3; j++)
                    {
                        IIhess[i].block(miidx[j], 3 * ip1, 3, 3) += dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i];
                        IIhess[i].block(miidx[j], 3 * ip2, 3, 3) += dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i];
                        IIhess[i].block(miidx[j], 3 * i, 3, 3) += -2.0 * dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i];

                        IIhess[i].block(miidx[j], 3 * ip1, 3, 3) += -dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();
                        IIhess[i].block(miidx[j], 3 * ip2, 3, 3) += -dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();
                        IIhess[i].block(miidx[j], 3 * i, 3, 3) += 2.0 * dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();

                        IIhess[i].block(3 * j, 3 * ip1, 3, 3) += -dcn.block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();
                        IIhess[i].block(3 * j, 3 * ip2, 3, 3) += -dcn.block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();
                        IIhess[i].block(3 * j, 3 * i, 3, 3) += 2.0 * dcn.block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();

                        IIhess[i].block(3 * ip1, miidx[j], 3, 3) += dn[i].block(0, 3 * j, 3, 3) / mnorms[i];
                        IIhess[i].block(3 * ip2, miidx[j], 3, 3) += dn[i].block(0, 3 * j, 3, 3) / mnorms[i];
                        IIhess[i].block(3 * i, miidx[j], 3, 3) += -2.0 * dn[i].block(0, 3 * j, 3, 3) / mnorms[i];

                        for (int k = 0; k < 3; k++)
                        {
                            IIhess[i].block(miidx[j], miidx[k], 3, 3) += -dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal) * (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() * dn[i].block(0, 3 * k, 3, 3);                    
                            IIhess[i].block(3 * j, miidx[k], 3, 3) += -dcn.block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal) * (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() * dn[i].block(0, 3 * k, 3, 3);
                        }

                        IIhess[i].block(3 * ip1, miidx[j], 3, 3) += -1.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * j, 3, 3);
                        IIhess[i].block(3 * ip2, miidx[j], 3, 3) += -1.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * j, 3, 3);
                        IIhess[i].block(3 * i, miidx[j], 3, 3) += 2.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * j, 3, 3);

                        IIhess[i].block(3 * ip1, 3 * j, 3, 3) += -1.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * j, 3, 3);
                        IIhess[i].block(3 * ip2, 3 * j, 3, 3) += -1.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * j, 3, 3);
                        IIhess[i].block(3 * i, 3 * j, 3, 3) += 2.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * j, 3, 3);

                        for (int k = 0; k < 3; k++)
                        {
                            IIhess[i].block(miidx[j], miidx[k], 3, 3) += -dn[i].block(0, 3*j, 3, 3).transpose() * (qs[ip1] + qs[ip2] - 2.0*qs[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * k, 3, 3);
                            IIhess[i].block(miidx[j], 3*k, 3, 3) += -dn[i].block(0, 3*j, 3, 3).transpose() * (qs[ip1] + qs[ip2] - 2.0*qs[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * k, 3, 3);
                        }

                        for (int k = 0; k < 3; k++)
                        {
                            IIhess[i].block(miidx[j], miidx[k], 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * dn[i].block(0, 3 * j, 3, 3).transpose() * dn[i].block(0, 3 * k, 3, 3);
                            IIhess[i].block(miidx[j], 3*k, 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * dn[i].block(0, 3 * j, 3, 3).transpose() * dcn.block(0, 3 * k, 3, 3);
                            IIhess[i].block(3*j, miidx[k], 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * dcn.block(0, 3 * j, 3, 3).transpose() * dn[i].block(0, 3 * k, 3, 3);
                            IIhess[i].block(3*j, 3*k, 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * dcn.block(0, 3 * j, 3, 3).transpose() * dcn.block(0, 3 * k, 3, 3);

                            IIhess[i].block(miidx[j], miidx[k], 3, 3) += 3.0 * (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] * dn[i].block(0, 3 * j, 3, 3).transpose() * (oppNormals[i] + cNormal) * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * k, 3, 3);
                            IIhess[i].block(miidx[j], 3 * k, 3, 3) += 3.0 * (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] * dn[i].block(0, 3 * j, 3, 3).transpose() * (oppNormals[i] + cNormal) * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * k, 3, 3);
                            IIhess[i].block(3 * j, miidx[k], 3, 3) += 3.0 * (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] * dcn.block(0, 3 * j, 3, 3).transpose() * (oppNormals[i] + cNormal) * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * k, 3, 3);
                            IIhess[i].block(3 * j, 3 * k, 3, 3) += 3.0 * (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] * dcn.block(0, 3 * j, 3, 3).transpose() * (oppNormals[i] + cNormal) * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * k, 3, 3);

                            for (int l = 0; l < 3; l++)
                            {
                                IIhess[i].block(miidx[j], miidx[k], 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal)[l] * hn[l].block(3 * j, 3 * k, 3, 3);
                                IIhess[i].block(3*j, 3*k, 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal)[l] * hn[l].block(3 * j, 3 * k, 3, 3);
                                IIhess[i].block(miidx[j], miidx[k], 3, 3) += 1.0 / mnorms[i] * (qs[ip1] + qs[ip2] - 2.0*qs[i])[l] * hn[l].block(3 * j, 3 * k, 3, 3);
                            }
                        }
                    }
                }
                Eigen::Matrix<T, 4, 18> dC_div_dx;
                dC_div_dx.setZero();
                dC_div_dx.row(0) += IIderiv.row(0);
                dC_div_dx.row(0) += IIderiv.row(1);
                dC_div_dx.row(1) += IIderiv.row(0);
                dC_div_dx.row(2) += IIderiv.row(0);
                dC_div_dx.row(3) += IIderiv.row(0);
                dC_div_dx.row(3) += IIderiv.row(2);

                Eigen::Matrix<T, 18, 18> bhess[4];
                bhess[0] = IIhess[0] + IIhess[1];
                bhess[1] = IIhess[0];
                bhess[2] = IIhess[0];
                bhess[3] = IIhess[0] + IIhess[2];

                Eigen::Matrix<T, 1, 18> inner;
                for (int i = 0; i < 18; ++i) {
                    inner[i] = dC_div_dx(0, i) * IB(0, 0) + dC_div_dx(1, i) * IB(1, 0) + 
                        dC_div_dx(2, i) * IB(0, 1) + dC_div_dx(3, i) * IB(1, 1);
                }
                Eigen::Matrix<T, 18, 18> hessian = lambda * inner.transpose() * inner;

                MATRIX<T, dim - 1> Mainv = M * IB;
                for (int i = 0; i < 2; ++i) {
                    for(int j = 0; j < 2; ++j) {
                        hessian += (lambda * M.trace() * IB(i, j) + 2 * mu * Mainv(i, j)) * bhess[i + j * 2];
                    }
                }

                Eigen::Matrix<T, 1, 18> inner00 = IB(0, 0) * dC_div_dx.row(0) + IB(0, 1) * dC_div_dx.row(2);
                Eigen::Matrix<T, 1, 18> inner01 = IB(0, 0) * dC_div_dx.row(1) + IB(0, 1) * dC_div_dx.row(3);
                Eigen::Matrix<T, 1, 18> inner10 = IB(1, 0) * dC_div_dx.row(0) + IB(1, 1) * dC_div_dx.row(2);
                Eigen::Matrix<T, 1, 18> inner11 = IB(1, 0) * dC_div_dx.row(1) + IB(1, 1) * dC_div_dx.row(3);
                hessian += 2 * mu * inner00.transpose() * inner00;
                hessian += 2 * mu * (inner01.transpose() * inner10  + inner10.transpose() * inner01);
                hessian += 2 * mu * inner11.transpose() * inner11;

                hessian *= h * h * coeff * dA;
                if (projectSPD) {
                    makePD(hessian);
                }

                int indMap[18];
                for (int i = 0; i < 6; ++i) {
                    if (vInd[i] < 0) {
                        for (int d = 0; d < dim; ++d) {
                            int idx = i * dim + d;
                            indMap[idx] = 0;
                            hessian.row(idx).setZero();
                            hessian.col(idx).setZero();
                        }
                    }
                    else {
                        for (int d = 0; d < dim; ++d) {
                            indMap[i * dim + d] = vInd[i] * dim + d;
                        }
                    }
                }
                for (int i = 0; i < 18; ++i) {
                    int startI = newTripletStartI + id * 18 * 18 + i * 18;
                    for (int j = 0; j < 18; ++j) {
                        triplets[startI + j] = std::move(Eigen::Triplet<T>(indMap[i], indMap[j], hessian(i, j)));
                    }
                }
            });
        }
        else {
            if (elemAttr.size) {
                BASE_STORAGE<int> threads(edgeStencil.size());
                int nonDBCECount = 0;
                for (int i = 0; i < edgeStencil.size(); ++i) {
                    if (!(DBCb[edgeStencil[i][0]] && DBCb[edgeStencil[i][1]] && 
                        DBCb[edgeStencil[i][2]] && DBCb[edgeStencil[i][3]]))
                    { 
                        threads.Append(nonDBCECount++);
                    }
                    else {
                        threads.Append(-1);
                    }
                }

                const T k = bendingStiffMult * std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::P>(elemAttr.Get_Unchecked(0))(0, 0);
                int newTripletStartI = triplets.size();
                triplets.resize(newTripletStartI + nonDBCECount * 12 * 12);
                threads.Par_Each([&](int eI, auto data) {
                    auto& [nonDBCEI] = data;
                    if (nonDBCEI >= 0) {
                        const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][0]));
                        const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][1]));
                        const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][2]));
                        const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(edgeStencil[eI][3]));
                        Eigen::Matrix<T, dim, 1> x0e(x0.data), x1e(x1.data), x2e(x2.data), x3e(x3.data);

                        const T thetabar = edgeInfo[eI][0];
                        const T ebarnorm = edgeInfo[eI][1];
                        const T hbar = edgeInfo[eI][2];

                        T theta;
                        Compute_Dihedral_Angle(x0e, x1e, x2e, x3e, theta);
                        Eigen::Matrix<T, 12, 1> grad;
                        Compute_Dihedral_Angle_Gradient(x0e, x1e, x2e, x3e, grad);
                        Eigen::Matrix<T, 12, 12> H;
                        Compute_Dihedral_Angle_Hessian(x0e, x1e, x2e, x3e, H);

                        H *= h * h * k * 2 * (theta - thetabar) * ebarnorm / hbar;
                        H += ((h * h * k * 2 * ebarnorm / hbar) * grad) * grad.transpose();
                        if (projectSPD) { 
                            makePD(H);
                        }

                        T indMap[12];
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                indMap[i * 3 + j] = edgeStencil[eI][i] * 3 + j;
                            }
                        }
                        for (int i = 0; i < 12; ++i) {
                            int startI = newTripletStartI + nonDBCEI * 12 * 12 + i * 12;
                            for (int j = 0; j < 12; ++j) {
                                triplets[startI + j] = std::move(Eigen::Triplet<T>(indMap[i], indMap[j], H(i, j)));
                            }
                        }
                    }
                });
            }
        }
    }
}

template <class T, int dim, bool KL>
void Check_Bending_Gradient(
    MESH_ELEM<dim - 1>& Elem,
    const VECTOR_STORAGE<T, dim + 1>& DBC, T h, 
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    T thickness, MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    T eps = 1.0e-6;

    T E0 = 0;
    Compute_Bending_Energy<T, dim, KL>(Elem, 1.0, edge2tri, edgeStencil, edgeInfo, thickness, T(1), std::vector<bool>(X.size, false), X, nodeAttr, elemAttr, elasticityAttr, E0); 
    nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Bending_Gradient<T, dim, KL>(Elem, 1.0, edge2tri, edgeStencil, edgeInfo, thickness, T(1), std::vector<bool>(X.size, false), X, nodeAttr, elemAttr, elasticityAttr); 

    std::vector<T> grad_FD(X.size * dim);
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        T E = 0;
        Compute_Bending_Energy<T, dim, KL>(Elem, 1.0, edge2tri, edgeStencil, edgeInfo, thickness, T(1), std::vector<bool>(X.size, false), Xperturb, nodeAttr, elemAttr, elasticityAttr, E);
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

template <class T, int dim, bool KL>
void Check_Bending_Hessian(
    MESH_ELEM<dim - 1>& Elem,
    const VECTOR_STORAGE<T, dim + 1>& DBC, T h, 
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    T thickness, MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix, #row = 4 * (#segments + 1)
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    T eps = 1.0e-8;

    MESH_NODE_ATTR<T, dim> nodeAttr0;
    nodeAttr.deep_copy_to(nodeAttr0);
    nodeAttr0.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
    Compute_Bending_Gradient<T, dim, KL>(Elem, 1.0, edge2tri, edgeStencil, edgeInfo, thickness, T(1), std::vector<bool>(X.size, false), X, nodeAttr0, elemAttr, elasticityAttr);
    std::vector<Eigen::Triplet<T>> HStriplets;
    Compute_Bending_Hessian<T, dim, KL>(Elem, 1.0, false, edge2tri, edgeStencil, edgeInfo, thickness, T(1), std::vector<bool>(X.size, false), X, nodeAttr, elemAttr, elasticityAttr, HStriplets); 
    CSR_MATRIX<T> HS;
    HS.Construct_From_Triplet(X.size * dim, X.size * dim, HStriplets);

    std::vector<Eigen::Triplet<T>> HFDtriplets;
    HFDtriplets.reserve(HStriplets.size());
    for (int i = 0; i < X.size * dim; ++i) {
        MESH_NODE<T, dim> Xperturb;
        Append_Attribute(X, Xperturb);
        std::get<0>(Xperturb.Get_Unchecked(i / dim))[i % dim] += eps;
        
        nodeAttr.template Fill<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(VECTOR<T, dim>(0));
        Compute_Bending_Gradient<T, dim, KL>(Elem, 1.0, edge2tri, edgeStencil, edgeInfo, thickness, T(1), std::vector<bool>(X.size, false), Xperturb, nodeAttr, elemAttr, elasticityAttr);
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