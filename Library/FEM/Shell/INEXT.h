#pragma once

namespace JGSL {

template <class T>
T f_sl_u(T s, T slim, T sHat)
{
#ifdef C1SL
    return (sHat - s) * std::log((slim - s) / (slim - sHat));
#else
    return -(sHat - s) * (sHat - s) * std::log((slim - s) / (slim - sHat)) / ((slim - sHat) * (slim - sHat));
#endif
}

template <class T>
T g_sl_u(T s, T slim, T sHat)
{
#ifdef C1SL
    T temp = (slim - sHat) / (slim - s);
    return std::log(temp) + temp - 1;
#else
    T a = sHat - s;
    return (2 * a * std::log((slim - s) / (slim - sHat)) + a * a / (slim - s)) / ((slim - sHat) * (slim - sHat));
#endif
}

template <class T>
T h_sl_u(T s, T slim, T sHat)
{
#ifdef C1SL
    T temp = 1 / (slim - s);
    return (1 + (slim - sHat) * temp) * temp;
#else
    T a = sHat - s;
    T b = slim - s;
    T c = a / b;
    return (-2 * std::log(b / (slim - sHat)) + c * (c - 4)) / ((slim - sHat) * (slim - sHat));
#endif
}

template <class T>
T f_sl_l(T s, T slim, T sHat)
{
#ifdef C1SL
    return -f_sl_u(s, slim, sHat);
#else
    return f_sl_u(s, slim, sHat);
#endif
}

template <class T>
T g_sl_l(T s, T slim, T sHat)
{
#ifdef C1SL
    return -g_sl_u(s, slim, sHat);
#else
    return g_sl_u(s, slim, sHat);
#endif
}

template <class T>
T h_sl_l(T s, T slim, T sHat)
{
#ifdef C1SL
    return -h_sl_u(s, slim, sHat);
#else
    return h_sl_u(s, slim, sHat);
#endif
}

template<class T, int dim>
void Compute_Inextensibility(
    MESH_ELEM<dim - 1>& Elem, T h,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    std::vector<VECTOR<T, 2>>& strain)
{
    strain.resize(Elem.size);
    Elem.Join(elemAttr).Par_Each([&](int id, auto data) {
        auto &[elemVInd, Bsqr, P] = data;
        if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
            const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
            const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
            const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));

            Eigen::Matrix<T, dim - 1, dim - 1> B;
            B(0, 0) = std::sqrt(Bsqr(0, 0));
            B(1, 0) = 0;
            B(0, 1) = Bsqr(0, 1) / B(0, 0);
            B(1, 1) = std::sqrt(Bsqr(1, 1) - Bsqr(0, 1) * Bsqr(0, 1) / Bsqr(0, 0));
            Eigen::Matrix<T, dim, dim - 1> A;
            for (int d = 0; d < 3; ++d) { 
                A(d, 0) = x2[d] - x1[d];
                A(d, 1) = x3[d] - x1[d];
            }
            Eigen::Matrix<T, dim, dim - 1> F = A * B.inverse();

            Eigen::JacobiSVD<Eigen::Matrix<T, dim, dim - 1>> svd(F, 0); // matrix U and V not needed for energy evaluation
            strain[id][0] = svd.singularValues()[0];
            strain[id][1] = svd.singularValues()[1];
            if (strain[id][0] < svd.singularValues()[1]) {
                strain[id][0] = svd.singularValues()[1];
                strain[id][1] = svd.singularValues()[0];
            }
        }
        else {
            strain[id][0] = strain[id][1] = 1;
        }
    });
}

template<class T, int dim>
bool Compute_Inextensibility_Energy(
    MESH_ELEM<dim - 1>& Elem, T h,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    T& E)
{
    TIMER_FLAG("Compute_Inextensibility_Energy");
    bool valid = true;
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F_, vol, lambda, mu] = data;
            if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                const MATRIX<T, dim - 1>& Bsqr = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));

                Eigen::Matrix<T, dim - 1, dim - 1> B;
                B(0, 0) = std::sqrt(Bsqr(0, 0));
                B(1, 0) = 0;
                B(0, 1) = Bsqr(0, 1) / B(0, 0);
                B(1, 1) = std::sqrt(Bsqr(1, 1) - Bsqr(0, 1) * Bsqr(0, 1) / Bsqr(0, 0));
                Eigen::Matrix<T, dim, dim - 1> A;
                for (int d = 0; d < 3; ++d) { 
                    A(d, 0) = x2[d] - x1[d];
                    A(d, 1) = x3[d] - x1[d];
                }
                Eigen::Matrix<T, dim, dim - 1> F = A * B.inverse();

                Eigen::JacobiSVD<Eigen::Matrix<T, dim, dim - 1>> svd(F, 0); // matrix U and V not needed for energy evaluation
                T stiffness = h * h * vol * kappa_s[0];
                for (int i = 0; i < dim - 1; ++i) {
                    if (svd.singularValues()[i] >= s[0]) {
                        valid = false;
                    }
                    else if (svd.singularValues()[i] > sHat[0]) {
                        E += stiffness * f_sl_u(svd.singularValues()[i], s[0], sHat[0]);
                    }
                }

                if (kappa_s[1] > 0) {
                    stiffness = h * h * vol * kappa_s[1];
                    for (int i = 0; i < dim - 1; ++i) {
                        if (svd.singularValues()[i] <= s[1]) {
                            valid = false;
                        }
                        else if (svd.singularValues()[i] < sHat[1]) {
                            E += stiffness * f_sl_l(svd.singularValues()[i], s[1], sHat[1]);
                        }
                    }
                }
            }
        });
    }
    return valid;
}

template<class T, int dim>
void Compute_Inextensibility_Gradient(
    MESH_ELEM<dim - 1>& Elem, T h,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr)
{
    TIMER_FLAG("Compute_Inextensibility_Gradient");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        //TODO: parallelize
        Elem.Join(elasticityAttr).Each([&](int id, auto data) {
            auto &[elemVInd, F_, vol, lambda, mu] = data;
            if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                const MATRIX<T, dim - 1>& Bsqr = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));

                Eigen::Matrix<T, dim - 1, dim - 1> B;
                B(0, 0) = std::sqrt(Bsqr(0, 0));
                B(1, 0) = 0;
                B(0, 1) = Bsqr(0, 1) / B(0, 0);
                B(1, 1) = std::sqrt(Bsqr(1, 1) - Bsqr(0, 1) * Bsqr(0, 1) / Bsqr(0, 0));
                Eigen::Matrix<T, dim - 1, dim - 1> IB = B.inverse();
                Eigen::Matrix<T, dim, dim - 1> A;
                for (int d = 0; d < 3; ++d) { 
                    A(d, 0) = x2[d] - x1[d];
                    A(d, 1) = x3[d] - x1[d];
                }
                Eigen::Matrix<T, dim, dim - 1> F = A * IB;

                Eigen::JacobiSVD<Eigen::Matrix<T, dim, dim - 1>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix<T, dim, dim - 1> db_div_dF;
                db_div_dF.setZero();
                T stiffness = h * h * vol * kappa_s[0];
                bool triggered = false;
                for (int i = 0; i < dim - 1; ++i) {
                    if (svd.singularValues()[i] > sHat[0]) {
                        triggered = true;

                        T db_div_dS = stiffness * g_sl_u(svd.singularValues()[i], s[0], sHat[0]);
                        db_div_dF += svd.matrixU().col(i) * (db_div_dS * svd.matrixV().col(i).transpose());
                    }
                }

                if (kappa_s[1] > 0) {
                    stiffness = h * h * vol * kappa_s[1];
                    for (int i = 0; i < dim - 1; ++i) {
                        if (svd.singularValues()[i] < sHat[1]) {
                            triggered = true;

                            T db_div_dS = stiffness * g_sl_l(svd.singularValues()[i], s[1], sHat[1]);
                            db_div_dF += svd.matrixU().col(i) * (db_div_dS * svd.matrixV().col(i).transpose());
                        }
                    }
                }

                if (triggered) {
                    Eigen::Matrix<T, dim * 3, 1> db_div_dx;
                    db_div_dx.template segment<dim>(0) = -(IB(0, 0) + IB(1, 0)) * db_div_dF.col(0) - (IB(0, 1) + IB(1, 1)) * db_div_dF.col(1);
                    db_div_dx.template segment<dim>(3) = IB(0, 0) * db_div_dF.col(0) + IB(0, 1) * db_div_dF.col(1);
                    db_div_dx.template segment<dim>(6) = IB(1, 0) * db_div_dF.col(0) + IB(1, 1) * db_div_dF.col(1);
                    for (int v = 0; v < 3; ++v) {
                        VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(elemVInd[v]));
                        for (int d = 0; d < dim; ++d) {
                            g[d] += db_div_dx[v * dim + d];
                        }
                    }
                }
            }
        });
    }
}

template <class T>
void Compute_DU_And_DV_Div_DF(const Eigen::JacobiSVD<Eigen::Matrix<T, 3, 2>>& svd,
    Eigen::Matrix<T, 9, 6>& dU_div_dF, Eigen::Matrix<T, 4, 6>& dV_div_dF)
{
    Eigen::Matrix<T, 3, 3> V;
    V.setZero();
    V.template block<2, 2>(0, 0) = svd.matrixV();
    Eigen::Matrix<T, 3, 1> sigma;
    sigma.template segment<2>(0) = svd.singularValues();
    sigma[2] = 0;
    Eigen::Matrix<T, 9, 9> dU_div_dF_ext, dV_div_dF_ext;
    dU_div_dF_ext.setZero();
    dV_div_dF_ext.setZero();
    for (int cI = 0; cI < 3; ++cI) {
        int cI_post = (cI + 1) % 3;

        Eigen::Matrix<T, 2, 2> coefMtr;
        if (std::abs(sigma[cI] - sigma[cI_post]) < 2e-6) {
            coefMtr << sigma[cI] + 1e-6, sigma[cI_post] - 1e-6,
                sigma[cI_post] - 1e-6, sigma[cI] + 1e-6;
        }
        else {
            coefMtr << sigma[cI], sigma[cI_post],
                sigma[cI_post], sigma[cI];
        }
        const Eigen::FullPivLU<Eigen::Matrix<T, 2, 2>> solver(coefMtr);

        Eigen::Matrix<T, 2, 1> b;
        for (int rowI = 0; rowI < 3; ++rowI) {
            for (int colI = 0; colI < 3; ++colI) {
                b << svd.matrixU()(rowI, cI_post) * V(colI, cI),
                    -svd.matrixU()(rowI, cI) * V(colI, cI_post);
                const Eigen::Matrix<T, 2, 1> wij21 = solver.solve(b);

                dU_div_dF_ext.template block<3, 1>(cI * 3, rowI + colI * 3) += wij21[0] * svd.matrixU().col(cI_post);
                dU_div_dF_ext.template block<3, 1>(cI_post * 3, rowI + colI * 3) += -wij21[0] * svd.matrixU().col(cI);

                dV_div_dF_ext.template block<3, 1>(cI * 3, rowI + colI * 3) += -wij21[1] * V.col(cI_post);
                dV_div_dF_ext.template block<3, 1>(cI_post * 3, rowI + colI * 3) += wij21[1] * V.col(cI);
            }
        }
    }
    dU_div_dF = dU_div_dF_ext.leftCols(6);
    dV_div_dF.template block<2, 6>(0, 0) = dV_div_dF_ext.template block<2, 6>(0, 0);
    dV_div_dF.template block<2, 6>(2, 0) = dV_div_dF_ext.template block<2, 6>(3, 0);
}

template<class T, int dim>
void Compute_Inextensibility_Hessian(
    MESH_ELEM<dim - 1>& Elem, 
    T h, bool projectSPD,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
    const std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    std::vector<Eigen::Triplet<T>>& triplets)
{
    TIMER_FLAG("Compute_Inextensibility_Hessian");
    if constexpr (dim == 2) {
        //TODO
    }
    else {
        std::vector<Eigen::JacobiSVD<Eigen::Matrix<T, dim, dim - 1>>> svds(Elem.size);
        std::vector<Eigen::Matrix<T, dim - 1, dim - 1>> IBs(Elem.size);
        Elem.Par_Each([&](int id, auto data) {
            auto &[elemVInd] = data;
            if (!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) {
                const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
                const VECTOR<T, dim>& x2 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
                const VECTOR<T, dim>& x3 = std::get<0>(X.Get_Unchecked(elemVInd[2]));
                const MATRIX<T, dim - 1>& Bsqr = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked(id));

                Eigen::Matrix<T, dim - 1, dim - 1> B;
                B(0, 0) = std::sqrt(Bsqr(0, 0));
                B(1, 0) = 0;
                B(0, 1) = Bsqr(0, 1) / B(0, 0);
                B(1, 1) = std::sqrt(Bsqr(1, 1) - Bsqr(0, 1) * Bsqr(0, 1) / Bsqr(0, 0));
                IBs[id] = B.inverse();
                Eigen::Matrix<T, dim, dim - 1> A;
                for (int d = 0; d < 3; ++d) { 
                    A(d, 0) = x2[d] - x1[d];
                    A(d, 1) = x3[d] - x1[d];
                }
                Eigen::Matrix<T, dim, dim - 1> F = A * IBs[id];

                svds[id].compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            }
        });

        std::vector<int> elemTripletStartInd(svds.size());
        int tripletStartInd = triplets.size();
        for (int i = 0; i < svds.size(); ++i) {
            const VECTOR<int, dim>& elemVInd = std::get<0>(Elem.Get_Unchecked(i));
            if ((!DBCb[elemVInd[0]] || !DBCb[elemVInd[1]] || !DBCb[elemVInd[2]]) &&
                (svds[i].singularValues()[0] > sHat[0] || svds[i].singularValues()[1] > sHat[0] ||
                (kappa_s[1] > 0 && (svds[i].singularValues()[0] < sHat[1] || svds[i].singularValues()[1] < sHat[1]))))
            {
                elemTripletStartInd[i] = tripletStartInd;
                tripletStartInd += 81;
            }
            else {
                elemTripletStartInd[i] = -1;
            }
        }
        triplets.resize(tripletStartInd);

        Elem.Join(elasticityAttr).Par_Each([&](int id, auto data) {
            auto &[elemVInd, F_, vol, lambda, mu] = data;

            int startInd = elemTripletStartInd[id];
            if (startInd >= 0) {
                Eigen::Matrix<T, 9, 6> dU_div_dF;
                Eigen::Matrix<T, 4, 6> dV_div_dF;
                Compute_DU_And_DV_Div_DF(svds[id], dU_div_dF, dV_div_dF);

                Eigen::Matrix<T, dim * (dim - 1), dim * (dim - 1)> d2b_div_dF2;
                d2b_div_dF2.setZero();
                T stiffness = h * h * vol * kappa_s[0];
                for (int i = 0; i < dim - 1; ++i) {
                    if (svds[id].singularValues()[i] > sHat[0]) {
                        T d2b_div_dS2 = stiffness * h_sl_u(svds[id].singularValues()[i], s[0], sHat[0]);
                        Eigen::Matrix<T, dim, dim - 1> dS_div_dF = svds[id].matrixU().col(i) * svds[id].matrixV().col(i).transpose();
                        Eigen::Matrix<T, dim * (dim - 1), 1> dS_div_dF_reshaped;
                        dS_div_dF_reshaped.template segment<dim>(0) = dS_div_dF.col(0);
                        dS_div_dF_reshaped.template segment<dim>(dim) = dS_div_dF.col(1);
                        d2b_div_dF2 += (dS_div_dF_reshaped * d2b_div_dS2) * dS_div_dF_reshaped.transpose();
                        
                        Eigen::Matrix<T, 6, 6> d2S_div_dF2;
                        for (int Fij = 0; Fij < 6; ++Fij) {
                            Eigen::Matrix<T, 3, 2> d2S_div_dF2ij = dU_div_dF.template block<3, 1>(i * 3, Fij) * svds[id].matrixV().col(i).transpose() + 
                                svds[id].matrixU().col(i) * dV_div_dF.template block<2, 1>(i * 2, Fij).transpose();
                            d2S_div_dF2.template block<3, 1>(0, Fij) = d2S_div_dF2ij.col(0);
                            d2S_div_dF2.template block<3, 1>(3, Fij) = d2S_div_dF2ij.col(1);
                        }
                        T db_div_dS = stiffness * g_sl_u(svds[id].singularValues()[i], s[0], sHat[0]);
                        d2b_div_dF2 += db_div_dS * d2S_div_dF2;
                    }
                }

                if (kappa_s[1] > 0) {
                    T stiffness = h * h * vol * kappa_s[1];
                    for (int i = 0; i < dim - 1; ++i) {
                        if (svds[id].singularValues()[i] < sHat[1]) {
                            T d2b_div_dS2 = stiffness * h_sl_l(svds[id].singularValues()[i], s[1], sHat[1]);

                            Eigen::Matrix<T, dim, dim - 1> dS_div_dF = svds[id].matrixU().col(i) * svds[id].matrixV().col(i).transpose();
                            Eigen::Matrix<T, dim * (dim - 1), 1> dS_div_dF_reshaped;
                            dS_div_dF_reshaped.template segment<dim>(0) = dS_div_dF.col(0);
                            dS_div_dF_reshaped.template segment<dim>(dim) = dS_div_dF.col(1);
                            d2b_div_dF2 += (dS_div_dF_reshaped * d2b_div_dS2) * dS_div_dF_reshaped.transpose();
                            
                            Eigen::Matrix<T, 6, 6> d2S_div_dF2;
                            for (int Fij = 0; Fij < 6; ++Fij) {
                                Eigen::Matrix<T, 3, 2> d2S_div_dF2ij = dU_div_dF.template block<3, 1>(i * 3, Fij) * svds[id].matrixV().col(i).transpose() + 
                                    svds[id].matrixU().col(i) * dV_div_dF.template block<2, 1>(i * 2, Fij).transpose();
                                d2S_div_dF2.template block<3, 1>(0, Fij) = d2S_div_dF2ij.col(0);
                                d2S_div_dF2.template block<3, 1>(3, Fij) = d2S_div_dF2ij.col(1);
                            }
                            T db_div_dS = stiffness * g_sl_l(svds[id].singularValues()[i], s[1], sHat[1]);
                            d2b_div_dF2 += db_div_dS * d2S_div_dF2;
                        }
                    }
                }

                if (projectSPD) {
                    makePD(d2b_div_dF2);
                }

                const Eigen::Matrix<T, dim - 1, dim - 1>& IB = IBs[id];
                Eigen::Matrix<T, 9, 6> intermediate;
                for (int colI = 0; colI < 6; ++colI) {
                    intermediate.col(colI).template segment<dim>(0) = -(IB(0, 0) + IB(1, 0)) * d2b_div_dF2.col(colI).template segment<dim>(0) - 
                        (IB(0, 1) + IB(1, 1)) * d2b_div_dF2.col(colI).template segment<dim>(3);
                    intermediate.col(colI).template segment<dim>(3) = IB(0, 0) * d2b_div_dF2.col(colI).template segment<dim>(0) + 
                        IB(0, 1) * d2b_div_dF2.col(colI).template segment<dim>(3);
                    intermediate.col(colI).template segment<dim>(6) = IB(1, 0) * d2b_div_dF2.col(colI).template segment<dim>(0) + 
                        IB(1, 1) * d2b_div_dF2.col(colI).template segment<dim>(3);
                }
                Eigen::Matrix<T, 9, 9> Hessian;
                for (int i = 0; i < 9; ++i) {
                    Hessian.row(i).template segment<dim>(0) = -(IB(0, 0) + IB(1, 0)) * intermediate.row(i).template segment<dim>(0) - 
                        (IB(0, 1) + IB(1, 1)) * intermediate.row(i).template segment<dim>(3);
                    Hessian.row(i).template segment<dim>(3) = IB(0, 0) * intermediate.row(i).template segment<dim>(0) + 
                        IB(0, 1) * intermediate.row(i).template segment<dim>(3);
                    Hessian.row(i).template segment<dim>(6) = IB(1, 0) * intermediate.row(i).template segment<dim>(0) + 
                        IB(1, 1) * intermediate.row(i).template segment<dim>(3);
                }
                
                for (int i = 0; i < 3; ++i) {
                    for (int id = 0; id < 3; ++id) {
                        for (int j = 0; j < 3; ++j) {
                            for (int jd = 0; jd < 3; ++jd) {
                                triplets[startInd + (i * 3 + id) * 9 + j * 3 + jd] = std::move(
                                    Eigen::Triplet<T>(elemVInd[i] * 3 + id, elemVInd[j] * 3 + jd, 
                                    Hessian(i * 3 + id, j * 3 + jd)));
                            }
                        }
                    }
                }
            }
        });
    }
}

}