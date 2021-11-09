#pragma once
//#####################################################################
// Function Project_Strain_NACC
//#####################################################################
#include <Math/VECTOR.h>
#include <Math/MATH_TOOLS.h>

namespace py = pybind11;
namespace JGSL {

//#####################################################################
// Function Project_Strain_NACC
//#####################################################################
template <class T, int dim>
void Project_Strain_NACC(NEOHOOKEAN_BORDEN<T, dim>& nhb) {
    T friction_angle = 45.0, beta = 0.8, xi = 1.0; //logJp is initialized at log(1) = 0
    bool hardeningOn = true, qHard = true;
    nhb.Each([&](const int i, auto data) {
        auto& [F, logJp, vol, lambda, mu, kappa, g] = data;
        //Computed parameters
        T sin_phi = sin(friction_angle / (T)180 * (T)3.141592653);
        T mohr_columb_friction = sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
        T M = mohr_columb_friction * (T)dim / sqrt((T)2 / ((T)6 - dim));
        //SVD to grab sigmas
        MATRIX<T, dim> U(1), V(1);
        VECTOR<T, dim> sigma;
        Singular_Value_Decomposition(F, U, sigma, V);

        //compute p0
        T p0 = kappa * (T(0.00001) + sinh(xi * std::max(-logJp, (T)0))); //must compute p0 every time we project strain for a particle

        //compute J^{e,tr} and BhatTrial
        T J = 1.0;
        VECTOR<T,dim> B_hat_trial;
        for (int i = 0; i < dim; ++i){
            J *= sigma(i);
            B_hat_trial(i) = sigma(i) * sigma(i);
        } 

        //Compute sHatTrial
        VECTOR<T,dim> s_hat_trial = mu * pow(J, -(T)2 / (T)dim) * MATH_TOOLS::Deviatoric(B_hat_trial);

        //Uprime and pTrial
        T prime = kappa / (T)2 * (J - 1 / J);
        T p_trial = -prime * J;

        //Cases 1 and 2 (Ellipsoid tips)
        //Project to the tips
        T pMin = beta * p0;
        T pMax = p0;
        if (p_trial > pMax) {
            T Je_new = sqrt(-2 * pMax / kappa + 1);
            T sigma_new = pow(Je_new, (T)1 / dim);
            MATRIX<T,dim> sigma_m(sigma_new); //diag matrix with new sigma
            MATRIX<T,dim> Fe = U * sigma_m * V.transpose();
            F = Fe;
            if (hardeningOn) {
                logJp += log(J / Je_new);
            }
            return;
        }
        else if (p_trial < -pMin) {
            T Je_new = sqrt(2 * pMin / kappa + 1);
            T sigma_new = pow(Je_new, (T)1 / dim);
            MATRIX<T,dim> sigma_m(sigma_new); //diag matrix with new sigma
            MATRIX<T,dim> Fe = U * sigma_m * V.transpose();
            F = Fe;
            if (hardeningOn) {
                logJp += log(J / Je_new);
            }
            return;
        }

        //Case 3 --> check if inside YS (a) or not (b)
        T y_s_half_coeff = ((T)6 - dim) / (T)2 * ((T)1 + (T)2 * beta);
        T y_p_half = M * M * (p_trial + pMin) * (p_trial - pMax);
        T y = y_s_half_coeff * s_hat_trial.length2() + y_p_half;

        //Case 3a (inside YS, no projection)
        if (y < 1e-4) return;

        //Case 3b Step 1: outside YS, return to surface
        VECTOR<T,dim> B_hat_new = pow(J, (T)2 / (T)dim) / mu * sqrt(-y_p_half / y_s_half_coeff) * s_hat_trial / (s_hat_trial.length());
        T Bsum = 0;
        for(int i = 0; i < dim; ++i)
            Bsum += B_hat_trial(i);
        B_hat_new += (T)1 / dim * Bsum * (VECTOR<T,dim>(1.0));
        VECTOR<T,dim> sigma_new;
        for (int i = 0; i < dim; ++i)
            sigma_new(i) = sqrt(B_hat_new(i));
        MATRIX<T,dim> sigma_m(sigma_new);
        MATRIX<T,dim> Fe = U * sigma_m * V.transpose();
        F = Fe;

        //Case 3b Step 2: Hardening
        //Three approaches to hardening:
        //0 -> hack the hardening by computing a fake delta_p
        //1 -> q based
        if (p0 > 1e-4 && p_trial < pMax - 1e-4 && p_trial > 1e-4 - pMin) {
            T dAlpha = 0, dOmega = 0;
            
            T p_center = p0 * ((1 - beta) / (T)2);
            T q_trial = sqrt(((T)6 - (T)dim) / (T)2) * s_hat_trial.length();
            VECTOR<T, 2> direction;
            direction(0) = p_center - p_trial;
            direction(1) = 0 - q_trial;
            direction = direction / (direction.length());

            T C = M * M * (p_center + beta * p0) * (p_center - p0);
            T B = M * M * direction(0) * (2 * p_center - p0 + beta * p0);
            T A = M * M * direction(0) * direction(0) + (1 + 2 * beta) * direction(1) * direction(1);

            T l1 = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
            T l2 = (-B - sqrt(B * B - 4 * A * C)) / (2 * A);

            T p1 = p_center + l1 * direction(0);
            T p2 = p_center + l2 * direction(0);
            T p_fake = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;

            //Only for pFake Hardening
            T Je_new_fake = sqrt(abs(-2 * p_fake / kappa + 1));
            dAlpha = log(J / Je_new_fake);

            //Only for q Hardening
            T qNPlus = sqrt(M * M * (p_trial + pMin) * (pMax - p_trial) / ((T)1 + (T)2 * beta));
            T Jtrial = J;
            T zTrial = sqrt(((q_trial * pow(Jtrial, ((T)2 / (T)dim))) / (mu * sqrt(((T)6 - (T)dim) / (T)2))) + 1);
            T zNPlus = sqrt(((qNPlus * pow(Jtrial, ((T)2 / (T)dim))) / (mu * sqrt(((T)6 - (T)dim) / (T)2))) + 1);
            if (p_trial > p_fake) {
                dOmega = -1 * log(zTrial / zNPlus);
            }
            else {
                dOmega = log(zTrial / zNPlus);
            }

            if (hardeningOn) {
                if (!qHard) {
                    if (Je_new_fake > 1e-4) {
                        logJp += dAlpha;
                    }
                }
                else if (qHard) {
                    if (zNPlus > 1e-4) {
                        logJp += dOmega;
                    }
                }
            }
        }
    });
}

}
