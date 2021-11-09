#pragma once

namespace JGSL {

template <class T, int dim = 3>
VECTOR<int, 4> Add_Discrete_Particles(
    const VECTOR<T, dim>& len,
    const VECTOR<int, dim>& num,
    const VECTOR<T, dim>& randScale,
    MESH_NODE<T, dim>& X,
    std::vector<int>& particle,
    std::vector<int>& compNodeRange)
{
    VECTOR<T, dim> step = len;
    for (int d = 0; d < dim; ++d) {
        if (num[d] > 1) {
            step[d] /= num[d] - 1;
        }
    }

    VECTOR<int, 4> counter;
    counter[0] = X.size;
    counter[1] = particle.size();

    std::srand(0);
    particle.reserve(particle.size() + num.prod());
    for (int i = 0; i < num[0]; ++i) {
        for (int j = 0; j < num[1]; ++j) {
            for (int k = 0; k < num[2]; ++k) {
                T randNum[3] = {
                    (T)rand() / RAND_MAX * 2 - 1,
                    (T)rand() / RAND_MAX * 2 - 1,
                    (T)rand() / RAND_MAX * 2 - 1
                };

                particle.emplace_back(X.size);
                X.Append(VECTOR<T, dim>(
                    -len[0] / 2 + step[0] * i + randNum[0] * randScale[0], 
                    -len[1] / 2 + step[1] * j + randNum[1] * randScale[1], 
                    -len[2] / 2 + step[2] * k + randNum[2] * randScale[2]));
            }
        }
    }

    counter[2] = X.size;
    counter[3] = particle.size();
    compNodeRange.emplace_back(X.size);
    return counter;
}

template <class T, int dim = 3>
void Initialize_Discrete_Particle(
    MESH_NODE<T, dim>& X,
    const std::vector<int>& particle,
    T rho, T E, T thickness, 
    const VECTOR<T, dim>& gravity,
    std::vector<T>& b,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M) // mass matrix
{
    std::vector<Eigen::Triplet<T>> triplets;
    const T mass = 4.0 / 3.0 * M_PI * thickness * thickness * thickness / 8 * rho;
    for (const auto& vI : particle) {
        std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::m>(nodeAttr.Get_Unchecked(vI)) += mass;
        for (int dimI = 0; dimI < dim; ++dimI) {
            triplets.emplace_back(vI * dim + dimI, vI * dim + dimI, mass);
            b[vI * dim + dimI] += mass * gravity[dimI];
        }
    }
    CSR_MATRIX<T> M_add;
    M_add.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
    M.Get_Matrix() += M_add.Get_Matrix();

    //TODO: EIPC volume
}

}