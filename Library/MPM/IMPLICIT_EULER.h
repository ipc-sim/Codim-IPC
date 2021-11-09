#pragma once

#include <unordered_map>
#include <unordered_set>
#include <Math/BSPLINES.h>
#include <MPM/DATA_TYPE.h>
#include <MPM/COLOR.h>
#include <Grid/SPARSE_GRID.h>
#include <Utils/LOGGING.h>
#include <Utils/PROFILER.h>
#include <Utils/COLLIDER.h>
#include <Storage/storage.hpp>
#include <Math/VECTOR.h>
#include <Math/CSR_MATRIX.h>
#include <Math/AMGCL_SOLVER.h>
#include <Physics/CONSTITUTIVE_MODEL.h>

namespace JGSL {

template <class T, int dim>
int Label_DOF(SPARSE_GRID<VECTOR<T, dim + 1>, dim> &grid, SPARSE_GRID<int, dim>& dof)
{
    TIMER_FLAG("Lable DOF");
    int index = 0;
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        if (g(dim) != 0) dof(node) = index ++;
        else dof(node) = -1; // invalid nodes.
    });
    std::cout << "Num nodes: " << index << std::endl;
    return index;
}

template<int dim>
int Linear_Offset(const VECTOR<int, dim>& node_offset)
{
    int grid_range = 5;
    auto rel_offset = node_offset + VECTOR<int, dim>(2); // move (-2, -2) to (0, 0)
    if constexpr (dim == 2)
        return rel_offset(0) * grid_range + rel_offset(1);
    else if constexpr (dim == 3)
        return ((rel_offset(0) * grid_range) + rel_offset(1)) * grid_range + rel_offset(2);
}

template <class T, int dim>
T Elastic_Energy(T weight) {return 0.0;}
template <class T, int dim, class CONSTITUTIVE_MODEL_FUNCTOR, class ... OTHER_FUNCTOR>
T Elastic_Energy(T weight, typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttr, typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrOther) {
    T potential = 0.0;
    CONSTITUTIVE_MODEL_FUNCTOR::Compute_Psi(elasticityAttr, weight, potential);
    return potential + Elastic_Energy<T, dim, OTHER_FUNCTOR ...>(weight, elasticityAttrOther ...);
}

template <class T, int dim, class ... CONSTITUTIVE_MODEL_FUNCTOR>
T Compute_Potential(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>>& dv,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE ... elasticityAttr, 
    const VECTOR<T, dim>& gravity,
    T dt
) {
    T potential = 0.0; 
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        if(dof(node) == -1) return;
        const auto &dvI = dv.template Get_Component_Unchecked<0>(dof(node));
        potential += 0.5 * g(dim) * dvI.length2() - dt * g(dim) * gravity.dot(dvI);
    });
    return potential + Elastic_Energy<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(T(1.), elasticityAttr ...);
}

template <class T, int dim>
void Add_Elastic_Force(SPARSE_GRID<int, dim>&, SPARSE_GRID<VECTOR<T, dim + 1>, dim>&, MPM_PARTICLES<T, dim>&, BASE_STORAGE<VECTOR<T, dim>>&, T, T) {}
template <class T, int dim, class CONSTITUTIVE_MODEL_FUNCTOR, class ... OTHER_FUNCTOR>
void Add_Elastic_Force(
    SPARSE_GRID<int, dim>& dof,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttr,
    typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrOther,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttrMoved,
    typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrMovedOther,
    MPM_PARTICLES<T, dim>& particles,
    BASE_STORAGE<VECTOR<T, dim>> &force,
    T dx, T weight
) {
    BASE_STORAGE<MATRIX<T, dim>> dF;
    BASE_STORAGE<T> dJ;
    if constexpr(CONSTITUTIVE_MODEL_FUNCTOR::useJ)
        CONSTITUTIVE_MODEL_FUNCTOR::Compute_First_PiolaKirchoff_Stress(elasticityAttrMoved, weight, dJ);
    else
        CONSTITUTIVE_MODEL_FUNCTOR::Compute_First_PiolaKirchoff_Stress(elasticityAttrMoved, weight, dF);
    Colored_Par_Each(particles, grid, dx, [&](const int i) {
        if (!elasticityAttrMoved.contains(i)) return;
        auto& X = particles.template Get_Component_Unchecked<0>(i);
        BSPLINE_WEIGHTS<T, dim, 2> spline(X, dx);
        for (auto [node, w, dw, g] : grid.Iterate_Kernel(spline)) {
            if (dof(node) == -1) continue;
            auto& rI = force.template Get_Component_Unchecked<0>(dof(node));
            if constexpr (CONSTITUTIVE_MODEL_FUNCTOR::useJ)
                rI -= dJ.template Get_Component_Unchecked<0>(i) * elasticityAttr.template Get_Component_Unchecked<0>(i) * dw;
            else
                rI -= dF.template Get_Component_Unchecked<0>(i) * elasticityAttr.template Get_Component_Unchecked<0>(i).transpose() * dw;
        }
    });
    Add_Elastic_Force<T, dim, OTHER_FUNCTOR ...>(dof, grid, elasticityAttrOther ..., elasticityAttrMovedOther ..., particles, force, dx, weight);
}

template <class T, int dim, class ... CONSTITUTIVE_MODEL_FUNCTOR>
void Compute_Gradient(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>> dv,
    MPM_PARTICLES<T, dim>& particles,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttr,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttrMoved,
    BASE_STORAGE<VECTOR<T, dim>>& gradient, 
    const VECTOR<T, dim>& gravity,
    T dx, T dt
) {
    TIMER_FLAG("Compute_Gradient");
    gradient.Fill(VECTOR<T, dim>(0));
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        if (dof(node) == -1) return; 
        auto& rI = gradient.template Get_Component_Unchecked<0>(dof(node));
        auto& dvI = dv.template Get_Component_Unchecked<0>(dof(node));
        rI += g(dim) * (dvI - gravity * dt);
    });
    Add_Elastic_Force<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, grid, elasticityAttr ..., elasticityAttrMoved ..., particles, gradient, dx, -dt);
}

template <class T, int dim>
void Add_Elastic_Matrix(SPARSE_GRID<int, dim>&, SPARSE_GRID<VECTOR<T, dim + 1>, dim>&, MPM_PARTICLES<T, dim>&, BASE_STORAGE<int>&, BASE_STORAGE<MATRIX<T, dim>>&, bool, T, T) {}
template <class T, int dim, class CONSTITUTIVE_MODEL_FUNCTOR, class ... OTHER_FUNCTOR>
void Add_Elastic_Matrix(
    SPARSE_GRID<int, dim>& dof,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    MPM_PARTICLES<T, dim>& particles,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttr,
    typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrOther,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttrMoved,
    typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrOtherMoved,
    BASE_STORAGE<int>& entryCol, 
    BASE_STORAGE<MATRIX<T, dim>>& entryVal,
    bool project_spd,
    T dx, T weight
) {
    typename CONSTITUTIVE_MODEL_FUNCTOR::DIFFERENTIAL firstPiolaDiff;
    if constexpr (CONSTITUTIVE_MODEL_FUNCTOR::useJ)
        CONSTITUTIVE_MODEL_FUNCTOR::Compute_First_PiolaKirchoff_Stress_Derivative(elasticityAttrMoved, weight, firstPiolaDiff);
    else if constexpr (CONSTITUTIVE_MODEL_FUNCTOR::projectable)
        CONSTITUTIVE_MODEL_FUNCTOR::Compute_First_PiolaKirchoff_Stress_Derivative(elasticityAttrMoved, weight, project_spd, firstPiolaDiff);
    else 
        CONSTITUTIVE_MODEL_FUNCTOR::Compute_First_PiolaKirchoff_Stress_Derivative(elasticityAttrMoved, weight, firstPiolaDiff);
    int colSize = std::pow(5, dim);
    Colored_Par_Each(particles, grid, dx, [&](const int i) {
        if (!elasticityAttr.contains(i)) return;
        auto &X = std::get<0>(particles.Get_Unchecked(i));
        auto &FJ = std::get<0>(elasticityAttr.Get_Unchecked(i));
        
        std::vector<VECTOR<T, dim>> cached_dw; cached_dw.reserve(colSize);
        std::vector<VECTOR<int, dim>> cached_node; cached_node.reserve(colSize);
        std::vector<int> cached_idx; cached_idx.reserve(colSize);

        BSPLINE_WEIGHTS<T, dim, 2> spline(X, dx);
        for (auto [node, w, dw, g] : grid.Iterate_Kernel(spline)) {
            if (dof(node) == -1) continue;
            if constexpr (CONSTITUTIVE_MODEL_FUNCTOR::useJ) cached_dw.push_back(FJ * dw);
            else cached_dw.push_back(FJ.transpose() * dw);
            cached_node.push_back(node);
            cached_idx.push_back(dof(node));
        }
        
        auto& firstPiolaDiffI = std::get<0>(firstPiolaDiff.Get_Unchecked(i));
        for (int _i = 0; _i < (int)cached_dw.size(); ++_i) {
            const VECTOR<T, dim>& dwi =  cached_dw[_i];
            const VECTOR<int, dim>& nodei = cached_node[_i];
            const int& dofi = cached_idx[_i];
            for (int j = 0; j < (int)cached_dw.size(); ++j) {
                const VECTOR<T, dim>& dwj = cached_dw[j];
                const VECTOR<int, dim>& nodej = cached_node[j];
                const int& dofj = cached_idx[j];
                if (dofj < dofi) continue;
                MATRIX<T, dim> dFdX;
                if constexpr (CONSTITUTIVE_MODEL_FUNCTOR::useJ) { dFdX += firstPiolaDiffI * outer_product(dwi, dwj); }
                else {
                    for (int q = 0; q < dim; q++) { for (int v = 0; v < dim; v++) {
                        dFdX += firstPiolaDiffI.template block<dim, dim>(dim * v, dim * q) * dwi(v) * dwj(q);}}
                }

                std::get<0>(entryCol.Get_Unchecked(dofi * colSize + Linear_Offset(nodei - nodej))) = dofj;
                std::get<0>(entryVal.Get_Unchecked(dofi * colSize + Linear_Offset(nodei - nodej))) += dFdX;

                if (dofi != dofj) {
                    std::get<0>(entryCol.Get_Unchecked(dofj * colSize + Linear_Offset(nodej - nodei))) = dofi;
                    std::get<0>(entryVal.Get_Unchecked(dofj * colSize + Linear_Offset(nodej - nodei))) += dFdX.transpose();
                }
            }
        }
    });
    Add_Elastic_Matrix<T, dim, OTHER_FUNCTOR...>(dof, grid, particles, elasticityAttrOther ..., elasticityAttrOtherMoved ..., entryCol, entryVal, project_spd, dx, weight);
}

template <class T, int dim, class ... CONSTITUTIVE_MODEL_FUNCTOR>
void Compute_Hessian(
    SPARSE_GRID<int, dim>& dof,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    MPM_PARTICLES<T, dim>& particles,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttr,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttrMoved,
    BASE_STORAGE<int>& entryCol, 
    BASE_STORAGE<MATRIX<T, dim>>& entryVal,
    bool project_spd,
    T dx, T dt
) {
    TIMER_FLAG("Compute_Hessian");
    int colSize = std::pow(5, dim);
    entryCol.Fill(-1);
    entryVal.Fill(MATRIX<T, dim>(0));
    // inertia
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        if(dof(node) == -1) return;
        int middle_index = Linear_Offset(VECTOR<int, dim>(0));
        std::get<0>(entryCol.Get_Unchecked(dof(node) * colSize + middle_index)) = dof(node);
        std::get<0>(entryVal.Get_Unchecked(dof(node) * colSize + middle_index)) += MATRIX<T, dim>(g(dim));
    });
    Add_Elastic_Matrix<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, grid, particles, elasticityAttr ..., elasticityAttrMoved ..., entryCol, entryVal, project_spd, dx, dt * dt);
}

template <class T, int dim>
void Compress_CSR(
    BASE_STORAGE<int>& entryCol, 
    BASE_STORAGE<MATRIX<T, dim>>& entryVal,
    int rowSize,
    CSR_MATRIX<T> &spMat)
{
    TIMER_FLAG("construct Matrix");
    std::vector<int> ptr;
    std::vector<int> col;
    std::vector<T> val;
    int colSize = std::pow(5, dim);
    ptr.resize(rowSize * dim + 1);
    ptr[0] = 0;
    for(int row_idx = 0; row_idx < rowSize; ++row_idx)
    {
        int idx = row_idx * colSize;
        int cnt = 0;
        for (int j = 0; j < colSize; ++idx, ++j) {
            int col_idx = std::get<0>(entryCol.Get_Unchecked(idx));
            if (col_idx != -1)
                cnt++;
        }
        for (int d = 0; d < dim; ++d)
            ptr[dim * row_idx + d + 1] = cnt * dim;
    }

    for (int i = 0; i < rowSize * dim; ++i) { ptr[i + 1] += ptr[i]; }
    col.resize(ptr.back()); // for parallelism
    val.resize(ptr.back()); // for parallelism
    for(int row_idx = 0; row_idx < rowSize; ++row_idx)
    {
        std::vector<std::tuple<MATRIX<T, dim>, int>> zipped;
        zipped.resize(ptr[row_idx * dim + 1] / dim - ptr[row_idx * dim] / dim);
        int idx = row_idx * colSize;
        int cnt = 0;
        for (int j = 0; j < colSize; ++idx, ++j) {
            auto& col_idx = std::get<0>(entryCol.Get_Unchecked(idx));
            auto& Val = std::get<0>(entryVal.Get_Unchecked(idx));
            if (col_idx != -1) {
                std::get<0>(zipped[cnt]) = Val;
                std::get<1>(zipped[cnt++]) = col_idx;
            }
        }
        // sort each row's entry by col index;
        std::sort(std::begin(zipped), std::end(zipped),
            [&](const auto& a, const auto& b) {
                return std::get<1>(a) < std::get<1>(b);
            });

        for (int c = 0; c < (int)zipped.size(); ++c) {
            int col_idx = std::get<1>(zipped[c]);
            MATRIX<T, dim> value = std::get<0>(zipped[c]);
            for (int d1 = 0; d1 < dim; ++d1)
                for (int d2 = 0; d2 < dim; ++d2) {
                    col[ptr[row_idx * dim + d1] + c * dim + d2] = col_idx * dim + d2;
                    val[ptr[row_idx * dim + d1] + c * dim + d2] = value(d1, d2);
                }
        }
    }
    spMat.Construct_From_CSR(ptr, col, val);
}

template <class T, int dim>
void Solve_DDV(BASE_STORAGE<int>& entryCol, BASE_STORAGE<MATRIX<T, dim>> &entryVal, BASE_STORAGE<VECTOR<T, dim>>& gradient, BASE_STORAGE<VECTOR<T, dim>>& ddv, T tol= 1e-5, bool verbose=false)
{
    TIMER_FLAG("Solve_DDV");
    CSR_MATRIX<T> spMat;
    Compress_CSR(entryCol, entryVal, gradient.size, spMat);
    std::vector<T> rhs(gradient.size * dim), sol(gradient.size * dim, 0);
    gradient.Par_Each([&](int id, auto data) {
        auto &[g] = data;
        for(int d = 0; d < dim; ++d)
            rhs[id * dim + d] = -g(d);
    });
    auto prm = Default_Params();
    Solve(spMat, rhs, sol, tol, 1000, prm, verbose);
    ddv.Par_Each([&](auto id, auto data) {
        auto &[ddvI] = data;
        for(int d = 0; d < dim; ++d) { ddvI(d) = sol[id * dim + d]; }
    });
}

template <class T, int dim>
T Norm(BASE_STORAGE<VECTOR<T, dim>>& dv)
{
    T norm = 0;
    dv.Each([&](auto, auto data)
    {
        norm += std::get<0>(data).length2();
    });
    return std::sqrt(norm);
}

template <class T, int dim>
void Update_State(SPARSE_GRID<int, dim>&, BASE_STORAGE<VECTOR<T, dim>>&, MPM_PARTICLES<T, dim>, SPARSE_GRID<VECTOR<T, dim + 1>, dim>, T, T) { }
template <class T, int dim, class CONSTITUTIVE_MODEL_FUNCTOR, class ... OTHER_FUNCTOR>
void Update_State(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>>& dv, 
    MPM_PARTICLES<T, dim> particles,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim> grid,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttr,
    typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrOther,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttrMoved,
    typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrOtherMoved,
    T dx, T dt
) {
    TIMER_FLAG("Update_State");
    if constexpr (CONSTITUTIVE_MODEL_FUNCTOR::useJ) {
        using JOINED = decltype(elasticityAttr.Join(particles));
        elasticityAttr.Join(particles).Par_Each([&](const int i, auto data) {    
            auto& J = std::get<FIELDS<JOINED>::J>(data);
            MATRIX<T, dim> gradV(0);
            auto& X = std::get<FIELDS<JOINED>::X>(data);
            BSPLINE_WEIGHTS<T, dim, 2> spline(X, dx);
            for (auto [node, w, dw, g] : grid.Iterate_Kernel(spline)) {
                if (dof(node) == -1) return;
                gradV += outer_product(VECTOR<T, dim>(g) + dv.template Get_Component_Unchecked<0>(dof(node)), dw);
            }
            elasticityAttrMoved.template Get_Component_Unchecked<0>(i) = (1 + dt * gradV.trace()) * J;
        });
    }
    else {
        using JOINED = decltype(elasticityAttr.Join(particles));
        elasticityAttr.Join(particles).Par_Each([&](const int i, auto data) {
            auto& F = std::get<FIELDS<JOINED>::F>(data);
            MATRIX<T, dim> gradV(0);
            auto& X = std::get<FIELDS<JOINED>::X>(data);
            BSPLINE_WEIGHTS<T, dim, 2> spline(X, dx);
            for (auto [node, w, dw, g] : grid.Iterate_Kernel(spline)) {
                if (dof(node) == -1) return;
                gradV += outer_product(VECTOR<T, dim>(g) + dv.template Get_Component_Unchecked<0>(dof(node)), dw);
            }
            elasticityAttrMoved.template Get_Component_Unchecked<0>(i) = (MATRIX<T, dim>(1) + dt * gradV) * F;
        });
    }
    Update_State<T, dim, OTHER_FUNCTOR...>(dof, dv, particles, grid, elasticityAttrOther..., elasticityAttrOtherMoved..., dx, dt);
}

template <typename T, int dim>
void Build_Collison_Nodes(
    SPARSE_GRID<int, dim>& dof,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    std::map<int, std::pair<COLLISION_OBJECT_TYPE, MATRIX<T, dim>>>& collisionNodes
) {
    //TODO
}

template <typename T, int dim>
T Initial_Step_Size(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>>& dv,
    BASE_STORAGE<VECTOR<T, dim>>& ddv,
    MPM_PARTICLES<T, dim>& particles,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid
) {
    return 1.;
}

template <typename T, int dim, class CONSTITUTIVE_MODEL_FUNCTOR, class ... OTHER_FUNCTOR>
T Initial_Step_Size(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>>& dv,
    BASE_STORAGE<VECTOR<T, dim>>& ddv,
    MPM_PARTICLES<T, dim>& particles,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttr,
    typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrOther,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& elasticityAttrMoved,
    typename OTHER_FUNCTOR::STORAGE& ... elasticityAttrOtherMoved
) {
    //TODO
    return 1;
    // T alpha = 1;
    // return std::min(Initial_Step_Size<T, dim, OTHER_FUNCTOR...>(dof, dv, ddv, particles, grid, elasticityAttrOther..., elasticityAttrOtherMoved...), alpha);
}

template <typename CONSTITUTIVE_MODEL_FUNCTOR>
std::unique_ptr<typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE> Copy_Elasticity_Attr(typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE &storage)
{
    auto new_storage_p = std::make_unique<typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE>(storage.size);
    storage.deep_copy_to(*new_storage_p);
    return new_storage_p;
}

template <class T, int dim, class ... CONSTITUTIVE_MODEL_FUNCTOR>
void Check_Gradient(SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>> &dv,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    MPM_PARTICLES<T, dim>& particles,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttr,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttrMoved,
    T dx, T dt, T epsilon);
template <class T, int dim, class ... CONSTITUTIVE_MODEL_FUNCTOR>
void Check_Hessian(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>> &dv,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    MPM_PARTICLES<T, dim>& particles,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttr,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttrMoved,
    const VECTOR<T, dim>& gravity,
    BASE_STORAGE<int>& entryCol, BASE_STORAGE<MATRIX<T, dim>> &entryVal,
    T dx, T dt, T epsilon);

template <typename T, int dim, typename ... CONSTITUTIVE_MODEL_FUNCTOR>
void Implicit_Update(
    MPM_PARTICLES<T, dim>& particles,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttr, 
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttrMoved, 
    const COLLIDER<T, dim>& collider,
    const VECTOR<T, dim>& gravity,
    T dx,
    T dt,
    T newtonTol=1e-2
) {
    TIMER_FLAG("Implicit_Update");
    SPARSE_GRID<int, dim> dof;
    Colored_Par_Each(particles, dof, dx, [&](const int i) {});
    int num_nodes = Label_DOF(grid, dof);
    std::map<int, std::pair<COLLISION_OBJECT_TYPE, MATRIX<T, dim>>> collisionNodes;
    // Build_Collison_Nodes(dof, grid, collisionNodes);
    int colSize = std::pow(5, dim);
    BASE_STORAGE<VECTOR<T, dim>> dv(num_nodes);
    BASE_STORAGE<VECTOR<T, dim>> ddv(num_nodes);
    BASE_STORAGE<VECTOR<T, dim>> gradient(num_nodes);
    BASE_STORAGE<int> entryCol(num_nodes * colSize);
    BASE_STORAGE<MATRIX<T, dim>> entryVal(num_nodes * colSize);
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        if(dof(node) == -1) return;
        VECTOR<T, dim> new_v = VECTOR<T, dim>(g) + gravity * dt;
        collider.Resolve(node.template cast<T>() * dx, new_v);
        dv.Insert(dof(node), new_v - VECTOR<T, dim>(g));
        ddv.Insert(dof(node), VECTOR<T, dim>(0));
        gradient.Insert(dof(node), VECTOR<T, dim>(0));
        for(int j = 0; j < colSize; ++j) {
            entryCol.Insert(dof(node) * colSize + j, -1);
            entryVal.Insert(dof(node) * colSize + j, MATRIX<T, dim>());
        }
    });
    std::cout << "Initialization done." << std::endl;
    Update_State<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., dx, dt);
    T E = Compute_Potential<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, grid, elasticityAttrMoved..., gravity, dt);
    while (true) {
        Compute_Gradient<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., gradient, gravity, dx, dt);
        T gradient_norm = Norm(gradient);
        std::cout << std::setprecision(6) << "Gradient norm: " << gradient_norm << std::endl;
        if (gradient_norm < newtonTol) break;
        Compute_Hessian<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, grid, particles, elasticityAttr ..., elasticityAttrMoved ..., entryCol, entryVal, true, dx, dt);
        Solve_DDV<T, dim>(entryCol, entryVal, gradient, ddv, 1e-7, true);
        T alpha = Initial_Step_Size<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, ddv, particles, grid, elasticityAttr ..., elasticityAttrMoved ...);
        while (true) {
            BASE_STORAGE<VECTOR<T, dim>> new_dv;
            dv.deep_copy_to(new_dv);
            new_dv.Join(ddv).Par_Each([&](auto i, auto data){
                auto &[new_dvI, ddvI] = data;
                new_dvI += alpha * ddvI;
            });
            Update_State<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, new_dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., dx, dt);
            T new_E = Compute_Potential<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, new_dv, grid, elasticityAttrMoved..., gravity, dt);
            // std::cout << "E: " << E << ", new_E: " << new_E << std::endl;
            // getchar();
            if (new_E < E) {
                new_dv.deep_copy_to(dv);
                E = new_E;
                break;
            }
            alpha *= 0.5;
        }
    }
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        int idx = dof(node);
        if(idx < 0) return;
        VECTOR<T, dim> new_v = VECTOR<T, dim>(g) + dv.template Get_Component_Unchecked<0>(idx);
        collider.Resolve(node.template cast<T>() * dx, new_v);
        g = VECTOR<T, dim+1>(new_v, g(dim));
    });
}

// ################# DEBUG ##################
template <class T, int dim, typename ... CONSTITUTIVE_MODEL_FUNCTOR>
void Write_System(std::string prefix, typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttr, typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttrMoved)
{
    // CSR_MATRIX<T> spMat;
    // Compress_CSR(entryCol, entryVal, force.size, spMat);
    // std::vector<T> rhs(force.size * dim), sol(force.size * dim, 0);
    // force.Par_Each([&](int id, auto data) {
    //     auto &[g] = data;
    //     for(int d = 0; d < dim; ++d)
    //         rhs[id * dim + d] = -g(d);
    // });
    // auto prm = Default_Params();
    // Solve(spMat, rhs, sol, T(1.0e-6), 1000, prm);
    // amgcl::io::mm_write(prefix + "_matrix.mtx", spMat.Get_Matrix());
    // amgcl::io::mm_write(prefix + "_rhs.mtx", rhs.data(), rhs.size());
    // amgcl::io::mm_write(prefix + "_sol.mtx", sol.data(), sol.size());
}

template <class T, int dim, class ... CONSTITUTIVE_MODEL_FUNCTOR>
void Check_Gradient(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>> &dv,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    MPM_PARTICLES<T, dim>& particles,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttr,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttrMoved,
    const VECTOR<T, dim>& gravity,
    T dx, T dt, T epsilon
) {
    BASE_STORAGE<VECTOR<T, dim>> gradient;
    BASE_STORAGE<VECTOR<T, dim>> gradient_FD;
    dv.deep_copy_to(gradient);
    gradient.Fill(VECTOR<T, dim>());
    gradient.deep_copy_to(gradient_FD);
    gradient_FD.Fill(VECTOR<T, dim>());
    Update_State<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., dx, dt);
    Compute_Gradient<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., gradient, gravity, dx, dt);
    T E0 = Compute_Potential<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, grid, elasticityAttrMoved..., gravity, dt);
    dv.Join(gradient_FD).Each([&](auto i, auto data){
        auto& [dvI, fI] = data;
        for (int d = 0; d < dim; ++d){
            dvI(d) += epsilon;
            Update_State<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., dx, dt);
            T E1 = Compute_Potential<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, grid, elasticityAttrMoved..., gravity, dt);
            fI(d) = (E1 - E0) / epsilon;
            dvI(d) -= epsilon;
            std::cout << fI(d) * 10000. << " " << gradient.template Get_Component_Unchecked<0>(i)(d) * 10000. << std::endl;
            getchar();
        }
    });
    // std::cout << std::setprecision(6) <<"err_abs = " << err << ", err_rel = " << err / norm << ", ||f|| = " << norm <<std::endl;
}

template <class T, int dim, class ... CONSTITUTIVE_MODEL_FUNCTOR>
void Check_Hessian(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>> &dv,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    MPM_PARTICLES<T, dim>& particles,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttr,
    typename CONSTITUTIVE_MODEL_FUNCTOR::STORAGE& ... elasticityAttrMoved,
    const VECTOR<T, dim>& gravity,
    BASE_STORAGE<int>& entryCol, BASE_STORAGE<MATRIX<T, dim>> &entryVal,
    T dx, T dt, T epsilon
) {
    int colSize = std::pow(5, dim);
    BASE_STORAGE<VECTOR<T, dim>> gradient0;
    BASE_STORAGE<VECTOR<T, dim>> gradient1;
    dv.deep_copy_to(gradient0);
    dv.deep_copy_to(gradient1);
    gradient0.Fill(VECTOR<T, dim>());
    gradient1.Fill(VECTOR<T, dim>());
    Update_State<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., dx, dt);
    Compute_Gradient<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., gradient0, gravity, dx, dt);
    Compute_Hessian<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof,  grid, particles, elasticityAttr ..., elasticityAttrMoved ..., entryCol, entryVal, false, dx, dt);
    dv.Each([&](auto i, auto data){
        auto& [dvI] = data;
        for (int d = 0; d < dim; ++d){
            dvI(d) += epsilon;
            Update_State<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., dx, dt);
            Compute_Gradient<T, dim, CONSTITUTIVE_MODEL_FUNCTOR ...>(dof, dv, particles, grid, elasticityAttr ..., elasticityAttrMoved ..., gradient1, gravity, dx, dt);
            int st = i * colSize;
            int ed = st + colSize;
            for (; st < ed; ++st) {
                int j = entryCol.template Get_Component_Unchecked<0>(st);
                if (j == -1) continue;
                MATRIX<T, dim> valj = entryVal.template Get_Component_Unchecked<0>(st);
                VECTOR<T, dim> delta = (gradient1.template Get_Component_Unchecked<0>(j) - gradient0.template Get_Component_Unchecked<0>(j)) / epsilon;
                for (int d2 = 0; d2 < dim; ++d2) {
                    std::cout << std::setprecision(6) << "Hessian_FD: " << delta(d2) << "Real: " << valj(d, d2) << std::endl;
                    getchar();
                }
            }
            dvI(d) -= epsilon;
        }
    });
    // std::cout << std::setprecision(6) <<"err_abs = " << err << ", err_rel = " << err / norm << ", ||f|| = " << norm <<std::endl;
}

template <class T, int dim>
void Check_Gradient(
    SPARSE_GRID<int, dim>& dof,
    BASE_STORAGE<VECTOR<T, dim>> &dv,
    SPARSE_GRID<VECTOR<T, dim + 1>, dim>& grid,
    const VECTOR<T, dim>& gravity,
    T dt, T epsilon = 1e-6
) {
    BASE_STORAGE<VECTOR<T, dim>> f;
    BASE_STORAGE<VECTOR<T, dim>> f_FD;
    dv.deep_copy_to(f);
    f.Fill(VECTOR<T, dim>());
    f.deep_copy_to(f_FD);
    f_FD.Fill(VECTOR<T, dim>());
    T E0 = Inertia_Energy(dof, dv, grid, gravity, dt);
    Add_Inertia_Force(dof, dv, f, grid, gravity, dt);
    dv.Join(f_FD).Each([&](auto i, auto data){
        auto& [dvI, fI] = data;
        for (int d = 0; d < dim; ++d){
            dvI(d) += epsilon;
            T E1 = Inertia_Energy(dof, dv, grid, gravity, dt);
            fI(d) = (E1 - E0) / epsilon;
            dvI(d) -= epsilon;
        }
    });
    auto diff_p = f - f_FD;
    T err = Norm(*diff_p);
    T norm = Norm(f);
    std::cout << std::setprecision(6) <<"err_abs = " << err << ", err_rel = " << err / norm << ", ||f|| = " << norm <<std::endl;
}

// ##################### Export #########################
template <class T, int dim>
void Export_Implicit_Euler_Impl(py::module& m)
{
    m.def("Implicit_Update", &Implicit_Update<T, dim, 
                                                FIXED_COROTATED_FUNCTOR<T, dim>, 
                                                EQUATION_OF_STATE_FUNCTOR<T, dim>,
                                                LINEAR_COROTATED_FUNCTOR<T, dim>,
                                                NEOHOOKEAN_BORDEN_FUNCTOR<T, dim>>);
    m.def("Copy_Elasticity_Attr", &Copy_Elasticity_Attr<FIXED_COROTATED_FUNCTOR<T, dim>>);
    m.def("Copy_Elasticity_Attr", &Copy_Elasticity_Attr<EQUATION_OF_STATE_FUNCTOR<T, dim>>);
    m.def("Copy_Elasticity_Attr", &Copy_Elasticity_Attr<LINEAR_COROTATED_FUNCTOR<T, dim>>);
    m.def("Copy_Elasticity_Attr", &Copy_Elasticity_Attr<NEOHOOKEAN_BORDEN_FUNCTOR<T, dim>>);
}

void Export_Implicit_Euler(py::module& m)
{
    Export_Implicit_Euler_Impl<double, 2>(m);
    Export_Implicit_Euler_Impl<float, 2>(m);
    Export_Implicit_Euler_Impl<double, 3>(m);
    Export_Implicit_Euler_Impl<float, 3>(m);
}
}
