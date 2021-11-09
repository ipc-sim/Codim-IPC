#pragma once
//#####################################################################
// Function Dump_Particles
//#####################################################################
#include <Particles/PARTICLES.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Partio.h>

namespace py = pybind11;
namespace JGSL {

//#####################################################################
// Function Dump_Particles
//#####################################################################
template <class T, int dim>
void Dump_Particles(MPM_PARTICLES<T, dim>& particles, const std::string& filename) {
    TIMER_FLAG("Dump Particles");
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    parts->addParticles(particles.size);
    particles.Each([&](const int k, auto data) {
        auto& [X, V, _, __, m] = data;
        auto* posP = parts->dataWrite<float>(posH, k);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = X(d);
    });
    Partio::write(filename.c_str(), *parts);
    parts->release();
}

/* move to somewhere else
//#####################################################################
// Function Dump_FLIP_Particles
//#####################################################################
template <class T, int dim>
void Dump_FLIP_Particles(FLIP_PARTICLES<T, dim>& particles, const std::string& filename) {
    TIMER_FLAG("Dump_FLIP_Particles");
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    parts->addParticles(particles.size);
    particles.Each([&](const int k, auto data) {
        auto& [X, V] = data;
        auto* posP = parts->dataWrite<float>(posH, k);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = X(d);
    });
    Partio::write(filename.c_str(), *parts);
    parts->release();
}

//#####################################################################
// Function Dump_APIC_Particles
//#####################################################################
template <class T, int dim>
void Dump_APIC_Particles(APIC_PARTICLES<T, dim>& particles, const std::string& filename) {
    TIMER_FLAG("Dump_FLIP_Particles");
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    parts->addParticles(particles.size);
    particles.Each([&](const int k, auto data) {
        auto& [X, V, _] = data;
        auto* posP = parts->dataWrite<float>(posH, k);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = X(d);
    });
    Partio::write(filename.c_str(), *parts);
    parts->release();
}
*/

//#####################################################################
// Function Dump_Particles_J
//#####################################################################
template <class T, int dim>
void Dump_Particles_J(MPM_PARTICLES<T, dim>& particles, EQUATION_OF_STATE<T>& eos, const std::string& filename) {
    TIMER_FLAG("Dump Particles J");
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH, JH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    JH = parts->addAttribute("J", Partio::VECTOR, 1);
    parts->addParticles(particles.size);
    particles.Join(eos).Each([&](const int k, auto data) {
        auto& [X, V, _, __, m, J, vol, bulk, gamma] = data;
        auto* posP = parts->dataWrite<float>(posH, k);
        auto* JP = parts->dataWrite<float>(JH, k);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = X(d);
        JP[0] = J;
    });
    Partio::write(filename.c_str(), *parts);
    parts->release();
}

//#####################################################################

template <class T, int dim>
void Dump_Particles_WithLogJp(const PARTICLES<VECTOR<T, dim>>& particles, const PARTICLES<T>& logJp, const std::string& filename)
{
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH, logJpH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    logJpH = parts->addAttribute("logJp", Partio::VECTOR, 1);

    parts->addParticles(particles.data.size());

    for (int k = 0; k < particles.data.size(); k++) {
        auto* posP = parts->dataWrite<float>(posH, k);
        auto* logJpP = parts->dataWrite<float>(logJpH, k);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = particles.data[k](d);
        logJpP[0] = logJp.data[k];
    }
    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template<class Scalar, class T, int dim>
void Dump_Grid(SPARSE_GRID<Scalar, dim>& grid, T dx, const std::string& filename)
{
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("info", Partio::VECTOR, 1);
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        int idx = parts->addParticle();
        auto* posP = parts->dataWrite<float>(posH, idx);
        auto* infoP = parts->dataWrite<float>(infoH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = (float)node(d) * dx;
        infoP[0] = g;
    });
    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template <class _T, int _dim, class T, int dim>
void Dump_Grid(SPARSE_GRID<VECTOR<_T, _dim>, dim>& grid, T dx, const std::string& filename)
{
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("info", Partio::VECTOR, 3);
    grid.Iterate_Grid([&](const auto& node, auto& g) {
        int idx = parts->addParticle();
        auto* posP = parts->dataWrite<float>(posH, idx);
        auto* infoP = parts->dataWrite<float>(infoH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = (float)node(d) * dx;
        for (int d = 0; d < 3; ++d) infoP[d] = 0;
        for (int d = 0; d < _dim; ++d) infoP[d] = g(d);
    });
    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template <class T, int dim>
void Export_Visualizer_Impl(py::module& m) {
    m.def("Dump_Particles", &Dump_Particles<T, dim>);
    m.def("Dump_Particles_J", &Dump_Particles_J<T, dim>);
    m.def("Dump_Particles_WithLogJp", &Dump_Particles_WithLogJp<T, dim>);
    m.def("Dump_Grid", &Dump_Grid<T, T, dim>);
    m.def("Dump_Grid", &Dump_Grid<T, dim + 1, T, dim>);
    m.def("Dump_Grid", &Dump_Grid<T, dim, T, dim>);
    /* move to somewhere else
    m.def("Dump_FLIP_Particles", &Dump_FLIP_Particles<T, dim>);
    m.def("Dump_APIC_Particles", &Dump_APIC_Particles<T, dim>);
     */
}

void Export_Visualizer(py::module& m) {
    Export_Visualizer_Impl<double, 2>(m);
    Export_Visualizer_Impl<double, 3>(m);
}

};