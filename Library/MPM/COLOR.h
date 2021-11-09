#pragma once

#include <Utils/PROFILER.h>
#include <Storage/storage.hpp>

namespace JGSL {

template <class T, int dim, class ATTR>
void Colored_Par_Each(BASE_STORAGE<VECTOR<T, dim>, VECTOR<T, dim>, MATRIX<T, dim>, MATRIX<T, dim>, T>& particles, SPARSE_GRID<ATTR, dim>& grid, T dx, std::function<void(int)> f) {
    std::vector<int> indices;
    BASE_STORAGE<VECTOR<int, 2>> groups[1 << dim];
    std::vector<std::pair<uint64_t, int>> zorder_idx(particles.size);

    const T one_over_dx = 1 / dx;

    { TIMER_FLAG("P2G 1");
    particles.Par_Each([&](const int i, auto data) {
        auto&[X, V, gradV, C, _] = data;
        zorder_idx[i] = std::make_pair((grid.Get_Zorder(Base_Node<2, T, dim>(X * one_over_dx)) >> 6), i);
    });
    }

    { TIMER_FLAG("P2G 2");
    std::sort(zorder_idx.begin(), zorder_idx.end());
    }
    { TIMER_FLAG("P2G 3");
    int last_index = 0;
    for (int i = 0; i < particles.size; ++i) {
        indices.push_back(zorder_idx[i].second);
        if (i == particles.size - 1 || (zorder_idx[i].first) != (zorder_idx[i + 1].first)) {
            groups[zorder_idx[i].first & ((1 << dim) - 1)].Append(VECTOR<int ,2>(last_index, i));
            last_index = i + 1;
            VECTOR<int, dim> base_node = grid.Get_Node(zorder_idx[i].first << 6);
            if constexpr (dim == 2) {
                for (int delta_x = 0; delta_x <= 8; delta_x += 8)
                    for (int delta_y = 0; delta_y <= 8; delta_y += 8)
                        grid(base_node + VECTOR<int, dim>(delta_x, delta_y));
            } else {
                for (int delta_x = 0; delta_x <= 4; delta_x += 4)
                    for (int delta_y = 0; delta_y <= 4; delta_y += 4)
                        for (int delta_z = 0; delta_z <= 4; delta_z += 4)
                            grid(base_node + VECTOR<int, dim>(delta_x, delta_y, delta_z));
            }
        }
    }
    }
    { TIMER_FLAG("P2G 4");
    for (int color = 0; color < (1 << dim); ++color) {
        groups[color].Par_Each([&](const int i, auto data) {
            auto& [range] = data;
            for (int idx = range(0); idx <= range(1); ++idx)
                f(indices[idx]);
        });
    }
    }
}

}