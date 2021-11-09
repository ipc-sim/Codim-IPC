#pragma once

#include <Utils/LOGGING.h>
#include <map>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

namespace py = pybind11;
namespace JGSL {
namespace TIMER {

/*
ATTRIBUTE_RESET = "\033[0m";
BLACK = "\033[22;30m";
RED = "\033[22;31m";
GREEN = "\033[22;32m";
BROWN = "\033[22;33m";
BLUE = "\033[22;34m";
MAGENTA = "\033[22;35m";
CYAN = "\033[22;36m";
GREY = "\033[22;37m";
DARKGREY = "\033[01;30m";
LIGHTRED = "\033[01;31m";
LIGHTGREEN = "\033[01;32m";
YELLOW = "\033[01;33m";
LIGHTBLUE = "\033[01;34m";
LIGHTMAGENTA = "\033[01;35m";
LIGHTCYAN = "\033[01;36m";
WHITE = "\033[01;37m";
*/

std::map<std::pair<std::string, int>, int> name_scope;

std::vector<std::pair<std::string, int>> scope_name(1, std::make_pair("Global", -1));
std::vector<std::chrono::duration<double>> scope_duration(1, std::chrono::duration<double>(0));
std::vector<std::chrono::duration<double>> global_duration(1, std::chrono::duration<double>(0));
std::vector<std::vector<int>> scope_edges(1, std::vector<int>());
std::vector<int> scope_stack(1, 0);

std::string duration2string(const std::chrono::duration<double>& elapsed_seconds)
{
    using namespace std::chrono;
    std::string str;
    auto dur = elapsed_seconds;

    int num_days = int(dur / hours(24));
    if (num_days) str += std::to_string(num_days) + "d ";
    dur -= num_days * hours(24);

    int num_hours = int(dur / hours(1));
    if (num_hours) str += std::to_string(num_hours) + "h ";
    dur -= num_hours * hours(1);

    int num_minutes = int(dur / minutes(1));
    if (num_minutes) str += std::to_string(num_minutes) + "m ";
    dur -= num_minutes * minutes(1);

    str += std::to_string(dur.count()) + "s";
    return str;
}

class ScopedTimer {
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> end_time;
    int id;
    std::string name;
    bool analyze;

public:
    explicit ScopedTimer(const std::string& name, bool analyze) : name(name), analyze(analyze)
    {
        start_time = std::chrono::steady_clock::now();
        auto name_parent = std::make_pair(name, scope_stack.back());
        if (name_scope.find(name_parent) != name_scope.end()) {
            id = name_scope[name_parent];
        }
        else {
            id = scope_name.size();
            name_scope[name_parent] = id;
            scope_name.push_back(name_parent);
            scope_duration.emplace_back(0);
            global_duration.emplace_back(0);
            scope_edges.emplace_back();
            scope_edges[scope_stack.back()].push_back(id);
        }
        scope_stack.push_back(id);
    }

    ~ScopedTimer()
    {
        end_time = std::chrono::steady_clock::now();
        scope_duration[id] += end_time - start_time;
        scope_stack.pop_back();
        if (analyze) {
            JGSL_ANALYZE(name + std::string(" Timing"), (end_time - start_time).count() / 1e9);
        }
        // for (int i = 0; i < scope_stack.size(); ++i)
        //     printf("\t");
        // printf("%s\n", scope_name[id].first.c_str());
    }
};

class GlobalTimer {

public:
    std::chrono::time_point<std::chrono::steady_clock> current_time;

    GlobalTimer()
    {
        current_time = std::chrono::steady_clock::now();
    }
};
GlobalTimer global_timer;

void traverseScopes(int id, int depth)
{
    auto printScope = [&]() {
        std::string scope_str = duration2string(scope_duration[id]);
        std::string global_str = duration2string(global_duration[id]);
        double scope_percent = scope_duration[id].count() * 100 / scope_duration[0].count();
        double global_percent = global_duration[id].count() * 100 / global_duration[0].count();
        std::cout << std::fixed << std::setprecision(1);
        std::cout << " " << scope_name[id].first << " : " << scope_str << " (" << scope_percent << "%%)   \033[22;31m" << global_str<< " (" << global_percent << "%%)\033[0m" << std::endl;
    };
    if (scope_edges[id].empty()) {
        for (int i = 0; i < depth; ++i) std::cout << "┃";
        printScope();
    }
    else {
        for (int i = 0; i < depth; ++i) std::cout << "┃";
        std::cout << "┏";
        printScope();
        for (auto s : scope_edges[id])
            traverseScopes(s, depth + 1);

        // other scope
        std::chrono::duration<double> other_scope = scope_duration[id];
        std::chrono::duration<double> other_global = global_duration[id];
        for (auto s : scope_edges[id]) {
            other_scope -= scope_duration[s];
            other_global -= global_duration[s];
        }
        double other_scope_percent = other_scope.count() * 100 / scope_duration[0].count();
        double other_global_percent = other_global.count() * 100 / global_duration[0].count();
        if (other_global_percent >= 0.05) {
            for (int i = 0; i <= depth; ++i) std::cout << "┃";
            std::string scope_str = duration2string(other_scope);
            std::string global_str = duration2string(other_global);
            std::cout << " Uncounted : " << scope_str << " (" << other_scope_percent << "%%)   \033[22;31m" << global_str<< " (" << other_global_percent << "%%)\033[0m" << std::endl;
        }

        for (int i = 0; i < depth; ++i) std::cout << "┃";
        std::cout << "┗" << std::endl;
    }
}

void progress(double current_t, double duration)
{
    int length = 50;

    std::cout << "[";
    double simu_percent = (double)current_t / duration;
    int simu_int = (int)(simu_percent * length);
    for (int i = 0; i < simu_int; ++i) std::cout << "\033[22;32m#\033[0m";
    for (int i = simu_int; i < length; ++i) std::cout << " ";
    std::cout << "]";
    std::cout << std::fixed << std::setprecision(3) << std::setw(8);
    std::cout << current_t / duration * 100;
}

void flush(int current_frame, int frame_num, double elasped_dt, double frame_dt) {
    progress(elasped_dt, frame_dt);
    std::cout << " Frame " << current_frame << std::endl;
    progress(current_frame, frame_num);
    std::cout << std::endl;
    std::chrono::time_point<std::chrono::steady_clock> last_time = global_timer.current_time;
    std::chrono::time_point<std::chrono::steady_clock> current_time = std::chrono::steady_clock::now();
    global_timer.current_time = current_time;
    scope_duration[0] = current_time - last_time;
    for (int i = 0; i < scope_duration.size(); ++i)
        global_duration[i] += scope_duration[i];
    traverseScopes(0, 0);
    for (int i = 0; i < scope_duration.size(); ++i)
        scope_duration[i] = std::chrono::duration<double>(0);
    puts("");
}

} // namespace TIMER

#define TIMER_FLAG(name) TIMER::ScopedTimer scoped_timer(name, false)
#define TIMER_ANALYZE(name) TIMER::ScopedTimer scoped_timer(name, true)
//#define TIMER_FLUSH() TIMER::flush()
//#define TIMER_PROGRESS(current_t, duration) TIMER::progress(current_t, duration)

void Export_Profiler(py::module& m) {
    m.def("TIMER_FLUSH", &TIMER::flush, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
}

}