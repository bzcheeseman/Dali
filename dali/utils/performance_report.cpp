#include "performance_report.h"

#include <algorithm>
#include <iomanip>

#include "dali/utils/assert2.h"

using namespace std::placeholders;

void PerformanceReport::start_capture() {
    if (observer == nullptr) {
        observer = std::make_shared<debug::ScopeObserver>(
            std::bind(&PerformanceReport::on_enter, this, _1),
            std::bind(&PerformanceReport::on_exit, this, _1)
        );
    }
}

void PerformanceReport::stop_capture() {
    observer.reset();
}

void PerformanceReport::print(std::basic_ostream<char>& stream) {
    double all_functions_total_time_ms = 0.0;
    std::vector<std::string> keys;
    int max_fname_len = 0;

    for (auto& kv: function_name_to_stats) {
        keys.emplace_back(kv.first);
        max_fname_len = std::max(max_fname_len, (int)kv.first.size());
        all_functions_total_time_ms += kv.second.total_time_ms;
    }

    for (auto& kv: function_name_to_stats) {
        function_name_to_stats[kv.first].fraction_total =
                kv.second.total_time_ms / all_functions_total_time_ms;
    }

    std::sort(keys.begin(), keys.end(),
        [this](const std::string& key1, const std::string key2) {
            return function_name_to_stats[key1].fraction_total >
                   function_name_to_stats[key2].fraction_total;
        });

    stream << "============== PERFORMANCE REPORT ==================" << std::endl;

    stream << "% total | name | #calls | total time" << std::endl;
    for(auto& fname: keys) {
        auto& stats = function_name_to_stats[fname];
        stream    << std::setw(5) << std::setprecision(2)
                  << stats.fraction_total * 100.0  << "%\t"
                  << std::setw(max_fname_len + 2)
                  << fname                         << "\t"
                  << stats.num_calls               << "\t"
                  << std::setprecision(3)
                  << stats.total_time_ms / 1000.0  << " s"
                  << std::endl;
    }

    stream << "====================================================" << std::endl;

}

void PerformanceReport::on_enter(const debug::ScopeObserver::State& state) {
    std::lock_guard<decltype(state_mutex)> guard(state_mutex);
    // on the off chance it exists ignore that call. This is very
    // unliklely to happen.
    if (call_to_start_time.find(*state.trace.back()) == call_to_start_time.end()) {
        call_to_start_time[*state.trace.back()] = clock_t::now();
    }
}

void PerformanceReport::on_exit(const debug::ScopeObserver::State& state) {
    std::lock_guard<decltype(state_mutex)> guard(state_mutex);

    // This if may fail if we start capturing mid execution.
    // That's fine though - just ignore that call.
    auto start_time_iter = call_to_start_time.find(*state.trace.back());
    if (start_time_iter != call_to_start_time.end()) {
        auto duration =  std::chrono::duration_cast<std::chrono::milliseconds>(
                clock_t::now() - start_time_iter->second).count();
        call_to_start_time.erase(start_time_iter);
        auto& stats = function_name_to_stats[*state.trace.back()];
        stats.total_time_ms += duration;
        stats.num_calls  += 1;
    }
}
