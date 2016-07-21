#include "performance_report.h"

#include <algorithm>

#include "dali/utils/assert2.h"

void PerformanceReport::start_capture() {
    if (!capturing) {
        dali_function_start_handle = debug::dali_function_start.register_callback(
            [this](const std::string& fname, int callid) {
                std::lock_guard<decltype(state_mutex)> guard(state_mutex);
                // on the off chance it exists ignore that call. This is very
                // unliklely to happen.
                if (call_to_start_time.find(callid) == call_to_start_time.end()) {
                    call_to_start_time[callid] = clock_t::now();
                }
            }
        );
        dali_function_end_handle = debug::dali_function_end.register_callback(
            [this](const std::string& fname, int callid) {

                std::lock_guard<decltype(state_mutex)> guard(state_mutex);

                // This if may fail if we start capturing mid execution.
                // That's fine though - just ignore that call.
                auto start_time_iter = call_to_start_time.find(callid);
                if (start_time_iter != call_to_start_time.end()) {
                    auto duration =  std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock_t::now() - start_time_iter->second).count();
                    call_to_start_time.erase(start_time_iter);
                    auto& stats = function_name_to_stats[fname];
                    stats.total_time_ms += duration;
                    stats.num_calls  += 1;
                }
            }
        );
        capturing = true;
    }
}

void PerformanceReport::stop_capture() {
    if (capturing) {
        debug::dali_function_start.deregister_callback(dali_function_start_handle);
        debug::dali_function_end.deregister_callback(dali_function_end_handle);

        capturing = false;
    }
}

void PerformanceReport::print(std::basic_ostream<char>& stream) {
    double all_functions_total_time_ms = 0.0;

    std::vector<std::string> keys;

    for (auto& kv: function_name_to_stats) {
        keys.emplace_back(kv.first);
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

    stream << "% total\tname\t#calls\ttotal time (s)" << std::endl;
    for(auto& fname: keys) {
        auto& stats = function_name_to_stats[fname];
        stream << stats.fraction_total * 100.0     << "%\t"
                  << fname                         << "\t"
                  << stats.num_calls               << "\t"
                  << stats.total_time_ms / 1000.0  << std::endl;
    }

    stream << "====================================================" << std::endl;

}
