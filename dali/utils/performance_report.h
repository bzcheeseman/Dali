#ifndef DALI_UTILS_PERFORMANCE_REPORT_H
#define DALI_UTILS_PERFORMANCE_REPORT_H

#include <chrono>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <tuple>

#include "dali/array/debug.h"

struct FunctionStats {
    double total_time_ms;
    int    num_calls;
    double fraction_total;
};

// Captures various statistics about runtime of
// various Dali functions.
// TODO(szymon): make thread safe
struct PerformanceReport {
    typedef std::chrono::system_clock        clock_t;
    typedef std::chrono::time_point<clock_t> time_point_t;

    // start capturing statistics about function calls
    // (may impact performance). May be called multiple times.
    void start_capture();
    // stop capturing statistics
    void stop_capture();
    // display summary of captured statistics
    void print(std::basic_ostream<char>& stream = std::cout);

  private:
    void on_enter(const debug::ScopeObserver::State& state);
    void on_exit(const debug::ScopeObserver::State& state);

    std::shared_ptr<debug::ScopeObserver> observer;

    std::mutex state_mutex;
    std::unordered_map<std::string, time_point_t> call_to_start_time;

    std::unordered_map<std::string, FunctionStats> function_name_to_stats;
};


#endif  // DALI_UTILS_PERFORMANCE_REPORT_H
