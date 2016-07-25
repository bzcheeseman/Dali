#ifndef DALI_UTILS_PERFORMANCE_REPORT_H
#define DALI_UTILS_PERFORMANCE_REPORT_H

#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <stack>
#include <vector>
#include <tuple>

#include "dali/utils/scope.h"

// Captures various statistics about runtime of
// various Dali functions.
// TODO(szymon): make thread safe
struct PerformanceReport {
    typedef std::chrono::system_clock        clock_t;
    typedef std::chrono::time_point<clock_t> time_point_t;

    struct ScopeStats {
        double total_time_ms;
        int    num_calls;
        time_point_t recent_start_time;
        std::unordered_map<std::string,
                           std::unique_ptr<ScopeStats>> children;
        ScopeStats();
    };

    PerformanceReport();
    // start capturing statistics about function calls
    // (may impact performance). May be called multiple times.
    void start_capture();
    // stop capturing statistics
    void stop_capture();
    // display summary of captured statistics
    void print(std::basic_ostream<char>& stream = std::cout);

  private:
    void on_enter(const ScopeObserver::State& state);
    void on_exit(const ScopeObserver::State& state);

    void print_scope(std::basic_ostream<char>& stream,
                     const std::string& name,
                     int max_name_len,
                     const ScopeStats& stats,
                     int indent = 0);

    std::shared_ptr<ScopeObserver> observer;

    std::mutex state_mutex;

    ScopeStats root_scope;
    std::stack<ScopeStats*> scope_stack;
};


#endif  // DALI_UTILS_PERFORMANCE_REPORT_H
