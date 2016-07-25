#include "performance_report.h"

#include <algorithm>
#include <iomanip>

#include "dali/utils/assert2.h"

using namespace std::placeholders;

PerformanceReport::ScopeStats::ScopeStats() :
        total_time_ms(0),
        num_calls(0) {
    children.clear();
}

PerformanceReport::PerformanceReport() {
    scope_stack.emplace(&root_scope);
}

void PerformanceReport::start_capture() {
    if (observer == nullptr) {
        root_scope.recent_start_time = clock_t::now();
        observer = std::make_shared<ScopeObserver>(
            std::bind(&PerformanceReport::on_enter, this, _1),
            std::bind(&PerformanceReport::on_exit, this, _1)
        );
    }
}

void PerformanceReport::stop_capture() {
    if (observer != nullptr) {
        observer.reset();
        auto duration =  std::chrono::duration_cast<std::chrono::milliseconds>(
                clock_t::now() - root_scope.recent_start_time).count();
        root_scope.total_time_ms += duration;
        root_scope.num_calls     += 1;
    }
}

void PerformanceReport::print(std::basic_ostream<char>& stream) {
    stream << "============== PERFORMANCE REPORT ==================" << std::endl;

    stream << "% total | name | #calls | total time" << std::endl;
    print_scope(stream, "root", 4, root_scope);
    stream << "====================================================" << std::endl;
}

void print_scope_helper(std::basic_ostream<char>& stream,
                        int indent,
                        double fraction_total,
                        const std::string& name,
                        int max_name_len,
                        int num_calls,
                        double total_time_ms) {
    stream    << std::string(indent, ' ') << "+"
              << std::fixed << std::setprecision(2) << std::setw(6)
              << fraction_total * 100.0        << " %  "
              << std::left << std::setw(max_name_len + 2)
              << name                          << "  "
              << std::right << std::setw(7)
              << ((num_calls == -1) ? "N/A" : std::to_string(num_calls)) << "   "
              << std::setprecision(3) << std::setw(7)
              << total_time_ms / 1000.0  << " s"
              << std::endl;
}

void PerformanceReport::print_scope(std::basic_ostream<char>& stream,
                                    const std::string& name,
                                    int max_name_len,
                                    const ScopeStats& stats,
                                    int indent) {
    const auto not_tracked_name = std::string("(not tracked)");
    double fraction_total   = stats.total_time_ms / root_scope.total_time_ms;

    print_scope_helper(stream, indent, fraction_total, name, max_name_len,
                       stats.num_calls, stats.total_time_ms);

    std::vector<std::string> keys;
    double time_not_tracked = stats.total_time_ms;

    int max_child_name_len = 0;
    for (auto& kv: stats.children) {
        keys.emplace_back(kv.first);
        max_child_name_len = std::max(max_child_name_len, (int)kv.first.size());
        time_not_tracked -= kv.second->total_time_ms;
    }
    bool should_print_not_tracked = keys.size() > 0 && time_not_tracked > 0.01;
    if (should_print_not_tracked) {
        max_child_name_len = std::max(max_child_name_len, (int)not_tracked_name.size());
    }

    std::sort(keys.begin(), keys.end(),
              [&](const std::string& key1, const std::string key2) {
                  return stats.children.at(key1)->total_time_ms >
                         stats.children.at(key2)->total_time_ms;
              }
    );

    for (auto& key : keys) {
        print_scope(stream, key, max_child_name_len, *stats.children.at(key), indent + 2);
    }
    if (should_print_not_tracked) {
        fraction_total = time_not_tracked / root_scope.total_time_ms;
        print_scope_helper(stream, indent + 2, fraction_total,
                           not_tracked_name, max_child_name_len,
                           -1, time_not_tracked);
    }
}


void PerformanceReport::on_enter(const ScopeObserver::State& state) {
    std::lock_guard<decltype(state_mutex)> guard(state_mutex);

    if (state.trace.size() != scope_stack.size()) {
        // this may happen if we launch performance report mid execution.
        // in that case recording will commence once we exit back to the
        // root scope again.
        return;
    }

    auto& parent_scope = *scope_stack.top();
    const auto& scope_name   = *state.trace.back();

    if (!((bool)parent_scope.children[scope_name])) {
        parent_scope.children[scope_name] = std::unique_ptr<ScopeStats>(new ScopeStats());
    }

    scope_stack.emplace(parent_scope.children[scope_name].get());
    parent_scope.children[scope_name]->recent_start_time = clock_t::now();
}

void PerformanceReport::on_exit(const ScopeObserver::State& state) {
    std::lock_guard<decltype(state_mutex)> guard(state_mutex);

    if (state.trace.size() + 1 != scope_stack.size()) {
        // this may happen if we launch performance report mid execution.
        // in that case recording will commence once we exit back to the
        // root scope again.
        return;
    }

    auto duration =  std::chrono::duration_cast<std::chrono::milliseconds>(
                clock_t::now() - scope_stack.top()->recent_start_time).count();

    scope_stack.top()->total_time_ms += duration;
    scope_stack.top()->num_calls     += 1;

    scope_stack.pop();
}
