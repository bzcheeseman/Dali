#ifndef UTILS_REPORTING_H
#define UTILS_REPORTING_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <gflags/gflags.h>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

class Throttled {
    std::chrono::high_resolution_clock::time_point last_report;
    std::mutex lock;

    public:
        typedef std::chrono::high_resolution_clock Clock;
        void maybe_run(Clock::duration time_between_actions, std::function<void()> f);
};

template<typename T>
class ReportProgress {
    std::mutex lock;

    static const int RESOLUTION = 30;
    Throttled t;
    std::string name;
    double total_work;
    size_t max_line_length = 0;
    Throttled::Clock::duration report_frequency;
    bool printing_on = true;

    Throttled::Clock::time_point last_tick;
    double last_completed_work_report;
    std::chrono::duration<double, std::ratio<1l, 1000000000l> > estimated_total_time;


    void finish_line(const std::string& text);
    public:
        ReportProgress(std::string name,
                       const double& total_work,
                       Throttled::Clock::duration report_frequency=std::chrono::milliseconds(250));
        void tick(const double& completed_work);
        void tick(const double& completed_work, T extra_info);
        void tick(const double& completed_work, std::string extra_info);
        void pause();
        void resume();
        void done();
};

#endif
