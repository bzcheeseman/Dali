#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>


using Clock = std::chrono::high_resolution_clock;
using std::atomic;
using std::chrono::milliseconds;
using std::mutex;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

class Throttled {
    Clock::time_point last_report;
    mutex lock;

    public:
        void maybe_run(Clock::duration time_between_actions, std::function<void()> f);
};

class ReportProgress {
    static const int RESOLUTION = 15;
    Throttled t;
    string name;
    double total_work;
    Clock::duration report_frequency;

    public:
        ReportProgress(string name,
                       const double& total_work,
                       Clock::duration report_frequency=milliseconds(250));

        void tick(const double& completed_work);
        void done();
};
