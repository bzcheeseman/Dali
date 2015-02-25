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

#include "core/StackedModel.h"
#include "core/StackedShortcutModel.h"

DECLARE_string(save);
DECLARE_string(load);
DECLARE_int32(save_frequency_s);

using Clock = std::chrono::high_resolution_clock;
using std::atomic;
using std::chrono::milliseconds;
using std::chrono::seconds;
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

template<typename T>
class ReportProgress {
    static const int RESOLUTION = 30;
    Throttled t;
    string name;
    double total_work;
    Clock::duration report_frequency;

    public:
        ReportProgress(string name,
                       const double& total_work,
                       Clock::duration report_frequency=milliseconds(250));

        void tick(const double& completed_work, T work);
        void done();
};

static Throttled model_save_throttled;
static int model_snapshot_no;

template<typename T> void maybe_save_model(const T& model,
                                           const string& base_path="",
                                           const string& label="");
