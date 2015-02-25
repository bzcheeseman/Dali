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

class Throttled {
    std::chrono::high_resolution_clock::time_point last_report;
    std::mutex lock;

    public:
        typedef std::chrono::high_resolution_clock Clock;
        void maybe_run(Clock::duration time_between_actions, std::function<void()> f);
};

template<typename T>
class ReportProgress {
    static const int RESOLUTION = 30;
    Throttled t;
    std::string name;
    double total_work;
    Throttled::Clock::duration report_frequency;

    public:
        ReportProgress(std::string name,
                       const double& total_work,
                       Throttled::Clock::duration report_frequency=std::chrono::milliseconds(250));

        void tick(const double& completed_work, T work);
        void done();
};

static Throttled model_save_throttled;
static int model_snapshot_no;

template<typename T> void maybe_save_model(const T& model,
                                           const std::string& base_path="",
                                           const std::string& label="");
