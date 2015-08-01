#include "Reporting.h"

#include "dali/utils/print_time.h"
#include "dali/utils/core_utils.h"

typedef Throttled::Clock Clock;
using std::atomic;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::mutex;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;



void Throttled::maybe_run(Clock::duration time_between_actions, std::function<void()> f) {
    std::lock_guard<decltype(lock)> lg(lock);
    if (Clock::now() - last_report >= time_between_actions) {
        f();
        last_report = Clock::now();
    }
}

template<typename T>
ReportProgress<T>::ReportProgress(string _name,
               const double& _total_work,
               Clock::duration report_frequency) :
        name(_name),
        total_work(_total_work),
        report_frequency(report_frequency),
        last_completed_work_report(0.0),
        last_tick(Throttled::Clock::now()),
        estimated_total_time(seconds(0)) {
}

template<typename T>
void ReportProgress<T>::tick(const double& completed_work) {
    tick(completed_work, "");
}


template<typename T>
void ReportProgress<T>::tick(const double& completed_work, T extra_info) {
    tick(completed_work, std::to_string(extra_info));
}

template<typename T>
void ReportProgress<T>::tick(const double& completed_work, std::string extra_info) {
    {
        std::lock_guard<decltype(lock)> lg(lock);

        auto now = Throttled::Clock::now();

        auto time_since_last_tick      = now - last_tick;
        auto work_done_since_last_tick = (completed_work - last_completed_work_report) / total_work;

        last_tick                 = now;
        last_completed_work_report = completed_work;

        auto new_estimate  = time_since_last_tick / work_done_since_last_tick;

        const double forgetting = 0.1;
        estimated_total_time = estimated_total_time * (1.0 - forgetting) + new_estimate * forgetting;
    }
    if (printing_on) {
        t.maybe_run(report_frequency, [&]() {
            int active_bars = RESOLUTION * completed_work/total_work;
            std::stringstream ss;
            ss << "\r" << name << " [";
            for (int i = 0; i < RESOLUTION; ++i) {
                if (i < active_bars) {
                    ss << "█";
                } else {
                    ss << " ";
                }
            }
            ss << "] " << std::fixed
                       << std::setprecision( 3 ) // use 3 decimals
                       << std::setw(6)
                       << std::setfill( ' ' ) <<  100.0 * completed_work/total_work << "%";
            ss << " " << extra_info;
            auto eta = (1.0 - last_completed_work_report / total_work) * estimated_total_time;
            ss << " (ETA: " << utils::print_time(eta) << ")";
            max_line_length = std::max(ss.str().size(), max_line_length);
            std::cout << ss.str();
            for (int i = 0; i < max_line_length - ss.str().size(); ++i)
                std::cout << " ";
            std::cout.flush();
        });
    }
}

template<typename T>
void ReportProgress<T>::finish_line(const string& text) {
    std::cout << "\r" << name << ' ' <<  text;
    if (max_line_length > text.size() + 1 + name.size()) {
        for (int i = 0; i < std::max((size_t) 0, max_line_length - text.size() - 1 - name.size()); i++) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;
    max_line_length = 0;
}

template<typename T>
void ReportProgress<T>::done() {
    finish_line("done");
}

template<typename T>
void ReportProgress<T>::pause() {
    finish_line("");
    std::cout << "\r" << std::flush;
    printing_on = false;
}

template<typename T>
void ReportProgress<T>::resume() {
    printing_on = true;
}

template class ReportProgress<float>;
template class ReportProgress<double>;
