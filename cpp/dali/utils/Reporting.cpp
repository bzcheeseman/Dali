#include "Reporting.h"

#include "dali/models/RecurrentEmbeddingModel.h"

typedef Throttled::Clock Clock;
using std::atomic;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::mutex;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

DEFINE_string(save, "", "Where to save model to ?");
DEFINE_string(load, "", "Where to save model to ?");
DEFINE_int32(save_frequency_s, 60, "How often to save model (in seconds) ?");

void Throttled::maybe_run(Clock::duration time_between_actions, std::function<void()> f) {
    std::lock_guard<decltype(lock)> lg(lock);
    if (Clock::now() - last_report >= time_between_actions) {
        f();
        last_report = Clock::now();
    }
}

template<typename T>
ReportProgress<T>::ReportProgress(string name,
               const double& total_work,
               Clock::duration report_frequency) :
        name(name),
        total_work(total_work),
        report_frequency(report_frequency) {
}

template<typename T>
void ReportProgress<T>::tick(const double& completed_work, T work) {
    if (printing_on) {
        t.maybe_run(report_frequency, [this, &completed_work, &work]() {
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
            ss << " " << work;
            max_line_length = std::max(ss.str().size(), max_line_length);
            std::cout << ss.str();
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

