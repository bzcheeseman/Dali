#include "Reporting.h"

void Throttled::maybe_run(Clock::duration time_between_actions, std::function<void()> f) {
    std::lock_guard<decltype(lock)> lg(lock);
    if (Clock::now() - last_report >= time_between_actions) {
        f();
        last_report = Clock::now();
    }
}

ReportProgress::ReportProgress(string name,
               const double& total_work,
               Clock::duration report_frequency) :
        name(name),
        total_work(total_work),
        report_frequency(report_frequency) {
}

void ReportProgress::tick(const double& completed_work) {
    t.maybe_run(report_frequency, [this, &completed_work]() {
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
                   << std::setfill( ' ' ) <<  completed_work/total_work << "%";
        std::cout << ss.str();

        std::cout.flush();
    });
}

void ReportProgress::done() {
    std::cout << "\r" << name << " done                          " << std::endl;
}
