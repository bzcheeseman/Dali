#include <chrono>
#include <iostream>
#include <tuple>

#include <iostream>

namespace utils {
    template<typename T>
    std::string print_time(T duration) {
        long long n_seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        if (n_seconds < 0) {
            return "error";
        } else if (n_seconds == 0) {
            return "soon";
        } else if (n_seconds < 10) {
            return "a few seconds";
        } else if (n_seconds < 60) {
            return "less than a minute";
        } else if (n_seconds < 120) {
            return "a minute";
        } else if (n_seconds < 60 * 60) {
            return "" + std::to_string(n_seconds / 60) + " minutes";
        } else {
            long long n_days = n_seconds / (24 * 60 * 60);
            n_seconds -= n_days * 24 * 60 * 60;
            long long n_hours = n_seconds / (60 * 60);
            n_seconds -= n_hours * 60 * 60;
            long long n_minutes = n_seconds / 60;

            std::string res = "";
            if (n_days > 0)
                res += (n_days == 1) ? "a day" : "" + std::to_string(n_days) + " days";
            if (n_days >= 7) return res;

            if (n_hours != 0) {
                res += " and ";
                res += (n_hours == 1) ? "an hour" : std::to_string(n_hours) + " hours";
            }

            if (n_days == 0) {
                res += " and ";
                res += (n_minutes == 1) ? "a minute" : std::to_string(n_minutes) + " minutes";
            }
            return res;
        }
    }
}
