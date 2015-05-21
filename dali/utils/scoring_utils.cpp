#include "dali/utils/scoring_utils.h"

using std::vector;
using std::string;

namespace utils {
    ConfusionMatrix::ConfusionMatrix(int classes, const vector<string>& _names) : names(_names), totals(classes) {
        for (int i = 0; i < classes;++i) {
            grid.emplace_back(classes);
        }
    }
    void ConfusionMatrix::classified_a_when_b(int a, int b) {
        // update the misclassification:
        grid[b][a] += 1;
        // update the stakes:
        totals[b]  += 1;
    };

    void ConfusionMatrix::report() const {
        std::cout << "\nConfusion Matrix\n\t";
        for (auto & name : names) {
            std::cout << name << "\t";
        }
        std::cout << "\n";
        auto names_ptr = names.begin();
        auto totals_ptr = totals.begin();
        for (auto& category : grid) {
            std::cout << *names_ptr << "\t";
            for (auto & el : category) {
                std::cout << std::fixed
                          << std::setw(4)
                          << std::setprecision(2)
                          << std::setfill(' ')
                          << ((*totals_ptr) > 0 ? (100.0 * ((double) el / (double)(*totals_ptr))) : 0.0)
                          << "%\t";
            }
            std::cout << "\n";
            names_ptr++;
            totals_ptr++;
        }
    }

    Accuracy& Accuracy::true_positive(const int& _tp) {
        tp = _tp;
        return *this;
    }
    Accuracy& Accuracy::true_negative(const int& _tn) {
        tn = _tn;
        return *this;
    }
    Accuracy& Accuracy::false_positive(const int& _fp) {
        fp = _fp;
        return *this;
    }
    Accuracy& Accuracy::false_negative(const int& _fn) {
        fn = _fn;
        return *this;
    }

    int Accuracy::true_positive() const {
        return tp;
    }
    int Accuracy::true_negative() const {
        return tn;
    }
    int Accuracy::false_positive() const {
        return fp;
    }
    int Accuracy::false_negative() const {
        return fn;
    }

    double Accuracy::precision() const {
        return (double) tp / ((double)(tp + fp));
    }
    double Accuracy::recall() const {
        return (double) tp / ((double)(tp + fn));
    }
    double Accuracy::F1() const {
        auto P = precision();
        auto R = recall();
        return (2.0 * P * R) / (P + R);
    }

    template<typename T>
    double pearson_correlation(const std::vector<T>& x, const std::vector<T>& y) {
        utils::assert2(y.size() == x.size(), "Not an equal number of abscissa and ordinates.");

        double avg_x = 0;
        int    total = x.size();
        for (auto& datapoint : x) avg_x += datapoint;
        avg_x /= total;

        double avg_y = 0;
        for (auto& prediction : y) avg_y += prediction;
        avg_y /= total;

        double xdiff        = 0;
        double ydiff        = 0;
        double xdiff_square = 0;
        double ydiff_square = 0;
        double diffprod     = 0;

        for (int example_idx = 0; example_idx < total; example_idx++) {
            xdiff = x[example_idx] - avg_x;
            ydiff = y[example_idx] - avg_y;
            diffprod += xdiff * ydiff;
            xdiff_square += xdiff * xdiff;
            ydiff_square += ydiff * ydiff;
        }
        if (xdiff_square == 0 || ydiff_square == 0) return 0.0;
        return diffprod / std::sqrt(xdiff_square * ydiff_square);
    }

    template double pearson_correlation(const std::vector<float>& x, const std::vector<float>& y);
    template double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y);
}
