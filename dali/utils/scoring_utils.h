#ifndef SCORING_UTILS_DALI_H
#define SCORING_UTILS_DALI_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

namespace utils {
    class ConfusionMatrix {
        public:
            std::vector<std::vector<std::atomic<int>>> grid;
            std::vector<std::atomic<int>> totals;
            const std::vector<std::string>& names;
            ConfusionMatrix(int classes, const std::vector<std::string>& _names);
            void classified_a_when_b(int a, int b);
            void report() const;
    };

    struct Accuracy {
        int tp;
        int fp;
        int tn;
        int fn;
        Accuracy& true_positive(const int& _tn);
        Accuracy& true_negative(const int& _tn);
        Accuracy& false_positive(const int& _tn);
        Accuracy& false_negative(const int& _tn);
        double recall() const;
        double precision() const;
        double F1() const;
    };
}

#endif
