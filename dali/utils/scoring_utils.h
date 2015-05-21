#ifndef SCORING_UTILS_DALI_H
#define SCORING_UTILS_DALI_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <atomic>
#include "dali/utils/core_utils.h"

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
        int tp = 0;
        int fp = 0;
        int tn = 0;
        int fn = 0;
        Accuracy& true_positive(const int& _tp);
        Accuracy& true_negative(const int& _tn);
        Accuracy& false_positive(const int& _fp);
        Accuracy& false_negative(const int& _fn);

        int true_positive()  const;
        int true_negative()  const;
        int false_positive() const;
        int false_negative() const;

        double recall() const;
        double precision() const;
        double F1() const;
    };

    template<typename T>
    double pearson_correlation(const std::vector<T>& x, const std::vector<T>& y);
}

#endif
