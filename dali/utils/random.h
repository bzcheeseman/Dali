#ifndef DALI_UTILS_RANDOM_H
#define DALI_UTILS_RANDOM_H

#include <algorithm>
#include <random>
#include <vector>
#include "dali/utils/assert2.h"

// Utilities related to random number generation
// seeds, etc..
namespace utils {
     /**
    randint
    -------
    Sample integers from a uniform distribution between (and including)
    lower and upper int values.

    Inputs
    ------
    int lower
    int upper

    Outputs
    -------
    int sample
    **/
    int randint(int lower, int upper);
    template<typename T>
    T randinteger(T lower, T upper);
    double randdouble(double lower=0.0, double upper=1.0);
    // for shufflign datasets
    std::vector<size_t> random_arange(size_t);
    std::vector<std::vector<size_t>> random_minibatches(size_t total_elements, size_t minibatch_size);
    // control randomness
    namespace random {
        // use random device to reseed the process
        void reseed();
        void set_seed(int new_seed);
        std::mt19937& generator();
    }
}

#endif
