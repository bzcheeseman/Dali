#ifndef DALI_MAT_MATH_WEIGHTS_H
#define DALI_MAT_MATH_WEIGHTS_H

#include <functional>

#include "dali/config.h"
#include "dali/array/array.h"
#include "dali/utils/assert2.h"


namespace weights {
    typedef Array sync_t;
    typedef std::function<void(sync_t&)> initializer_t;

    initializer_t empty();
    initializer_t zeros();
    initializer_t ones();
    initializer_t uniform(double lower, double upper);
    initializer_t uniform(double bound);
    initializer_t gaussian(double mean, double std);
    initializer_t gaussian(double std);
    // bernoulli random samples
    initializer_t bernoulli(double prob);
    // bernoulli random samples, multiplied by 1.0 / prob
    initializer_t bernoulli_normalized(double prob);
    initializer_t eye(double diag = 1.0);

    // Preinitializer is first run on the matrix and then SVD initialization
    // happens. If you are unsure what kind of preinitializer to use, then try
    // weights<R>::uniform(m), were m is the number of columns in your matrix.
    // DISCLAIMER: do not use on big matrices (like embeddings) - faster
    // version is a subject of current research.
    initializer_t svd(initializer_t preinitializer = gaussian(1.0));
}

#endif
