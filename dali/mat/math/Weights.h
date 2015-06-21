#ifndef DALI_MAT_MATH_WEIGHTS_H
#define DALI_MAT_MATH_WEIGHTS_H

#include <functional>
#include "dali/utils/core_utils.h"

template<typename R, int dimension>
class SynchronizedMemory;

template<typename R>
struct weights {
    typedef SynchronizedMemory<R,2> sync_t;

    typedef std::function<void(sync_t&)> initializer_t;

    static initializer_t empty();
    static initializer_t zeros();
    static initializer_t uniform(R lower, R upper);
    static initializer_t uniform(R bound);
    static initializer_t gaussian(R mean, R std);
    static initializer_t gaussian(R std);
    static initializer_t eye(R diag = 1.0);

    // Preinitializer is first run on the matrix and then SVD initialization
    // happens. If you are unsure what kind of preinitializer to use, then try
    // weights<R>::uniform(m), were m is the number of columns in your matrix.
    // DISCLAIMER: do not use on big matrices (like embeddings) - faster version
    // is a subject of current research.
    static initializer_t svd(initializer_t preinitializer = gaussian(1.0));
};

#endif
