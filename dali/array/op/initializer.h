#ifndef DALI_ARRAY_OP_INITIALIZER_H
#define DALI_ARRAY_OP_INITIALIZER_H

#include <vector>

class Array;
template<typename OutType>
class Assignable;

namespace initializer {
    Assignable<Array> empty();
    Assignable<Array> zeros();
    Assignable<Array> ones();
    Assignable<Array> arange();
    template<typename T>
    Assignable<Array> fill(const T& constant);

    Assignable<Array> gaussian(const double& mean, const double& std);
    Assignable<Array> uniform(const double& low, const double& high);
    Assignable<Array> bernoulli(const double& prob);
    Assignable<Array> bernoulli_normalized(const double& prob);

    Assignable<Array> eye(const double& diag = 1.0);
    // Preinitializer is first run on the matrix and then SVD initialization
    // happens. If you are unsure what kind of preinitializer to use, then try
    // weights<R>::uniform(m), were m is the number of columns in your matrix.
    // DISCLAIMER: do not use on big matrices (like embeddings) - faster
    // version is a subject of current research.
    Assignable<Array> svd();
    Assignable<Array> svd(const Assignable<Array>& preinitializer);


} // namespace initializer
#endif
