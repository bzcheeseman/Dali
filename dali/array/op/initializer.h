#ifndef DALI_ARRAY_OP_INITIALIZER_H
#define DALI_ARRAY_OP_INITIALIZER_H

#include <vector>

class Array;
class AssignableArray;

namespace initializer {
    AssignableArray empty();
    AssignableArray zeros();
    AssignableArray ones();
    AssignableArray arange();
    template<typename T>
    AssignableArray fill(const T& constant);

    AssignableArray gaussian(const double& mean, const double& std);
    AssignableArray uniform(const double& low, const double& high);
    AssignableArray bernoulli(const double& prob);
    AssignableArray bernoulli_normalized(const double& prob);

    AssignableArray eye(const double& diag = 1.0);
    // Preinitializer is first run on the matrix and then SVD initialization
    // happens. If you are unsure what kind of preinitializer to use, then try
    // weights<R>::uniform(m), were m is the number of columns in your matrix.
    // DISCLAIMER: do not use on big matrices (like embeddings) - faster
    // version is a subject of current research.
    AssignableArray svd();
    AssignableArray svd(const AssignableArray& preinitializer);


} // namespace initializer
#endif
