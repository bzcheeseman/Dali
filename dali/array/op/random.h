#ifndef DALI_ARRAY_OP_RANDOM_H
#define DALI_ARRAY_OP_RANDOM_H

#include "dali/array/array.h"

namespace op {
	Array uniform(Array low, Array high, const std::vector<int>& shape);
	Array normal(Array loc, Array scale, const std::vector<int>& shape);
    Array bernoulli(Array prob, const std::vector<int>& shape);
    Array bernoulli_normalized(Array prob, const std::vector<int>& shape);
}

#endif  // DALI_ARRAY_OP_RANDOM_H
