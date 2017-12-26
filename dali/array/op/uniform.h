#ifndef DALI_ARRAY_EXPRESSION_UNIFORM_H
#define DALI_ARRAY_EXPRESSION_UNIFORM_H

#include "dali/array/array.h"

namespace op {
	Array uniform(Array low, Array high, const std::vector<int>& shape);
}

#endif  // DALI_ARRAY_EXPRESSION_UNIFORM_H
