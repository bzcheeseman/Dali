#ifndef DALI_ARRAY_EXPRESSION_OPTIMIZATION_H
#define DALI_ARRAY_EXPRESSION_OPTIMIZATION_H

#include <functional>
#include "dali/array/array.h"

// gets the right-hand side arguments from an assignment node.
const std::vector<Array>& right_args(Array node);

int register_optimization(std::function<bool(const Array&)> condition,
					      std::function<Array(const Array&)> transformation,
					      const std::string& name);

// simplify internal expression graph to return to a more common view.
Array canonical(const Array& array);

#endif  // DALI_ARRAY_EXPRESSION_OPTIMIZATION_H
