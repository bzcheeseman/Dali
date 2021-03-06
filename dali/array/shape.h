#ifndef DALI_ARRAY_SHAPE_H
#define DALI_ARRAY_SHAPE_H

#include "dali/macros.h"
#include <vector>

std::vector<int> collapsed_shape(const std::vector<int>& shape, int ndim);
int hypercube_volume(const std::vector<int>& shape);
std::vector<int> shape_to_trivial_strides(const std::vector<int>& shape);

#endif  // DALI_ARRAY_SHAPE_H
