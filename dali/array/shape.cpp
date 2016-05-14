#include "shape.h"

#include <cstdlib>

std::vector<int> bshape2shape(const std::vector<int>& bshape) {
    std::vector<int> res(bshape.size(), 0);
    for (int i = 0; i < bshape.size(); ++i) {
        res[i] = std::abs(bshape[i]);
    }
    return res;
}

std::vector<int> shape_to_trivial_strides(const std::vector<int>& shape) {
    std::vector<int> res(shape.size());
    int residual_shape = 1;
    for (int i = shape.size() - 1; i >= 0 ; --i) {
        res[i] = residual_shape;
        residual_shape *= shape[i];
    }
    return res;
}
