#include "shape.h"

#include <algorithm>
#include <cstdlib>
#include <vector>
#include <functional>
#include <numeric>

std::vector<int> collapsed_shape(const std::vector<int>& shape, int ndim) {
    if (shape.size() > 0) {
        std::vector<int> newshape(ndim);
        newshape[0] = shape[0];
        for (int i = 1; i < shape.size() - ndim + 1; i++) {
            newshape[0] *= shape[i];
        }
        for (int i = 1; i < ndim; i++) {
            newshape[i] = shape[shape.size() - ndim + i];
        }
        return newshape;
    } else {
        return {};
    }
}

int hypercube_volume(const std::vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
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


