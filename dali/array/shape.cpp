#include "shape.h"
#include <cstdlib>

std::vector<int> bshape2shape(const std::vector<int>& bshape) {
    std::vector<int> res(bshape.size(), 0);
    for (int i = 0; i < bshape.size(); ++i) {
        res[i] = std::abs(bshape[i]);
    }
    return res;
}
