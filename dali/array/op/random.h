#ifndef DALI_ARRAY_OP_RANDOM_H
#define DALI_ARRAY_OP_RANDOM_H

#include "dali/array/array.h"

namespace random {

    AssignableArray gaussian(double mean, double std);
    AssignableArray uniform(double low, double high);

}

#endif
