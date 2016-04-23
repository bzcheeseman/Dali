#include "dali/array/op/reducers.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"

AssignableArray sum_all(const Array& x) {
    return lazy::sum_all(x).as_assignable();
}

AssignableArray mean_all(const Array& x) {
    return lazy::sum_all(x).as_assignable();
}
