#include "dali/runtime_config.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"

AssignableArray scalar_add(const Array& x, const double& other) {
    return (x + other).as_assignable();
}
AssignableArray scalar_add(const Array& x, const float& other) {
    return (x + other).as_assignable();
}
AssignableArray scalar_add(const Array& x, const int& other) {
    return (x + other).as_assignable();
}
