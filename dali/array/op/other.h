#ifndef DALI_ARRAY_OP_OTHER_H
#define DALI_ARRAY_OP_OTHER_H

class Array;
template<typename OutType>
struct Assignable;

#include "dali/array/op2/expression/expression.h"

namespace op {
    expression::Expression any_isnan(const expression::Expression& array);
    expression::Expression any_isinf(const expression::Expression& array);
    expression::Expression any_isnan(const expression::Expression& array, int axis);
    expression::Expression any_isinf(const expression::Expression& array, int axis);
    expression::Expression all_equals(const expression::Expression& left, const expression::Expression& right);
    expression::Expression all_close(const expression::Expression& left, const expression::Expression& right, const double& atolerance);
    Assignable<Array> argsort(const Array& array, int axis);
    Assignable<Array> argsort(const Array& array);
}

#endif
