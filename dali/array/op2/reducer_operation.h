#ifndef DALI_ARRAY_OP2_REDUCER_OPERATION_H
#define DALI_ARRAY_OP2_REDUCER_OPERATION_H

#include <string>
#include <vector>

#include "dali/array/op2/expression/expression.h"

namespace op {
    expression::Expression all_reduce(const expression::Expression& a,
                         const std::string& reducer_name);
    expression::Expression argument_all_reduce(const expression::Expression& a,
                                  const std::string& reducer_name);
    expression::Expression axis_reduce(const expression::Expression& a,
                          const std::string& reducer_name,
                          const std::vector<int>& axes);
    expression::Expression argument_axis_reduce(const expression::Expression& a,
                                   const std::string& reducer_name,
                                   const int& axis);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_REDUCER_OPERATION_H
