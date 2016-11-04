#ifndef DALI_ARRAY_OP2_REDUCER_OPERATION_H
#define DALI_ARRAY_OP2_REDUCER_OPERATION_H

#include <string>
#include <vector>

#include "dali/array/op2/expression/expression.h"

namespace op {
    Expression all_reduce(const Expression& a,
                         const std::string& reducer_name);
    Expression argument_all_reduce(const Expression& a,
                                  const std::string& reducer_name);
    Expression axis_reduce(const Expression& a,
                          const std::string& reducer_name,
                          const std::vector<int>& axes);
    Expression argument_axis_reduce(const Expression& a,
                                   const std::string& reducer_name,
                                   const int& axis);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_REDUCER_OPERATION_H
