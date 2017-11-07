#ifndef DALI_ARRAY_OP2_REDUCER_OPERATION_H
#define DALI_ARRAY_OP2_REDUCER_OPERATION_H

#include <string>
#include <vector>

namespace op {
    expression::ExpressionGraph all_reduce(const expression::ExpressionGraph& a,
                         const std::string& reducer_name);
    expression::ExpressionGraph argument_all_reduce(const expression::ExpressionGraph& a,
                                  const std::string& reducer_name);
    expression::ExpressionGraph axis_reduce(const expression::ExpressionGraph& a,
                          const std::string& reducer_name,
                          const std::vector<int>& axes);
    expression::ExpressionGraph argument_axis_reduce(const expression::ExpressionGraph& a,
                                   const std::string& reducer_name,
                                   const int& axis);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_REDUCER_OPERATION_H
