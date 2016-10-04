#ifndef DALI_ARRAY_OP2_REDUCER_OPERATION_H
#define DALI_ARRAY_OP2_REDUCER_OPERATION_H

#include <string>
#include <vector>

struct Operation;

namespace op2 {
    Operation all_reduce(const Operation& a,
                         const std::string& reducer_name);
    Operation argument_all_reduce(const Operation& a,
                                  const std::string& reducer_name);
    Operation axis_reduce(const Operation& a,
                          const std::string& reducer_name,
                          const std::vector<int>& axes);
    Operation argument_axis_reduce(const Operation& a,
                                   const std::string& reducer_name,
                                   const int& axis);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_REDUCER_OPERATION_H
