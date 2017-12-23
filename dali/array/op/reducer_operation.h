#ifndef DALI_ARRAY_OP_REDUCER_OPERATION_H
#define DALI_ARRAY_OP_REDUCER_OPERATION_H

#include "dali/array/array.h"

namespace op {
    Array all_reduce(const Array& a,
                     const std::string& reducer_name);
    Array axis_reduce(const Array& a,
                      const std::string& reducer_name,
                      const std::vector<int>& axes,
                      bool keepdims=false);

    Array argument_all_reduce(const Array& a,
                              const std::string& reducer_name);
    Array argument_axis_reduce(const Array& a,
                               const std::string& reducer_name,
                               const int& axis);
}  // namespace op

#endif  // DALI_ARRAY_OP_REDUCER_OPERATION_H
