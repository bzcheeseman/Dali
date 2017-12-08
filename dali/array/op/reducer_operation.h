#ifndef DALI_ARRAY_OP_REDUCER_OPERATION_H
#define DALI_ARRAY_OP_REDUCER_OPERATION_H

#include "dali/array/array.h"
#include <string>
#include <vector>

namespace op {
    Array all_reduce(const Array& a,
                     const std::string& reducer_name);
    Array axis_reduce(const Array& a,
                      const std::string& reducer_name,
                      const std::vector<int>& axes);

    // Array argument_all_reduce(const Array& a,
    //                           const std::string& reducer_name);
    // Array argument_axis_reduce(const Array& a,
    //                            const std::string& reducer_name,
    //                            const int& axis);
}  // namespace op2

#endif  // DALI_ARRAY_OP_REDUCER_OPERATION_H
