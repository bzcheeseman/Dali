#ifndef DALI_ARRAY_DTYPE_H
#define DALI_ARRAY_DTYPE_H

namespace dtype {
    enum Dtype {
        Float = 0,
        Double = 1,
        Int32 = 2
    };
}
// macro for including the human-readable acceptable dtypes
// in Dali
#define DALI_ACCEPTABLE_DTYPE_STR "float, double, or int32"

#endif
