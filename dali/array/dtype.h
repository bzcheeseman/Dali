#ifndef DALI_ARRAY_DTYPE_H
#define DALI_ARRAY_DTYPE_H

#include <ostream>


enum DType {
    DTYPE_FLOAT  = 0,
    DTYPE_DOUBLE = 1,
    DTYPE_INT32  = 2
};

// macro for including the human-readable acceptable dtypes
// in Dali
#define DALI_ACCEPTABLE_DTYPE_STR "float, double, or int32"


template<typename T>
bool dtype_is(DType dtype) {
    return false;
}

template<> bool dtype_is<float>(DType dtype);
template<> bool dtype_is<double>(DType dtype);
template<> bool dtype_is<int>(DType dtype);

int size_of_dtype(DType dtype);

void print_dtype(std::basic_ostream<char>& stream, DType dtype, void* memory);


#endif
