#include "dtype.h"

#include "dali/utils/assert2.h"

template<>
bool dtype_is<float>(DType dtype) {
    return dtype == DTYPE_FLOAT;
}

template<>
bool dtype_is<double>(DType dtype) {
    return dtype == DTYPE_DOUBLE;
}

template<> 
bool dtype_is<int>(DType dtype) {
    return dtype == DTYPE_INT32;
}



int size_of_dtype(DType dtype) {
    if (dtype == DTYPE_FLOAT) {
        return sizeof(float);
    } else if (dtype == DTYPE_DOUBLE) {
        return sizeof(double);
    } else if (dtype == DTYPE_INT32) {
        return sizeof(int);
    }
    ASSERT2(false, "size_to_dtype only accepts " DALI_ACCEPTABLE_DTYPE_STR);
    return -1;
}

void print_dtype(std::basic_ostream<char>& stream, DType dtype, void* memory) {
    if (dtype == DTYPE_FLOAT) {
        stream << *((float*)memory);
    } else if (dtype == DTYPE_DOUBLE) {
        stream << *((double*)memory);
    } else if (dtype == DTYPE_INT32) {
        stream << *((int*)memory);
    } else {
        ASSERT2(false, "print_dtype only accepts " DALI_ACCEPTABLE_DTYPE_STR);
    }
}
