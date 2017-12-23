#include "dtype.h"

#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"

template<>
DType template_to_dtype<float>() {
    return DTYPE_FLOAT;
}

template<>
DType template_to_dtype<double>() {
    return DTYPE_DOUBLE;
}

template<>
DType template_to_dtype<int>() {
    return DTYPE_INT32;
}

int size_of_dtype(DType dtype) {
    if (dtype == DTYPE_FLOAT) {
        return sizeof(float);
    } else if (dtype == DTYPE_DOUBLE) {
        return sizeof(double);
    } else if (dtype == DTYPE_INT32) {
        return sizeof(int);
    }
    ASSERT2(false, utils::make_message(
        "size_to_dtype only accepts " DALI_ACCEPTABLE_DTYPE_STR " but got ", int(dtype), "."));
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

std::string dtype_to_name(DType dtype) {
    switch(dtype) {
        case DTYPE_FLOAT:
            return "float";
        case DTYPE_DOUBLE:
            return "double";
        case DTYPE_INT32:
            return "int32";
        default:
            ASSERT2(false, utils::make_message("dtype_to_name called with incorrect DType (", int(dtype), ")."));
            return "unknown";
    }
}

std::string dtype_to_cpp_name(DType dtype) {
    switch(dtype) {
        case DTYPE_FLOAT:
            return "float";
        case DTYPE_DOUBLE:
            return "double";
        case DTYPE_INT32:
            return "int";
        default:
            ASSERT2(false, "dtype_to_name called with incorrect DType.");
            return "unknown";
    }
}

std::ostream& operator<<(std::ostream& stream, const DType& dtype) {
    return stream << dtype_to_name(dtype);
}

namespace std {
    std::size_t hash<DType>::operator()(const DType& dtype) const {
        return dtype;
    }
}
