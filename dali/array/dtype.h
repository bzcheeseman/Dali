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
void assert_dali_dtype() {
    static_assert(std::is_same<T,float>::value ||
                  std::is_same<T,double>::value ||
                  std::is_same<T,int>::value,
            "must be used with float, double or int.");
}

template<typename T>
DType template_to_dtype() {
    assert_dali_dtype<T>();
   	return DTYPE_FLOAT;
}

template<> DType template_to_dtype<float>();
template<> DType template_to_dtype<double>();
template<> DType template_to_dtype<int>();


int size_of_dtype(DType dtype);

void print_dtype(std::basic_ostream<char>& stream, DType dtype, void* memory);

std::string dtype_to_name(DType dtype);
std::ostream& operator<<(std::ostream&, const DType&);

#endif
