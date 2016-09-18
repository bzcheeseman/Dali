#include "binary.h"
#include <unordered_map>
#include <string>

#include "dali/array/op2/fused_operation.h"

namespace op2 {
    FusedOperation add(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::add");
    }

    FusedOperation sub(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::sub");
    }

    FusedOperation eltmul(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::eltmul");
    }

    FusedOperation eltdiv(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::eltdiv");
    }

    FusedOperation pow(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::power");
    }

    FusedOperation equals(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::equals");
    }

    FusedOperation prelu(const FusedOperation& x, const FusedOperation& weights) {
        return elementwise(x, weights, "functor::prelu");
    }

    FusedOperation circular_convolution(const FusedOperation& x, const FusedOperation& weights) {
        return binary_kernel_function(
            x,
            weights,
            "circular_convolution_kernel",
            "template<template <typename,int> class C1,\n"
            "         template <typename,int> class C2,\n"
            "         typename T, int ndim>\n"
            "XINLINE T circular_convolution_kernel(\n"
            "         const C1<T, ndim> a_view,\n"
            "         const C2<T, ndim> b_view,\n"
            "         Shape<ndim> query) {\n"
            "    T res = static_cast<T>(0);\n"
            "    const int conv_size = b_view.shape()[ndim - 1];\n"
            "    const int x = query[ndim - 1];\n"
            "    Shape<ndim> a_query = query;\n"
            "    Shape<ndim> b_query = query;\n"
            "    int& shift_idx = b_query[ndim - 1];\n"
            "    int& offset = a_query[ndim - 1];\n"
            "    for (shift_idx = 0; shift_idx < conv_size; shift_idx++) {\n"
            "        offset = x + shift_idx;\n"
            "        if (offset >= conv_size) {\n"
            "            offset -= conv_size;\n"
            "        }\n"
            "        res += a_view[a_query] * b_view[b_query];\n"
            "    }\n"
            "    return res;\n"
            "}\n"
        );
    }
}
