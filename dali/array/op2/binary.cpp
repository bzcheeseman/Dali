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

    FusedOperation prelu_backward_weights(const FusedOperation& a, const FusedOperation& grad) {
        return elementwise(a, grad, "functor::prelu_backward_weights");
    }

    FusedOperation prelu_backward_inputs(const FusedOperation& a, const FusedOperation& weights) {
        return elementwise(a, weights, "functor::prelu_backward_inputs");
    }

    FusedOperation lessthanequal(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::lessthanequal");
    }

    FusedOperation greaterthanequal(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::greaterthanequal");
    }

    FusedOperation eltmax(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::max_scalar");
    }

    FusedOperation clip(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::clip");
    }

    FusedOperation eltmin(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::min_scalar");
    }

    FusedOperation binary_cross_entropy(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::binary_cross_entropy");
    }

    FusedOperation binary_cross_entropy_grad(const FusedOperation& a, const FusedOperation& b) {
        return elementwise(a, b, "functor::binary_cross_entropy_grad");
    }

    FusedOperation circular_convolution(const FusedOperation& x, const FusedOperation& weights) {
        // TODO(jonathan, szymon): clearly kernel writing is repetitive, a method could
        //                         be designed here to factor out all the boilerplate
        //                         to instantiate easily 2, 3, etc... arg templates.
        return binary_kernel_function(
            x,
            weights,
            "circular_convolution_kernel",
            "template<typename C1, typename C2>\n"
            "struct CircularConvolutionKernel {\n"
            "    const C1& a_view_;\n"
            "    const C2& b_view_;\n"
            "    static const int ndim = C1::ndim;\n"
            "    typedef typename C1::T T;\n"
            "    XINLINE CircularConvolutionKernel(const C1& a_view, const C2& b_view)\n"
            "        : a_view_(a_view), b_view_(b_view) {}\n"
            "    XINLINE T operator[](Shape<ndim> query) {\n"
            "        T res = static_cast<T>(0);\n"
            "        const int conv_size = b_view_.shape()[ndim - 1];\n"
            "        const int x = query[ndim - 1];\n"
            "        Shape<ndim> a_query = query;\n"
            "        Shape<ndim> b_query = query;\n"
            "        int& shift_idx = b_query[ndim - 1];\n"
            "        int& offset = a_query[ndim - 1];\n"
            "        #pragma clang loop vectorize(enable)\n"
            "        #pragma clang loop interleave(enable)\n"
            "        for (shift_idx = 0; shift_idx < conv_size; shift_idx++) {\n"
            "            offset = x + shift_idx;\n"
            "            if (offset >= conv_size) {\n"
            "                offset -= conv_size;\n"
            "            }\n"
            "            res += a_view_[a_query] * b_view_[b_query];\n"
            "        }\n"
            "        return res;\n"
            "    }\n"
            "};\n"
            "template<typename C1, typename C2>\n"
            "CircularConvolutionKernel<C1, C2> circular_convolution_kernel(const C1& a, const C2& b) {\n"
            "    return CircularConvolutionKernel<C1, C2>(a, b);\n"
            "}\n"
        );
    }
}
