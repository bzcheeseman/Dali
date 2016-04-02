#ifndef DALI_ARRAY_MSHADOW_INT_OPS_H
#define DALI_ARRAY_MSHADOW_INT_OPS_H

#include <mshadow/tensor.h>
#include <stdexcept>

namespace mshadow {
namespace expr {

template<typename SV, typename xpu,
                 bool transpose_left, bool transpose_right>
struct DotEngine<SV, xpu, 2, 2, 2, transpose_left, transpose_right, int> {
    inline static void Eval(Tensor<xpu, 2, int> *p_dst,
                                                    const Tensor<xpu, 2, int> &lhs,
                                                    const Tensor<xpu, 2, int> &rhs,
                                                    int scale) {
        throw std::runtime_error("Dot product is not implemented for integers.");
    }
};
template<typename SV, typename xpu, bool transpose_right>
struct DotEngine<SV, xpu, 1, 1, 2, false, transpose_right, int> {
    inline static void Eval(Tensor<xpu, 1, int> *p_dst,
                                                    const Tensor<xpu, 1, int> &lhs,
                                                    const Tensor<xpu, 2, int> &rhs,
                                                    int scale) {
        throw std::runtime_error("Dot product is not implemented for integers.");
    }
};
template<typename SV, typename xpu>
struct DotEngine<SV, xpu, 2, 1, 1, true, false, int> {
    inline static void Eval(Tensor<xpu, 2, int> *p_dst,
                                                    const Tensor<xpu, 1, int> &lhs,
                                                    const Tensor<xpu, 1, int> &rhs,
                                                    int scale) {
        throw std::runtime_error("Dot product is not implemented for integers.");
    }
};

} // expr
} // mshadow
#endif
