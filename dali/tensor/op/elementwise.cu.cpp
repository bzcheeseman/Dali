#include "dali/tensor/op/elementwise.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

using namespace TensorOps;
using std::vector;

namespace matops {
    #define DALI_UNARY_OP0(name, forward_op, backward) \
        template<typename R>                                                                                  \
        Mat<R> Elementwise<R>::name(Mat<R> matrix) {                                                          \
            auto out = Mat<R>::empty_like(matrix);                                                            \
                                                                                                              \
            MAT(out) = F<forward_op<R>>(MAT(matrix).wrapper());                                               \
                                                                                                              \
            if (graph::backprop_enabled() && !matrix.constant)                                                  \
                graph::emplace_back([matrix, out]() mutable {                                                 \
                    GRAD(matrix) += (backward) * GRAD(out).wrapper();                                         \
                });                                                                                           \
            return out;                                                                                       \
        }

    #define DALI_UNARY_OP1(name, arg1, forward_op, backward) \
        template<typename R>                                                                                  \
        Mat<R> Elementwise<R>::name(Mat<R> matrix, R arg1) {                                                  \
            auto out = Mat<R>::empty_like(matrix);                                                            \
                                                                                                              \
            MAT(out) = F<forward_op<R>>(MAT(matrix).wrapper(), arg1);                                         \
                                                                                                              \
            if (graph::backprop_enabled() && !matrix.constant)                                                  \
                graph::emplace_back([matrix, out, arg1]() mutable {                                           \
                    GRAD(matrix) += (backward) * GRAD(out).wrapper();                                         \
                });                                                                                           \
            return out;                                                                                       \
        }

    DALI_UNARY_OP0(tanh, op::tanh,
            F<op::dtanh<R>>(MAT(out).wrapper()));
    DALI_UNARY_OP0(softplus, op::softplus,
            F<op::softplus_backward<R>>(MAT(matrix).wrapper()));
    DALI_UNARY_OP0(abs, op::abs,
            F<op::sign<R>>(MAT(matrix).wrapper()));
    DALI_UNARY_OP0(log, op::log,
            F<op::inv<R>>(MAT(matrix).wrapper()));
    DALI_UNARY_OP0(relu, op::relu,
            F<op::relu_backward<R>>(MAT(out).wrapper()));

    DALI_UNARY_OP1(max, lower_bound, op::max_scalar,
            F<op::max_scalar_mask<R>>(MAT(matrix).wrapper(), lower_bound));
    DALI_UNARY_OP1(steep_sigmoid, aggressiveness, op::steep_sigmoid,
            F<op::steep_sigmoid_backward<R>>(MAT(out).wrapper(), aggressiveness));


    template<typename R>
    Mat<R> Elementwise<R>::exp(Mat<R> matrix) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = F<op::exp<R>>(MAT(matrix).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += (MAT(out).wrapper() * GRAD(out).wrapper());
            });
        return out;
    }

    template<typename R>
    Mat<R> Elementwise<R>::sigmoid(Mat<R> matrix) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = F<op::sigmoid<R>>(MAT(matrix).wrapper());
        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += (
                    F<op::dsigmoid<R>>(MAT(out).wrapper()) * GRAD(out).wrapper()
                );
            });
        return out;
    }

    template<typename R>
    Mat<R> Elementwise<R>::sqrt(Mat<R> matrix) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = F<op::sqrt_f<R>>(MAT(matrix).wrapper());
        if (graph::backprop_enabled())
            graph::emplace_back([matrix, out]() mutable {
                SAFE_GRAD(matrix) += ((R)0.5 / MAT(out).wrapper()) * GRAD(out).wrapper();
            });
        return out;
    }

    template<typename R>
    Mat<R> Elementwise<R>::elt_inv(Mat<R> matrix) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = F<op::inv<R>>(MAT(matrix).wrapper());
        if (graph::backprop_enabled())
            graph::emplace_back([matrix, out]() mutable {
                SAFE_GRAD(matrix) -= F<op::square<R>>(MAT(out).wrapper()) * GRAD(out).wrapper();
            });
        return out;
    }


    template<typename R>
    Mat<R> Elementwise<R>::square(Mat<R> matrix) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = F<op::square<R>>(MAT(matrix).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += GRAD(out).wrapper() * MAT(matrix).wrapper() * (R) 2.0;
            });
        return out;
    }

    template<typename R>
    Mat<R> Elementwise<R>::pow(Mat<R> matrix, R other) {
        if (std::abs(other - (R)-1.0) < 1e-9) {
            return Elementwise<R>::elt_inv(matrix);
        } else if (std::abs(other - (R)0.0) < 1e-9) {
            return Other<R>::fill(matrix, 1.0);
        } else if (std::abs(other - (R)0.5) < 1e-9) {
            return Elementwise<R>::sqrt(matrix);
        } else if (std::abs(other - (R)1.0) < 1e-9) {
            return matrix;
        } else if (std::abs(other - (R)2.0) < 1e-9) {
            return Elementwise<R>::square(matrix);
        }

        auto out = Mat<R>::empty_like(matrix);

        MAT(out) = F<op::power<R>>(MAT(matrix).wrapper(), other);

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, out, other]() mutable {
                SAFE_GRAD(matrix) += other * F<op::power<R>>(MAT(matrix).wrapper(), other - (R)1.0) * GRAD(out).wrapper();
            });
        return out;
    }

    /////////////////////////////////////  SCALAR ARITHMETIC //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////



    template<typename R>
    Mat<R> Elementwise<R>::add(
            Mat<R> matrix1,
            R alpha) {
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = MAT(matrix1).wrapper() + alpha;
        if (graph::backprop_enabled() && !matrix1.constant)
            graph::emplace_back([matrix1, out]() mutable {
                GRAD(matrix1) += GRAD(out).wrapper();
            });
        return out;
    }

    template<typename R>
    Mat<R> Elementwise<R>::sub_broadcast_reversed(Mat<R> matrix, R other) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = (other - MAT(matrix).wrapper());
        if (graph::backprop_enabled())
            graph::emplace_back([matrix, out] () mutable {
                SAFE_GRAD(matrix) -= GRAD(out).wrapper();
            });
        return out;
    }

    template<typename R>
    Mat<R> Elementwise<R>::eltdivide(
            Mat<R> matrix,
            R alpha) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = MAT(matrix).wrapper() / alpha;
        if (graph::backprop_enabled())
            graph::emplace_back([matrix, alpha, out]() mutable {
                SAFE_GRAD(matrix) += ((R)1.0 / alpha) * GRAD(out).wrapper();
            });
        return out;
    }

    template<typename R>
    Mat<R> Elementwise<R>::eltmul(
            Mat<R> matrix,
            R alpha) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = MAT(matrix).wrapper() * alpha;
        if (graph::backprop_enabled())
            graph::emplace_back([matrix, alpha, out]() mutable {
                SAFE_GRAD(matrix) += alpha * GRAD(out).wrapper();
            });
        return out;
    }

    template class Elementwise<float>;
    template class Elementwise<double>;
    template class Elementwise<int>;
}
