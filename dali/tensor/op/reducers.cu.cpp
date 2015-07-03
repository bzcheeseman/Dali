#include "dali/tensor/op/reducers.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

namespace matops {

    template<typename R>
    Mat<R> Reducers<R>::L2_norm(Mat<R> matrix) {
        auto out = Mat<R>(1, 1, weights<R>::empty());
        auto norm = MAT(matrix).L2_norm();
        out.w(0) = norm;

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out, norm]() mutable {
                GRAD(matrix) += (MAT(matrix).wrapper() * (out.dw(0) / norm) );
            });
        return out;
    }


    template<typename R>
    Mat<R> Reducers<R>::sum(Mat<R> matrix) {
        if (matrix.dims(0) == 1 && matrix.dims(1) == 1)
            return matrix;
        Mat<R> out(1,1, weights<R>::empty());
        out.w(0) = MAT(matrix).sum();

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += out.dw(0);
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::mean(Mat<R> matrix) {
        Mat<R> out (1,1, weights<R>::empty());
        auto ne = matrix.number_of_elements();
        out.w(0) = MAT(matrix).sum() / ne;
        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out, ne]() mutable {
                GRAD(matrix) += out.dw(0) / ne;
            });

        return out;
    }
    template class Reducers<float>;
    template class Reducers<double>;

}
