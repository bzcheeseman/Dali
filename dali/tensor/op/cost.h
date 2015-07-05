#ifndef DALI_TENSOR_OP_COST_H
#define DALI_TENSOR_OP_COST_H

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {
    template<typename R>
    struct Cost {
        static Mat<R> binary_cross_entropy(Mat<R>, R);
        static Mat<R> sigmoid_binary_cross_entropy(Mat<R>, R);
        static Mat<R> cross_entropy(Mat<R>, uint answer_idx);
        static Mat<R> cross_entropy(Mat<R>, Mat<R> targets);
        static Mat<R> softmax_cross_entropy(Mat<R> matrix, uint answer_idx);
        static Mat<R> softmax_cross_entropy(Mat<R> matrix, Indexing::Index targets);
        static Mat<R> softmax_cross_entropy(Mat<R> matrix, Mat<int> targets);
        static Mat<R> margin_loss(Mat<R> matrix, uint answer_idx, R margin=0.1);

        static Mat<R> softmax(Mat<R>, R temperature=1.0);
        static Mat<R> softmax_transpose(Mat<R>, R temperature=1.0);
        static Mat<R> softmax_no_grad(Mat<R>, R temperature = 1.0);
        static Mat<R> softmax_no_grad_transpose(Mat<R>, R temperature=1.0);

        static std::vector<Mat<R>> softmax(std::vector<Mat<R>>&, R temperature=1.0);
        static std::vector<Mat<R>> softmax_no_grad(const std::vector<Mat<R>>&, R temperature = 1.0);

    };
}

#endif
