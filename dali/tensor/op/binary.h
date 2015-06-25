#ifndef DALI_TENSOR_OP_BINARY_H
#define DALI_TENSOR_OP_BINARY_H

#include "dali/tensor/Tape.h"
#include "dali/utils.h"
#include "dali/tensor/op/elementwise.h"


template<typename R> class Mat;
namespace matops {
    template<typename R> class Elementwise;
}
namespace matops {
    template<typename R>
    struct Binary : matops::Elementwise<R> {
        static Mat<R> add(Mat<R>, Mat<R>);
        static Mat<R> add_broadcast(Mat<R>, Mat<R>);
        static Mat<R> sub(Mat<R>, Mat<R>);
        static Mat<R> sub_broadcast(Mat<R>, Mat<R>);
        static Mat<R> sub_broadcast_reversed(Mat<R>, Mat<R>);
        static Mat<R> eltmul_broadcast(Mat<R>, Mat<R>);
        static Mat<R> eltdivide_broadcast(Mat<R>, Mat<R>);
        static Mat<R> eltdivide_broadcast_reversed(Mat<R>, Mat<R>);
        static Mat<R> eltmul(Mat<R>, Mat<R>);
        static Mat<R> eltdivide(Mat<R>, Mat<R>);
        static Mat<R> eltmul_broadcast_rowwise(Mat<R>, Mat<R>);
        static Mat<R> eltmul_rowwise(Mat<R>, Mat<R>);
        static Mat<R> mul(Mat<R>, Mat<R>);
        static Mat<R> pow(Mat<R>, Mat<R>);

        static Mat<R> add(std::initializer_list<Mat<R>>);
        static Mat<R> add(std::vector<Mat<R>>&);
        static std::vector<Mat<R>> eltmul(const std::vector<Mat<R>>&, const std::vector<Mat<R>>&);
        static std::vector<Mat<R>> eltmul_broadcast_rowwise(const std::vector<Mat<R>>&, const std::vector<Mat<R>>&);
        static std::vector<Mat<R>> eltmul_rowwise(const std::vector<Mat<R>>&, const std::vector<Mat<R>>&);
    };
}

#endif
