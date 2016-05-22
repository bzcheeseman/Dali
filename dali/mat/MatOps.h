#ifndef CORE_MAT_OPS_H
#define CORE_MAT_OPS_H

#include <initializer_list>
#include <vector>
#include <memory>

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/tensor/Mat.h"
#include "dali/tensor/op/binary.h"
#include "dali/tensor/op/elementwise.h"
#include "dali/tensor/op/composite.h"
#include "dali/tensor/op/convolution.h"
#include "dali/tensor/op/cost.h"
#include "dali/tensor/op/dropout.h"
#include "dali/tensor/op/solver_updates.h"
#include "dali/tensor/op/other.h"
#include "dali/tensor/op/reducers.h"
#include "dali/tensor/op/reshaping.h"
#include "dali/utils.h"

namespace matops {
    template<typename R> class Binary;
    template<typename R> class Elementwise;
    template<typename R> class Reducers;
    template<typename R> class Cost;
    template<typename R> class Reshaping;
    template<typename R> class Dropout;
    template<typename R> class Composite;
    template<typename R> class Other;
    template<typename R> class Convolution;
    template<typename R> class SolverUpdates;
}

template<typename R>
struct MatOps : matops::Binary<R>,
                matops::Elementwise<R>,
                matops::Reducers<R>,
                matops::Cost<R>,
                matops::Reshaping<R>,
                matops::Dropout<R>,
                matops::SolverUpdates<R>,
                matops::Composite<R>,
                matops::Other<R>,
                matops::Convolution<R> {
        static Mat<R> add(Mat<R> x, R y) { return matops::Elementwise<R>::add(x,y); }
        static Mat<R> sub_broadcast_reversed(Mat<R> x, R y){
            return matops::Elementwise<R>::sub_broadcast_reversed(x,y);
        }
        static Mat<R> eltmul(Mat<R> x, R y) { return matops::Elementwise<R>::eltmul(x,y); }
        static Mat<R> eltdivide(Mat<R> x, R y) { return matops::Elementwise<R>::eltdivide(x,y) ; }
        static Mat<R> pow(Mat<R> x, R y) { return matops::Elementwise<R>::pow(x,y); }

        static Mat<R> add(Mat<R> x, Mat<R> y) { return matops::Binary<R>::add(x,y); }
        static Mat<R> sub_broadcast_reversed(Mat<R> x, Mat<R> y){
            return matops::Binary<R>::sub_broadcast_reversed(x,y);
        }
        static Mat<R> eltmul(Mat<R> x, Mat<R> y) { return matops::Binary<R>::eltmul(x,y); }
        static Mat<R> eltdivide(Mat<R> x, Mat<R> y) { return matops::Binary<R>::eltdivide(x,y) ; }
        static Mat<R> pow(Mat<R> x, Mat<R> y) { return matops::Binary<R>::pow(x,y); }

        static Mat<R> add(std::initializer_list<Mat<R>> v) { return matops::Binary<R>::add(v); }
        static Mat<R> add(std::vector<Mat<R>>& v) { return matops::Binary<R>::add(v); }
};



#endif
