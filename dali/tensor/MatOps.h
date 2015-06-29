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


// #ifdef NDEBUG
    #define DEBUG_ASSERT_POSITIVE(X) ;
    #define DEBUG_ASSERT_BOUNDS(X,a,b) ;
    #define DEBUG_ASSERT_NONZERO(X) ;
    #define DEBUG_ASSERT_MAT_NOT_NAN(X)
    #define DEBUG_ASSERT_GRAD_NOT_NAN(X)
    #define DEBUG_ASSERT_NOT_NAN(X) ;
// #else
//     #define DEBUG_ASSERT_POSITIVE(X) assert(((X).array() >= 0).all())
//     #define DEBUG_ASSERT_BOUNDS(X,a,b) assert(((X).array() >= (a)).all()  &&  ((X).array() <=(b)).all())
//     #define DEBUG_ASSERT_NONZERO(X) assert(((X).array().abs() >= 1e-10).all())
//     #define DEBUG_ASSERT_NOT_NAN(X) assert(!utils::contains_NaN(((X).array().abs().sum())))
//     #define DEBUG_ASSERT_MAT_NOT_NAN(X) if ( utils::contains_NaN((X).w()->w.norm()) ) { \
//         (X).print(); \
//         throw std::runtime_error(utils::explain_mat_bug((((X).name != nullptr) ? *(X).name : "?"), __FILE__,  __LINE__)); \
//     }
//     #define DEBUG_ASSERT_GRAD_NOT_NAN(X) if ( utils::contains_NaN((X).dw()->dw.norm()) ) { \
//         (X).print(); \
//         throw std::runtime_error(utils::explain_mat_bug((((X).name != nullptr) ? *(X).name : "?"), __FILE__,  __LINE__)); \
//     }
// #endif


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
