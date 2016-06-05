#ifndef DALI_ARRAY_FUNCTION_EVALUATION_DIM_H
#define DALI_ARRAY_FUNCTION_EVALUATION_DIM_H

class Array;

namespace lazy {
    static const int EVALUATION_DIM_DEFAULT =2;
    static const int EVALUATION_DIM_ANY = -1;
    static const int EVALUATION_DIM_ERROR = -2;

    template<typename ExprT>
    struct LazyEvaluationDim {
        static const int value = ExprT::evaluation_dim;
    };

    template<>
    struct LazyEvaluationDim<float> {
        static const int value = EVALUATION_DIM_ANY;
    };

    template<>
    struct LazyEvaluationDim<double> {
        static const int value = EVALUATION_DIM_ANY;
    };

    template<>
    struct LazyEvaluationDim<int> {
        static const int value = EVALUATION_DIM_ANY;
    };

    template<>
    struct LazyEvaluationDim<Array> {
        static const int value = EVALUATION_DIM_ANY;
    };
}

#endif
