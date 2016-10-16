#ifndef DALI_ARRAY_OP_H
#define DALI_ARRAY_OP_H

#include "dali/config.h"

#include "dali/array/op/other.h"
#include "dali/array/op/cast.h"
#include "dali/array/op/dot.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/reshape.h"
#include "dali/array/op/initializer.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/softmax.h"
#include "dali/array/op/spatial.h"
#include "dali/array/op/unary_scalar.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op2/operation.h"
#include "dali/array/op2/reducers.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/circular_convolution.h"
#include "dali/array/op2/one_hot.h"
#include "dali/array/op2/im2col.h"
#include "dali/utils/print_utils.h"

#if EXISTS_AND_TRUE(DALI_USE_LAZY)
    #include "dali/array/lazy/binary.h"
    #include "dali/array/lazy/reducers.h"
    #include "dali/array/lazy/unary.h"
    #include "dali/array/lazy/reshape.h"
    #include "dali/array/lazy/im2col.h"
    #include "dali/array/lazy/circular_convolution.h"
    #include "dali/array/function/lazy_evaluator.h"
    #include "dali/array/op_overload/lazy.h"

    namespace lazy {
        static bool ops_loaded = true;
    }
#else
    #include "dali/array/op_overload/nonlazy.h"

    namespace lazy {
        static bool ops_loaded = false;
    }
#endif

#endif
