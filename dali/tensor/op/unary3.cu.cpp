#include "dali/tensor/op/unary.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    #define DALI_DEFINE_RELU_ACTIVATION(NAME, UPPER_BOUND)\
        Tensor NAME(const Tensor& t) {\
            return relu(t, UPPER_BOUND);\
        }

    DALI_DEFINE_RELU_ACTIVATION(relu100, 100.0);
    DALI_DEFINE_RELU_ACTIVATION(relu20, 20.0);
    DALI_DEFINE_RELU_ACTIVATION(relu6, 6.0);
    DALI_DEFINE_RELU_ACTIVATION(relu5, 5.0);
}
