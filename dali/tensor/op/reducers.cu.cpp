#include "reducers.h"
#include "dali/tensor/tape.h"
#include "dali/array/lazy_op.h"
#include "dali/tensor/tensor_macros.h"
#include "dali/utils/print_utils.h"

namespace tensor_ops {
    Tensor sum(const Tensor& tensor) {
        if (tensor.number_of_elements() == 1) {
            auto out = tensor;
            out.w = tensor.w.reshape({});
            out.dw = tensor.dw.reshape({});
            return out;
        } else {
            // TODO(jonathan, szymon) also makes sure that device
            // of input tensor is also used here

            Tensor out(tensor.w.sum());

            if (graph::backprop_enabled() && !tensor.constant)
                graph::emplace_back([tensor, out]() mutable {
                    tensor.dw <<= out.dw.broadcast_scalar_to_ndim(tensor.ndim());
                });
            return out;
        }
    }

    Tensor mean(const Tensor& tensor) {
        if (tensor.number_of_elements() == 1) {
            auto out = tensor;
            out.w = tensor.w.reshape({});
            out.dw = tensor.dw.reshape({});
            return out;
        } else {
            Tensor out(tensor.w.mean());
            if (graph::backprop_enabled() && !tensor.constant)
                graph::emplace_back([tensor, out]() mutable {
                    tensor.dw <<= (
                        out.dw.broadcast_scalar_to_ndim(tensor.ndim()) /
                        tensor.number_of_elements()
                    );
                });
            return out;
        }
    }

    Tensor L2_norm(const Tensor& tensor) {
        Tensor out(tensor.w.L2_norm());
        if (graph::backprop_enabled() && !tensor.constant)
            graph::emplace_back([tensor, out]() mutable {
                MAYBE_GRAD(tensor) <<= (
                    tensor.w * (
                        out.dw.broadcast_scalar_to_ndim(tensor.ndim()) /
                        out.w.broadcast_scalar_to_ndim(tensor.ndim())
                    )
                );
            });
        return out;
    }

    Tensor L2_norm(const Tensor& tensor, const int& axis) {
        Tensor out(tensor.w.L2_norm(axis));
        if (graph::backprop_enabled() && !tensor.constant)
            graph::emplace_back([tensor, out, axis]() mutable {
                MAYBE_GRAD(tensor) <<= (
                    tensor.w * (
                        tensor.dw.insert_broadcast_axis(axis) /
                        out.w.insert_broadcast_axis(axis)
                    )
                );
            });
        return out;
    }

    Tensor sum(const Tensor& tensor, const int& axis) {
        Tensor out(op::sum(tensor.w, axis));
        if (graph::backprop_enabled() && !tensor.constant)
            graph::emplace_back([tensor, out, axis]() mutable {
                // make sure output has same shape as input
                // with the reduced dimension returned as
                // broadcasted
                auto reshaped_gradient = out.dw.insert_broadcast_axis(axis);
                tensor.dw <<= reshaped_gradient;
            });
        return out;
    }

    Tensor mean(const Tensor& tensor, const int& axis) {
        Tensor out(op::mean(tensor.w, axis));
        if (graph::backprop_enabled() && !tensor.constant)
            graph::emplace_back([tensor, out, axis]() mutable {
                int axis_size = tensor.shape()[axis];
                auto reshaped_gradient = out.dw.insert_broadcast_axis(axis);
                tensor.dw <<= reshaped_gradient / axis_size;
            });
        return out;
    }


    #define DALI_TENSOR_SUBSAMPLE_ALL_REDUCTION(FUNCTION_NAME)\
        Tensor FUNCTION_NAME(const Tensor& tensor) {\
            if (tensor.number_of_elements() == 1) {\
                auto out = tensor;\
                out.w = tensor.w.reshape({});\
                out.dw = tensor.dw.reshape({});\
                return out;\
            } else {\
                Tensor out(tensor.w.FUNCTION_NAME());\
                if (graph::backprop_enabled() && !tensor.constant)\
                    graph::emplace_back([tensor, out]() mutable {\
                        tensor.dw <<= lazy::subsample_partial_grad(\
                            out.w.broadcast_scalar_to_ndim(tensor.ndim()),\
                            tensor.w\
                        ) * out.dw.broadcast_scalar_to_ndim(tensor.ndim());\
                    });\
                return out;\
            }\
        }\

    DALI_TENSOR_SUBSAMPLE_ALL_REDUCTION(min);
    DALI_TENSOR_SUBSAMPLE_ALL_REDUCTION(max);

    #define DALI_TENSOR_SUBSAMPLE_AXIS_REDUCTION(FUNCTION_NAME, OPNAME)\
        Tensor FUNCTION_NAME(const Tensor& tensor, const int& axis) {\
            Tensor out(OPNAME(tensor.w, axis));\
            if (graph::backprop_enabled() && !tensor.constant)\
                graph::emplace_back([tensor, out, axis]() mutable {\
                    tensor.dw <<= lazy::subsample_partial_grad(\
                            out.w.insert_broadcast_axis(axis),\
                            tensor.w\
                        ) * out.dw.insert_broadcast_axis(axis);\
                });\
            return out;\
        }\

    DALI_TENSOR_SUBSAMPLE_AXIS_REDUCTION(min, op::min);
    DALI_TENSOR_SUBSAMPLE_AXIS_REDUCTION(max, op::max);


    #define DALI_TENSOR_SUBSAMPLE_ALL_REDUCTION(FUNCTION_NAME)\
        Tensor FUNCTION_NAME(const Tensor& tensor) {\
            if (tensor.number_of_elements() == 1) {\
                auto out = tensor;\
                out.w = tensor.w.reshape({});\
                out.dw = tensor.dw.reshape({});\
                return out;\
            } else {\
                Tensor out(tensor.w.FUNCTION_NAME());\
                if (graph::backprop_enabled() && !tensor.constant)\
                    graph::emplace_back([tensor, out]() mutable {\
                        tensor.dw <<= lazy::subsample_partial_grad(\
                            out.w.broadcast_scalar_to_ndim(tensor.ndim()),\
                            tensor.w\
                        ) * out.dw.broadcast_scalar_to_ndim(tensor.ndim());\
                    });\
                return out;\
            }\
        }\

    #define DALI_TENSOR_GETINDICES_ALL_REDUCTION(FUNCTION_NAME)\
        Tensor FUNCTION_NAME(const Tensor& tensor) {\
            return Tensor(op::FUNCTION_NAME(tensor.w.ravel(), 0));\
        }\

    DALI_TENSOR_GETINDICES_ALL_REDUCTION(argmin);
    DALI_TENSOR_GETINDICES_ALL_REDUCTION(argmax);

    #define DALI_TENSOR_GETINDICES_AXIS_REDUCTION(FUNCTION_NAME)\
        Tensor FUNCTION_NAME(const Tensor& tensor, const int& axis) {\
            return Tensor(op::FUNCTION_NAME(tensor.w, axis));\
        }\

    DALI_TENSOR_GETINDICES_AXIS_REDUCTION(argmin);
    DALI_TENSOR_GETINDICES_AXIS_REDUCTION(argmax);

}
