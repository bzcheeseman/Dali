#include "reducers.h"
#include "dali/tensor/tape.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/top_k.h"
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
            Tensor out(tensor.w.sum());
            if (graph::backprop_enabled() && !tensor.constant) {
                auto out_dw = out.dw;
                auto tensor_dw = tensor.dw;
                graph::emplace_back([tensor_dw, out_dw]() mutable {
                    tensor_dw <<= out_dw.broadcast_scalar_to_ndim(tensor_dw.ndim());
                });
            }
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
            if (graph::backprop_enabled() && !tensor.constant) {
                auto out_dw = out.dw;
                auto tensor_dw = tensor.dw;
                graph::emplace_back([tensor_dw, out_dw]() mutable {
                    tensor_dw <<= (
                        out_dw.broadcast_scalar_to_ndim(tensor_dw.ndim()) /
                        tensor_dw.number_of_elements()
                    );
                });
            }
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

    Tensor L2_norm(const Tensor& tensor, const std::vector<int>& axes, bool keepdims) {
        Tensor out(tensor.w.L2_norm(axes, keepdims));
        if (graph::backprop_enabled() && !tensor.constant)
            graph::emplace_back([tensor, out, axes, keepdims]() mutable {
                // MAYBE_GRAD(tensor) <<= (
                //     tensor.w * (
                //         out.dw.insert_broadcast_axis(axis) /
                //         out.w.insert_broadcast_axis(axis)
                //     )
                // );
            });
        return out;
    }

    Tensor sum(const Tensor& tensor, const std::vector<int>& axes, bool keepdims) {
        Tensor out(op::sum(tensor.w, axes, keepdims));
        if (graph::backprop_enabled() && !tensor.constant) {
            auto tensor_dw = tensor.dw;
            auto out_dw = out.dw;
            graph::emplace_back([tensor_dw, out_dw, axes, keepdims]() mutable {
                // make sure output has same shape as input
                // with the reduced dimension returned as
                // broadcasted
                // auto reshaped_gradient = out_dw.insert_broadcast_axis(axis);
                // tensor_dw <<= reshaped_gradient;
            });
        }
        return out;
    }

    Tensor mean(const Tensor& tensor, const std::vector<int>& axes, bool keepdims) {
        Tensor out(op::mean(tensor.w, axes, keepdims));
        if (graph::backprop_enabled() && !tensor.constant) {
            auto tensor_dw = tensor.dw;
            auto out_dw = out.dw;
            graph::emplace_back([tensor_dw, out_dw, axes, keepdims]() mutable {
                // if (axis < 0) axis = axis + tensor_dw.ndim();
                // int axis_size = tensor_dw.shape()[axis];
                // auto reshaped_gradient = out_dw.insert_broadcast_axis(axis);
                // tensor_dw <<= reshaped_gradient / axis_size;
            });
        }
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
                        tensor.dw <<= op::equals(\
                            out.w.broadcast_scalar_to_ndim(tensor.ndim()),\
                            tensor.w\
                        ) * out.dw.broadcast_scalar_to_ndim(tensor.ndim());\
                    });\
                return out;\
            }\
        }\

    DALI_TENSOR_SUBSAMPLE_ALL_REDUCTION(min);
    DALI_TENSOR_SUBSAMPLE_ALL_REDUCTION(max);

    // tensor.dw <<= op::equals(\
    //         out.w.insert_broadcast_axis(axis),\
    //         tensor.w\
    //     ) * out.dw.insert_broadcast_axis(axis);\

    #define DALI_TENSOR_SUBSAMPLE_AXIS_REDUCTION(FUNCTION_NAME, OPNAME)\
        Tensor FUNCTION_NAME(const Tensor& tensor, const std::vector<int>& axes, bool keepdims) {\
            Tensor out(OPNAME(tensor.w, axes, keepdims));\
            if (graph::backprop_enabled() && !tensor.constant)\
                graph::emplace_back([tensor, out, axes]() mutable {\
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
                        tensor.dw <<= op::equals(\
                            out.w.broadcast_scalar_to_ndim(tensor.ndim()),\
                            tensor.w\
                        ) * out.dw.broadcast_scalar_to_ndim(tensor.ndim());\
                    });\
                return out;\
            }\
        }\

    #define DALI_TENSOR_GETINDICES_ALL_REDUCTION(FUNCTION_NAME)\
        Tensor FUNCTION_NAME(const Tensor& tensor) {\
            return Tensor(op::FUNCTION_NAME(tensor.w));\
        }\

    DALI_TENSOR_GETINDICES_ALL_REDUCTION(argmin);
    DALI_TENSOR_GETINDICES_ALL_REDUCTION(argmax);

    Tensor top_k(const Tensor& tensor, int k, bool sorted) {
        return Tensor(op::top_k(tensor.w, k, sorted));
    }
    Tensor bottom_k(const Tensor& tensor, int k, bool sorted) {
        return Tensor(op::bottom_k(tensor.w, k, sorted));
    }

    Tensor argsort(const Tensor& tensor) {
        return Tensor(op::argsort(tensor.w, 0));
    }

    #define DALI_TENSOR_GETINDICES_AXIS_REDUCTION(FUNCTION_NAME)\
        Tensor FUNCTION_NAME(const Tensor& tensor, int axis) {\
            return Tensor(op::FUNCTION_NAME(tensor.w, axis));\
        }\

    DALI_TENSOR_GETINDICES_AXIS_REDUCTION(argmin);
    DALI_TENSOR_GETINDICES_AXIS_REDUCTION(argmax);
    DALI_TENSOR_GETINDICES_AXIS_REDUCTION(argsort);
}
