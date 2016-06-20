#ifndef DALI_TENSOR_TENSOR_MACROS_H
#define DALI_TENSOR_TENSOR_MACROS_H

#define MAYBE_GRAD(X) if (!(X).constant) X.dw

#define DALI_DEFINE_UNARY_OP0(FUNCTION_NAME, FORWARD_OPNAME, BACKWARD_OPNAME) \
    Tensor FUNCTION_NAME(const Tensor& t) {\
        Tensor out(FORWARD_OPNAME(t.w));\
        if (graph::backprop_enabled() && !t.constant)\
            graph::emplace_back([t, out]() mutable {\
                MAYBE_GRAD(t) += (BACKWARD_OPNAME) * out.dw;\
            });\
        return out;\
    }

#define DALI_DEFINE_UNARY_OP1(FUNCTION_NAME, arg1, FORWARD_OPNAME, BACKWARD_OPNAME) \
    Tensor FUNCTION_NAME(const Tensor& t, const double& arg1) {\
        Tensor out(lazy::F<FORWARD_OPNAME>(t.w, arg1));\
        if (graph::backprop_enabled() && !t.constant)\
            graph::emplace_back([t, out, arg1]() mutable {\
                MAYBE_GRAD(t) += (BACKWARD_OPNAME) * out.dw;\
            });\
        return out;\
    }

#endif
