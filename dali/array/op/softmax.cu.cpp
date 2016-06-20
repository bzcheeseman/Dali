#include "softmax.h"

#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include "dali/array/op.h"
#include "dali/array/mshadow_extension/kernelized_softmax.h"

template<OPERATOR_T operator_t, typename T, int devT>
struct SoftmaxFunctionHelper {
    template <
        OPERATOR_T var_operator_t = operator_t,
        typename var_T = T,
        typename std::enable_if<var_operator_t==OPERATOR_T_EQL>::type* = nullptr
    >
    static void run(const TypedArray<devT, T>& out, const TypedArray<devT, T>& a, const int& axis, const double& temperature) {
        if (axis == a.array.ndim() - 1) {
            // case 1: reduce over last dimension
            if (out.array.contiguous_memory()) {
                auto out_contig = out.contiguous_d2();
                if (a.array.contiguous_memory()) {
                    internal::softmax_rowwise(out_contig, a.contiguous_d2(), temperature);
                } else {
                    internal::softmax_rowwise(out_contig, a.d2(), temperature);
                }
            } else {
                auto out_noncontig = out.d2();
                if (a.array.contiguous_memory()) {
                    internal::softmax_rowwise(out_noncontig, a.contiguous_d2(), temperature);
                } else {
                    internal::softmax_rowwise(out_noncontig, a.d2(), temperature);
                }
            }
        } else if (axis == 0 && a.array.is_matrix() && out.array.contiguous_memory()) {
            // case 2: reduce over first dimension & input has 2 dimensions
            auto out_contig = out.contiguous_d2();
            if (out.array.contiguous_memory()) {
                internal::softmax_colwise(out_contig, a.contiguous_d2(), temperature);
            } else {
                internal::softmax_colwise(out_contig, a.d2(), temperature);
            }
        } else {
            // case 3: move reduction axis to the end, and recurse
            // (any other axis can be swapped with the last one to turn
            // the softmax into rowwise operation)
            auto new_a_array = a.array.swapaxes(axis, -1);
            // we also swap the dimensions of the output to undo the swapaxes when viewed normally
            // (TODO(jonathan,szymon): during prep it is also possible to make the output pre-swapped
            // so that computation can run on contiguous data, while result is a swapaxis view -> better performance)
            auto new_out_array = out.array.swapaxes(axis, -1);
            // create new typed-arrays to wrap these swapped arrays:
            TypedArray<devT,T> new_a_typedarray(new_a_array, a.device, new_a_array.shape());
            TypedArray<devT,T> new_out_typedarray(new_out_array, out.device, new_out_array.shape());

            // run op as usual with different axis:
            int newaxis = a.array.ndim() - 1;
            run(new_out_typedarray, new_a_typedarray, newaxis, temperature);
        }
    }

    template <
        OPERATOR_T var_operator_t = operator_t,
        typename var_T = T,
        typename std::enable_if<!(var_operator_t == OPERATOR_T_EQL)>::type* = nullptr
    >
    static void run(const TypedArray<devT, T>& out, const TypedArray<devT, T>& a, const int& axis, const double& temperature) {
        ASSERT2(var_operator_t == OPERATOR_T_EQL, "Softmax can only be computed with operator=");
        ASSERT2(false, "If asserts above are complete this message should never be displayed");
    }
};

struct SoftmaxFunction : public Function<SoftmaxFunction,
                                                Array,
                                                Array,
                                                int, double> {
    static std::vector<int> deduce_output_bshape(const Array& a, const int& axis, const double& temperature) {
        ASSERT2(0 <= axis && axis <= a.ndim(),
            utils::MS() << "Softmax axis must be contained between 0 and dimensionality of input (got axis="
                        << axis << ", while array.ndim()=" << a.ndim() << ").");
        return a.bshape();
    }

    template<OPERATOR_T operator_t, typename T, int devT>
    void typed_eval(TypedArray<devT, T> out, TypedArray<devT, T> a, const int& axis, const double& temperature) {
        SoftmaxFunctionHelper<operator_t, T, devT>::run(out, a, axis, temperature);
    }
};

namespace op {
    Assignable<Array> softmax(const Array& array, int axis, const double& temperature) {
        if (axis < 0) axis = array.ndim() + axis;
        return SoftmaxFunction::run(array, axis, temperature);
    }
} // namespace op
