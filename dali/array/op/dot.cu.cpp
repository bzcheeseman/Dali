#include "dot.h"

#include <vector>

#include "dali/array/function/function.h"


template<OPERATOR_T operator_t, int devT, typename T>
struct LazyDotRunner {
    template <
        OPERATOR_T var_operator_t = operator_t,
        typename var_T = T,
        typename std::enable_if<
            !(
                (var_operator_t == OPERATOR_T_MUL) ||
                (var_operator_t == OPERATOR_T_DIV) ||
                std::is_same<var_T,int>::value
            )
        >::type* = nullptr
    >
    static void run(TypedArray<devT, T> out, TypedArray<devT, T> a, TypedArray<devT, T> b) {
        operator_assign_contiguous<operator_t, 2>(
            out,
            mshadow::expr::dot(a.contiguous_d2(), b.contiguous_d2())
        );
    }

    template <
        OPERATOR_T var_operator_t = operator_t,
        typename var_T = T,
        typename std::enable_if<
            (var_operator_t == OPERATOR_T_MUL) ||
            (var_operator_t == OPERATOR_T_DIV) ||
            std::is_same<var_T, int>::value
        >::type* = nullptr
    >
    static void run(TypedArray<devT, T> out, TypedArray<devT, T> a, TypedArray<devT, T> b) {
        ASSERT2(!(std::is_same<var_T, int>::value),
            "Matrix multiplication is not supported for integers yet.");
        ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
                "Matrix multiplication's result cannot be inplace-multiplied or inplace-divided.");
        ASSERT2(false, "If asserts above are complete this message should never be displayed");
    }
};


struct LazyDot : public Function<LazyDot, Array, Array, Array> {
    static std::vector<int> deduce_output_bshape(const Array& a, const Array& b) {
        ASSERT2(a.ndim() == 2 && b.ndim() == 2,
                utils::MS() << "Dot product is only supported for matrices, got " << a.ndim() << "D and " << b.ndim() << "D tensors.");
        ASSERT2(a.shape()[1] == b.shape()[0],
            utils::MS() << "Dot product requires matching inner dimension (got shapes: " << a.shape() << ", " << b.shape() << ", which are shaped like a bowtie)");
        return {a.shape()[0], b.shape()[1]};
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT, T> out, TypedArray<devT, T> a, TypedArray<devT, T> b) {
        LazyDotRunner<operator_t,devT,T>::run(out, a, b);
    }
};

namespace op {

    AssignableArray dot(Array a, Array b) {
        // TODO(use this experassion to enable transposes)
        // DotExp<TA, TB, true, false, DType>(lhs.exp, rhs.self(), 1.0f);
        // TODO (jaro): assert not strided, except if transposed and
        // possibly strided on last dim.
        // bool a_t = a.is_simple_transpose();
        // bool b_t = b.is_simple_transpose();
        return LazyDot::run(a, b);
    }
}
