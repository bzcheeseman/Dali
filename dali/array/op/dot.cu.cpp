#include "dot.h"

#include <vector>

#include "dali/array/function/function.h"
#include "dali/array/shape.h"

#include "dali/array/function/args/dali_gemm_engine_exp.h"


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
    static void run(TypedArray<devT, T>& out,
                    const TypedArray<devT, T>& a,
                    const TypedArray<devT, T>& b) {
        typedef decltype(a.contiguous_d2()) mshadow_tensor_t;
        bool             a_transposed, b_transposed;
        mshadow_tensor_t a_tensor,     b_tensor;
        std::tie(a_transposed, a_tensor) = a.blas_friendly_tensor();
        std::tie(b_transposed, b_tensor) = b.blas_friendly_tensor();

        operator_assign_contiguous<operator_t, 2>(
            out,
            dali_gemm(
                a_tensor,
                b_tensor,
                a_transposed,
                b_transposed,
                (T)1.0f
            )
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
    static void run(TypedArray<devT, T>& out,
                    const TypedArray<devT, T>& a,
                    const TypedArray<devT, T>& b) {
        ASSERT2(!(std::is_same<var_T, int>::value),
            "Matrix multiplication is not supported for integers yet.");
        ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
                "Matrix multiplication's result cannot be inplace-multiplied or inplace-divided.");
        ASSERT2(false, "If asserts above are complete this message should never be displayed");
    }
};

struct LazyDot : public Function<LazyDot, Array, Array, Array> {
    static std::vector<int> deduce_output_bshape(
            const Array& a,
            const Array& b) {
        ASSERT2(a.ndim() == 2 && b.ndim() == 2,
                utils::MS() << "Dot product is only supported for matrices, got " << a.ndim() << "D and " << b.ndim() << "D tensors.");
        ASSERT2(a.shape()[1] == b.shape()[0],
            utils::MS() << "Dot product requires matching inner dimension (got shapes: " << a.shape() << ", " << b.shape() << ", which are shaped like a bowtie)");
        return {a.shape()[0], b.shape()[1]};
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(
            TypedArray<devT, T> out,
            TypedArray<devT, T> a,
            TypedArray<devT, T> b) {
        LazyDotRunner<operator_t,devT,T>::run(out, a, b);
    }
};


namespace op {
    // TODO (szymon): allow for scaling with Binary expression + template redundancy trick!
    AssignableArray dot(Array a, Array b) {
        // TODO(jonathan): implement the crazy transposes for cases > 2D.
        // from:jonathan, to:szymon, body: will do!
        // TODO(dali_developer_community): make sure that copies happend at the right places.


        return LazyDot::run(a, b);
    }
}
