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
                    const TypedArray<devT, T>& b,
                    bool a_transposed,
                    bool b_transposed) {
        operator_assign_contiguous<operator_t, 2>(
            out,
            dali_gemm(
                a.contiguous_d2(),
                b.contiguous_d2(),
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
                    const TypedArray<devT, T>& b,
                    bool a_transposed,
                    bool b_transposed) {
        ASSERT2(!(std::is_same<var_T, int>::value),
            "Matrix multiplication is not supported for integers yet.");
        ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
                "Matrix multiplication's result cannot be inplace-multiplied or inplace-divided.");
        ASSERT2(false, "If asserts above are complete this message should never be displayed");
    }
};

struct LazyDot : public Function<LazyDot, Array, Array, Array, bool, bool> {
    static std::vector<int> deduce_output_bshape(
            const Array& a,
            const Array& b,
            bool a_transposed,
            bool b_transposed) {
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
            TypedArray<devT, T> b,
            bool a_transposed,
            bool b_transposed) {
        LazyDotRunner<operator_t,devT,T>::run(out, a, b, a_transposed, b_transposed);
    }
};



namespace op {
    namespace internal {
        bool is_contiguous_2D_transpose(const Array& a) {
            if (a.ndim() != 2) return false;
            if (a.strides().size() == 0) return false;

            const std::vector<int>& a_strides = a.strides();

            std::vector<int> a_shape_before_transpose = {a.shape()[1], a.shape()[0]};
            std::vector<int> strides_before_transpose = shape_to_trivial_strides(a_shape_before_transpose);

            return a_strides[0] == strides_before_transpose[1] && a_strides[1] == strides_before_transpose[0];
        }
    }

    // TODO (szymon): allow for scaling with Binary expression + template redundancy trick!
    AssignableArray dot(Array a, Array b) {
        // TODO(jonathan): implement the crazy transposes for cases > 2D.
        // from:jonathan, to:szymon, body: will do!

        bool a_transposed = internal::is_contiguous_2D_transpose(a);
        bool b_transposed = internal::is_contiguous_2D_transpose(b);
        ASSERT2(a_transposed || a.contiguous_memory(),
                "Dot is only supported for contiguous_memory (except for 2D transpose)");
        ASSERT2(b_transposed || b.contiguous_memory(),
                "Dot is only supported for contiguous_memory (except for 2D transpose)");
        return LazyDot::run(a, b, a_transposed, b_transposed);
    }
}
