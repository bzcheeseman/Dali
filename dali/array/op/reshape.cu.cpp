#include "reshape.h"

#include "dali/array/array.h"
#include "dali/array/function/operator.h"
#include "dali/array/functor.h"
#include "dali/array/lazy/reshape.h"
#include "dali/array/memory/device.h"
#include "dali/array/op/unary.h"
#include "dali/array/lazy/unary.h"

// TODO(jonathan, szymon): use stream wait events to ensure concatenation
//                         kicks off when all predecessors are done working
//                         source: http://cedric-augonnet.com/declaring-dependencies-with-cudastreamwaitevent/



struct ConcatenateFunction : public Function<ConcatenateFunction,
                                                        Array,
                                                        std::vector<Array>,
                                                        int> {
    static std::vector<int> deduce_output_bshape(const std::vector<Array>& arrays, const int& axis) {
        ASSERT2(arrays.size() > 0,
            "concatenate requires have at least one array argument (got 0)");
        auto base_bshape = arrays[0].bshape();
        ASSERT2(
            0 <= axis && axis < base_bshape.size(),
            utils::MS() << "concatenation axis must be greater than 0 and less than input array dimensionality"
                        << "(got axis = " << axis  << ", and ndim = " << base_bshape.size() << ")."
        );
        base_bshape[axis] = 0;
        for (int i = 0; i < arrays.size(); i++)  {
            ASSERT2(arrays[i].ndim() == base_bshape.size(),
                utils::MS() << "concatenate requires all argument arrays to have same dimensionality (got "
                            << "arrays[" << i << "].ndim() = " << arrays[i].ndim()
                            << " != arrays[0].ndim() = " << arrays[0].ndim() << ").");
            auto other_bshape = arrays[i].bshape();
            for (int shape_idx = 0; shape_idx < base_bshape.size(); shape_idx++) {
                // make sure all shapes agree on all other axes except concat
                // dimension.
                if (shape_idx != axis) {
                    // if we are in a non-concat axis
                    // we ensure that the shapes are equal.
                    // if they differ, we check if these are broadcasted axes
                    // or not
                    if (base_bshape[shape_idx] != other_bshape[shape_idx]) {
                        // if base is broadcasted, then size checking
                        // was too lax up to now. We set the new
                        // gold standard to be the current other_bshape[i]
                        if (base_bshape[shape_idx] == -1) {
                            // new gold standard (non-broadcasted)
                            base_bshape[shape_idx] = std::abs(other_bshape[shape_idx]);
                        } else {
                            ASSERT2(base_bshape[shape_idx] == other_bshape[shape_idx],
                                utils::MS() << "all the input array dimensions except for the concatenation axis must match exactly"
                                            << "(got arrays[" << i << "].bshape()[" << shape_idx << "] = "
                                            << other_bshape[shape_idx] << " != common_bshape[" << shape_idx << "] = "
                                            << base_bshape[shape_idx] << ").");
                        }
                    }
                } else {
                    // increment concatenation dimension
                    base_bshape[axis] += std::abs(other_bshape[shape_idx]);
                }
            }
        }
        return base_bshape;
    }

    static DType deduce_output_dtype(const std::vector<Array>& arrays, const int& axis) {
        DType common = arrays.size() > 0 ? arrays[0].dtype() : DTYPE_FLOAT;
        for (const auto& arr : arrays) {
            ASSERT2(arr.dtype() == common,
                    utils::MS() << "all array arguments to concatenate must have the same dtype (got "
                                << dtype_to_name(arr.dtype()) << " != " << dtype_to_name(common) << ").");
        }
        return common;
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void compute(const Array& out,
                        const memory::Device& device,
                        const std::vector<Array>& arrays,
                        const int& axis) {
        // construct result chunks:
        std::vector<Array> out_pieces;
        out_pieces.reserve(arrays.size());
        int so_far = 0;
        for (const auto& arr : arrays) {
            out_pieces.emplace_back(
                out.pluck_axis(
                    axis,
                    Slice(so_far, so_far + arr.shape()[axis])
                )
            );
            so_far += arr.shape()[axis];
        }

        typedef ArrayWrapper<devT,T> wrapper_t;

        for (int arg_idx = 0; arg_idx < out_pieces.size(); arg_idx++) {
            typed_eval<operator_t>(wrapper_t::wrap(out_pieces[arg_idx], device),
                                   wrapper_t::wrap(arrays[arg_idx], device));
        }
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT,T> out,
                    TypedArray<devT,T> chunk) {
        operator_assign<operator_t, 1>(out, mshadow::expr::F<functor::identity<T>>(chunk.d1()));
    }
};

namespace op {
    AssignableArray concatenate(const std::vector<Array>& arrays, int axis) {
        if (arrays.size() == 1) return op::identity(arrays[0], /*always_copy=*/false);
        if (axis < 0 && arrays.size() > 0) {
            // handle negative axes that wrap around and are
            // counted in reverse:
            axis = arrays[0].ndim() + axis;
        }
        return ConcatenateFunction::run(arrays, axis);
    }

    AssignableArray hstack(const std::vector<Array>& arrays) {
        return concatenate(arrays, -1);
    }

    AssignableArray vstack(const std::vector<Array>& arrays) {
        return concatenate(arrays, 0);
    }

    AssignableArray take(const Array& source, const Array& indices) {
        return lazy::take(source, indices);
    }

    AssignableArray take_from_rows(const Array& source, const Array& indices) {
        return lazy::take_from_rows(source, indices);
    }

    template<OPERATOR_T operator_t>
    void assign_to_rows(const ArraySubtensor& destination, const Array& assignable) {
        lazy::EvalWithOperator<operator_t>::assign(
            destination,
            lazy::identity(assignable)
        );
    }

    template void assign_to_rows<OPERATOR_T_EQL>(const ArraySubtensor&, const Array&);

} // namespace op
