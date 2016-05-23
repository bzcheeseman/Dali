////////////////////////////////////////////////////////////////////////////////
//           Logic that converts Tensor dot to Matrix Dot                     //
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"

namespace op {
    template<typename Container>
    matmul_result_t<Container> tensordot_as_dot(
            const Container& a,
            const Container& b,
            const int& axis,
            bool batched) {
        // This code follows the logic from theano's tensordot as dot
        // [source https://github.com/Theano/Theano/blob/master/theano/tensor/basic.py#L5628]
        //
        // Theano code was also originally based elsewhere on
        // Tijmen Tieleman's gnumpy:
        // [source http://www.cs.toronto.edu/~tijmen/gnumpy.html]

        // if 'axes' is a single number of axes to multiply and sum over
        // (trailing axes of a, leading axes of b), we can just reshape
        // and use dot.
        // validate that the axis used for summing
        // is not out of bounds for the arguments a and b
        ASSERT2(
            axis >= 0,
            utils::MS() << "axis must be a non-negative integer (got " << axis << ")."
        );
        for (int i = 0; i < 2; i++) {
            auto& operand = i == 0 ? a : b;
            char operand_name = i == 0 ? 'a' : 'b';
            ASSERT2(
                axis <= operand.ndim(),
                utils::MS() << "axis can not be larger than the dimension of "
                            << operand_name
                            << " (" << operand_name << ".ndim()=" << operand.ndim()
                            << ", axis=" << axis <<")."
            );
            ASSERT2(!(axis == operand.ndim() && batched),
                utils::MS() << "axis to sum over must not include the batch axis "
                            << "of " << operand_name
                            << " (" << operand_name << ".ndim()=" << operand.ndim()
                            << ", axis=" << axis <<")."
            );
        }
        int batch_axes = batched ? 1 : 0;

        std::vector<int> a_shape = {1, 1};
        std::vector<int> b_shape = {1, 1};

        const auto& a_old_shape = a.shape();
        const auto& b_old_shape = b.shape();

        // compute total size of summed axes
        for (int i = 0; i < axis; i++) {
            a_shape[1] *= a_old_shape[a_old_shape.size() - (i + 1)];
            b_shape[0] *= b_old_shape[batch_axes + i];
        }
        // compute total size of other axes
        for (int i = 0; i < (a.ndim() - axis - batch_axes); i++) {
            a_shape[0] *= a_old_shape[batch_axes + i];
        }
        for (int i = 0; i < (b.ndim() - axis - batch_axes); i++) {
            b_shape[1] *= b_old_shape[b_old_shape.size() -(i + 1)];
        }

        if (batched) {
            a_shape.insert(a_shape.begin(), a_old_shape[0]);
            b_shape.insert(b_shape.begin(), b_old_shape[0]);
        }
        auto a_reshaped = a.reshape(a_shape);
        auto b_reshaped = b.reshape(b_shape);


        std::vector<int> output_shape;

        output_shape.insert(
            output_shape.begin(),
            a_old_shape.begin(),
            a_old_shape.begin() + a_old_shape.size() - axis
        );

        output_shape.insert(
            output_shape.end(),
            b_old_shape.begin() + batch_axes + axis,
            b_old_shape.end()
        );

        return matrix_multiply_with_reshape(
                a_reshaped,
                b_reshaped,
                output_shape,
                {a_shape[0], b_shape[1]});
    }

    template<typename Container>
    matmul_result_t<Container> tensordot_as_dot(
            const Container& a,
            const Container& b,
            const std::vector<int>& a_reduce_axes,
            const std::vector<int>& b_reduce_axes,
            bool batched) {

        ASSERT2(a_reduce_axes.size() == b_reduce_axes.size(),
            utils::MS() << "must have as many reduction axes for a than b "
                        << "(got a_reduce_axes=" << a_reduce_axes << " and "
                        << "b_reduce_axes=" << b_reduce_axes << ")."
        );

        check_tensordot_reduce_axes(a.shape(), 'a', a_reduce_axes, batched);
        check_tensordot_reduce_axes(b.shape(), 'b', b_reduce_axes, batched);

        auto a_new_axes = tensordot_nonreduced_axes(
            a.ndim(), a_reduce_axes, batched
        );
        auto b_new_axes = tensordot_nonreduced_axes(
            b.ndim(), b_reduce_axes, batched
        );

        // for A: add reduction axis at the end of shape
        a_new_axes.insert(a_new_axes.end(), a_reduce_axes.begin(), a_reduce_axes.end());
        // for B: add reduction axis at the beginning of shape
        b_new_axes.insert(b_new_axes.begin(), b_reduce_axes.begin(), b_reduce_axes.end());

        if (batched) {
            a_new_axes.insert(a_new_axes.begin(), 0);
            b_new_axes.insert(b_new_axes.begin(), 0);
        }

        // now call dimshuffle
        auto a_shuffled = a.dimshuffle(a_new_axes);
        auto b_shuffled = b.dimshuffle(b_new_axes);

        return tensordot_as_dot(
            a_shuffled,
            b_shuffled,
            (int)a_reduce_axes.size(),
            batched
        );
    }


};  // namespace op
