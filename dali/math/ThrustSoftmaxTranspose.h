#ifndef DALI_MATH_THRUST_SOFTMAX_TRANSPOSE_H
#define DALI_MATH_THRUST_SOFTMAX_TRANSPOSE_H
#ifdef DALI_USE_CUDA
// so we have access to to_thrust
#include "dali/math/TensorOps.h"

namespace TensorOps {
    namespace arg {
        // convert the indices of a 2d matrix in row major order to column major order
        template <typename T>
        struct linear_index_row_major_to_col_major: public thrust::unary_function<T, T> {
            T rows; // number of rows
            T cols;
            __host__ __device__
            linear_index_row_major_to_col_major(T _rows, T _cols) :
                    rows(_rows), cols(_cols) {}

            __host__ __device__ T operator() (T i) {
                return (i / rows) + ((i % rows) * cols);
            }
        };

        template<typename T>
        struct exp_difference: public thrust::binary_function<T, T, T> {
            __device__ T operator()(const T &a, const T &b) const {
                return expf(a - b);
            }
        };
    }

    template<typename R>
    void softmax(mshadow::Tensor<gpu,2,R> dest, mshadow::Tensor<gpu,2,R> src) {
        typedef int ind_t;

        using namespace thrust::placeholders;

        int total_size = dest.shape_.Size(),
            num_rows = src.shape_[0],
            num_cols = src.shape_[1];

        auto col_major_op  = arg::linear_index_row_major_to_col_major<ind_t>(
            num_rows,
            num_cols
        );
        auto col_major_index = thrust::make_transform_iterator(
            thrust::counting_iterator<ind_t>(0),
            col_major_op
        );

        auto src_ptr = to_thrust(src);

        auto reordered_values = thrust::make_permutation_iterator(
            src_ptr,
            col_major_index
        );

        // index over which same keys will be softmaxed together (column-wise)
        auto index_back_to_column = thrust::make_transform_iterator(
            thrust::counting_iterator<ind_t>(0),
            arg::linear_index_to_row_index<ind_t>(num_rows)
        );

        // store the first reduction in here
        thrust::device_vector<R> reduced_cols(num_cols);

        // gather the columwise maximums
        auto dest_ptr = to_thrust(dest);

        thrust::reduce_by_key(
            index_back_to_column,
            index_back_to_column + total_size,
            reordered_values,
            thrust::make_discard_iterator(),
            reduced_cols.begin(),
            thrust::equal_to<ind_t>(),
            thrust::maximum<R>()
        );

        auto repeated_max_back_to_column_index = thrust::make_transform_iterator(
            thrust::counting_iterator<ind_t>(0),
            arg::linear_index_to_col_index<ind_t>(num_cols)
        );

        auto repeated_max = thrust::make_permutation_iterator(
            reduced_cols.begin(),
            repeated_max_back_to_column_index
        );

        thrust::transform(
            src_ptr,
            src_ptr + total_size,
            repeated_max,
            dest_ptr,
            arg::exp_difference<R>()
        );

        auto reordered_exped_values = thrust::make_permutation_iterator(
            dest_ptr,
            col_major_index
        );

        thrust::reduce_by_key(
            index_back_to_column,
            index_back_to_column + total_size,
            reordered_exped_values,
            thrust::make_discard_iterator(),
            reduced_cols.begin(),
            thrust::equal_to<ind_t>(),
            thrust::plus<R>()
        );

        thrust::transform(
            dest_ptr,
            dest_ptr + total_size,
            repeated_max,
            dest_ptr,
            _1 / _2
        );

    }
}

#endif
#endif
