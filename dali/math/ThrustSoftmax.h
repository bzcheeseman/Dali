#ifndef DALI_MATH_THRUST_SOFTMAX_TRANSPOSE_H
#define DALI_MATH_THRUST_SOFTMAX_TRANSPOSE_H
#include "dali/math/TensorOps.h"
#include "dali/math/memory_bank/MemoryBank.h"
#include "dali/math/ThrustReduceByKey.h"
/**
Thrust Softmax
--------------

GPU and CPU implementations of Softmax. Both operations add
support for temperature (controls the roll off of the
exponentiation during Softmax).

The column-wise softmax is done using Thrust, while the
row-wise softmax (`softmax_transpose`) is achieved by
modifying one line from MShadow's version.
**/

namespace TensorOps {
    #ifdef DALI_USE_CUDA
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
            T temperature;
            __host__ __device__
            exp_difference(T _temperature) : temperature(_temperature) {}

            __device__ T operator()(const T &a, const T &b) const {
                return expf((a - b) / temperature);
            }
        };
    }

    template<typename R>
    void softmax(mshadow::Tensor<gpu,2,R> dest, mshadow::Tensor<gpu,2,R> src, R temperature = 1.0) {
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
        // Ask the memory bank if this type of memory was allocated before
        // and borrow it
        auto reduced_cols = temporary_array<R>(num_cols, num_cols);
        // wrap it in a Thrust pointer for convenience.
        auto reduced_cols_begin = reduced_cols.begin();

        // gather the columwise maximums
        auto dest_ptr = to_thrust(dest);

        auto keys_output = temporary_array<ind_t>(total_size, total_size);

        thrust::dali_reduce_by_key(
            index_back_to_column,
            index_back_to_column + total_size,
            reordered_values,
            keys_output.begin(),
            reduced_cols_begin,
            thrust::equal_to<ind_t>(),
            thrust::maximum<R>()
        );

        auto repeated_max_back_to_column_index = thrust::make_transform_iterator(
            thrust::counting_iterator<ind_t>(0),
            arg::linear_index_to_col_index<ind_t>(num_cols)
        );

        auto repeated_max = thrust::make_permutation_iterator(
            reduced_cols_begin,
            repeated_max_back_to_column_index
        );

        thrust::transform(
            src_ptr,
            src_ptr + total_size,
            repeated_max,
            dest_ptr,
            arg::exp_difference<R>(temperature)
        );

        auto reordered_exped_values = thrust::make_permutation_iterator(
            dest_ptr,
            col_major_index
        );

        thrust::dali_reduce_by_key(
            index_back_to_column,
            index_back_to_column + total_size,
            reordered_exped_values,
            keys_output.begin(),
            reduced_cols_begin,
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

    template<int x_bits, typename R,  typename DstPlan, typename SrcPlan>
    __global__ void SoftmaxTransposeKernel(DstPlan dst, SrcPlan src, mshadow::index_t xmax, R temperature) {
        const unsigned x_size = 1 << x_bits;
        const int y = blockIdx.x;
        __shared__ R s_rec[x_size];
        // step 1: get max
        if (threadIdx.x < xmax) {
            s_rec[threadIdx.x] = src.Eval(y, threadIdx.x);
        }
        for (unsigned x = x_size; x < xmax; x += x_size) {
            if (x + threadIdx.x < xmax) {
                R a = src.Eval(y, x + threadIdx.x);
                s_rec[threadIdx.x] = max(a, s_rec[threadIdx.x]);
            }
        }
        __syncthreads();
        if (threadIdx.x >= xmax) {
            s_rec[threadIdx.x] = s_rec[0];
        }
        __syncthreads();
        mshadow::cuda::Reduce1D<mshadow::red::maximum, x_bits>(s_rec);
        __syncthreads();
        R smax = s_rec[0];
        __syncthreads();
        s_rec[threadIdx.x] = 0.0f;
        __syncthreads();

        // calculate normalizer, with writeback
        for (unsigned x = 0; x < xmax; x += x_size) {
            if (x + threadIdx.x < xmax) {
                R p = expf((src.Eval(y, x + threadIdx.x) - smax) / temperature);
                s_rec[threadIdx.x] += p;
                // write back first, will fetch later
                dst.REval(y, x + threadIdx.x) = p;
            }
        }
        // calculate normalizer
        __syncthreads();
        mshadow::cuda::Reduce1D<mshadow::red::sum, x_bits>(s_rec);
        __syncthreads();
        R ssum = s_rec[0];

        for (unsigned x = 0; x < xmax; x += x_size) {
            if (x + threadIdx.x < xmax) {
                dst.REval(y, x + threadIdx.x) /= ssum;
            }
        }
    }
    template<typename R>
    inline void softmax_transpose(mshadow::Tensor<gpu, 2, R> dst,
                        const mshadow::Tensor<gpu, 2, R> src, R temperature = 1.0) {
      dim3 dimBlock(mshadow::cuda::kBaseThreadNum);
      dim3 dimGrid(dst.size(0));
      mshadow::utils::Check(dst.shape_ == src.shape_, "SoftmaxTranspose: shape mismatch");
      mshadow::cuda::CheckLaunchParam(dimGrid, dimBlock, "SoftmaxTranspose");
      cudaStream_t stream = mshadow::Stream<gpu>::GetStream(dst.stream_);
      SoftmaxTransposeKernel<mshadow::cuda::kBaseThreadBits, R>
          <<<dimGrid, dimBlock, 0, stream>>>
          (mshadow::expr::MakePlan(dst),
           mshadow::expr::MakePlan(src),
           dst.size(1),
           temperature);
    }
    #endif

    template<typename R>
    inline void softmax_transpose(mshadow::Tensor<cpu, 1, R> dst,
                        const mshadow::Tensor<cpu, 1, R> &src,
                        R& temperature) {
        R mmax = src[0];
        for (mshadow::index_t x = 1; x < dst.size(0); ++x) {
            if (mmax < src[x]) mmax = src[x];
        }
        R sum = 0.0f;
        for (mshadow::index_t x = 0; x < dst.size(0); ++x) {
            dst[x] = std::exp((src[x] - mmax) / temperature);
            sum += dst[x];
        }
        for (mshadow::index_t x = 0; x < dst.size(0); ++x) {
            dst[x] /= sum;
        }
    }
    template<typename R>
    inline void softmax_transpose(mshadow::Tensor<cpu, 2, R> dst,
                          const mshadow::Tensor<cpu, 2, R> &src,
                          R temperature) {
        mshadow::utils::Check(dst.shape_ == src.shape_, "SoftmaxTranspose: shape mismatch");
        for (mshadow::index_t y = 0; y < dst.size(0); ++y) {
            softmax_transpose(dst[y], src[y], temperature);
        }
    }

    template<typename R>
    void softmax(mshadow::Tensor<cpu,2,R> dst, mshadow::Tensor<cpu,2,R> src, R temperature = 1.0) {
        for (mshadow::index_t col = 0; col < dst.size(1); ++col) {
            R mmax = src[0][col];
            for (mshadow::index_t row = 1; row < dst.size(0); ++row) {
                if (mmax < src[row][col]) mmax = src[row][col];
            }
            R sum = 0.0f;
            for (mshadow::index_t row = 0; row < dst.size(0); ++row) {
                dst[row][col] = std::exp((src[row][col] - mmax) / temperature);
                sum += dst[row][col];
            }
            for (mshadow::index_t row = 0; row < dst.size(0); ++row) {
                dst[row][col] /= sum;
            }
        }
    }
}

#endif
