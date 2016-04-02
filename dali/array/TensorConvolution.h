#include <mshadow/tensor.h>

#include "dali/config.h"
#include "dali/array/TensorInternal.h"
#include "dali/array/ThrustUtils.h"
#include "dali/array/memory_bank/MemoryBank.h"

namespace TensorOps {
    using mshadow::gpu;
    using mshadow::cpu;
    /////////////////////// circular_convolution /////////////////////////////////////////


    #ifdef DALI_USE_CUDA

    template<int x_bits, typename R,  typename DstPlan, typename MatPlan, typename ShiftPlan>
    __global__ void CircularConvolutionKernel(DstPlan dest, MatPlan mat, ShiftPlan shift, mshadow::index_t num_cols) {
        const unsigned num_threads = 1 << x_bits;
        const int row        = blockIdx.x;
        const int thread_idx = threadIdx.x;

        for (int shift_idx = 0; shift_idx < num_cols; ++shift_idx) {
            R shift_mul = shift.Eval(row, shift_idx);
            for (int col = thread_idx; col < num_cols; col += num_threads) {
                int offset = col + shift_idx;
                offset -= (offset >= num_cols) ? num_cols : 0;
                dest.REval(row,col) += shift_mul * mat.Eval(row, offset);
            }
            __syncthreads();
        }
    }

    template<typename R>
    void circular_convolution(mshadow::Tensor<gpu, 2, R> dest,
                    const mshadow::Tensor<gpu, 2, R>& mat,
                    const mshadow::Tensor<gpu, 2, R>& shift) {

        const int num_threads = mshadow::cuda::kBaseThreadNum;
        const int thread_bits = mshadow::cuda::kBaseThreadBits;

        dim3 tiles(dest.size(0));
        // block size is a matrix column
        dim3 within_tile(num_threads);

        cudaStream_t stream = mshadow::Stream<mshadow::gpu>::GetStream(dest.stream_);

        CircularConvolutionKernel<thread_bits, R>
                <<<tiles, within_tile, 0, stream>>>
                (mshadow::expr::MakePlan(dest),
                 mshadow::expr::MakePlan(mat),
                 mshadow::expr::MakePlan(shift),
                dest.size(1));

        cudaDeviceSynchronize();
    }
    #endif

    template<typename R>
    void circular_convolution(mshadow::Tensor<cpu, 2, R> dest,
                              const mshadow::Tensor<cpu, 2, R>& mat,
                              const mshadow::Tensor<cpu, 2, R>& shift) {

        for (int row = 0; row < mat.shape_[0]; ++row) {
            for (int col = 0; col < mat.shape_[1]; ++col) {
                for (int shift_idx = 0; shift_idx < mat.shape_[1]; ++shift_idx) {
                    // here we intentionally avoid expensive % operation.
                    int offset = col + shift_idx;
                    if (offset >= mat.shape_[1]) {
                        offset -= mat.shape_[1];
                    }
                    dest[row][col] = dest[row][col] + mat[row][offset] * shift[row][shift_idx];
                }
            }
        }
    }


    template<typename R>
    void circular_convolution(TensorInternal<R,2> dest,
                    TensorInternal<R,2> mat,
                    TensorInternal<R,2> shift) {
        #ifdef DALI_USE_CUDA
        if (mat.compute_me_on_gpu()) {
            circular_convolution(dest.mutable_gpu_data(), mat.gpu_data(), shift.gpu_data());
            return;
        }
        #endif
        circular_convolution(dest.mutable_cpu_data(), mat.cpu_data(), shift.cpu_data());
    }
}
