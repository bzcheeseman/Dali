#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include "dali/tensor/Mat.h"
#include "dali/math/ThrustSoftmax.h"
#include "dali/utils/core_utils.h"
#include "dali/math/memory_bank/MemoryBank.h"

using std::vector;


typedef float R;

template<int buffer_bits, typename R,  typename DstPlan, typename SrcPlan>
__global__ void SoftmaxKernel(DstPlan dst, SrcPlan src, mshadow::index_t num_rows, R temperature) {
    const unsigned buffer_size = 1 << buffer_bits;
    const int column     = blockIdx.y;
    const int thread_idx = threadIdx.x;
    __shared__ R buffer[buffer_size];
    // step 1: get max
    if (thread_idx < num_rows) {
        buffer[thread_idx] = src.Eval(thread_idx, column);
    }
    for (unsigned offset = buffer_size; offset < num_rows; offset += buffer_size) {
        const int row = offset + thread_idx;
        if (row < num_rows) {
            const R a = src.Eval(row, column);
            buffer[thread_idx] = max(buffer[thread_idx], a);
        }
    }
    __syncthreads();
    // if number of rows is smaller than buffer,
    // fill buffer with copy of buffer[0] - this
    // makes sure reduction does not use uninitialized
    // values in the buffer and returns correct max.
    if (thread_idx >= num_rows) {
        buffer[thread_idx] = buffer[0];
    }
    __syncthreads();
    mshadow::cuda::ReduceX<mshadow::red::maximum, buffer_bits, R>(buffer, thread_idx);
    __syncthreads();
    // every thread memorizes max value in column,
    // so that we can reuse the buffer, for next
    // task
    R max_value_in_column = buffer[0];
    __syncthreads();
    // clear buffer (so that sum works out later)
    buffer[thread_idx] = 0.0f;
    __syncthreads();
    for (unsigned offset = 0; offset < num_rows; offset += buffer_size) {
        const int row = offset + thread_idx;
        if (row < num_rows) {
            R p = expf((src.Eval(row, column) - max_value_in_column) / temperature);
            // add sum to buffer, so that we can later reduce it to
            // column-wise sum of exps and use as normalizer.
            buffer[thread_idx] += p;
            // save exped value to the corresponding idx in destination.
            dst.REval(row, column) = p;
        }
    }
    __syncthreads();
    // calculate normalizer by reducing partial sums
    mshadow::cuda::ReduceX<mshadow::red::sum, buffer_bits, R>(buffer, thread_idx);
    __syncthreads();
    R colwise_sum = buffer[0];

    for (unsigned offset = 0; offset < num_rows; offset += buffer_size) {
        const int row = offset + thread_idx;
        if (row < num_rows) {
            dst.REval(row, column) /= colwise_sum;
        }
    }
}

// Note: in a dim3 (width, height, depth)
// every uninitialized dimension defaults to 1.

// Note: <<<Dg, Db, Ns, S>>> CUDA Language Extension is explained here:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration
template<typename R>
void softmax(mshadow::Tensor<mshadow::gpu, 2, R> dst,
                    const mshadow::Tensor<mshadow::gpu, 2, R> src, R temperature = 1.0) {
  const int num_threads = 1024;//mshadow::cuda::kBaseThreadNum;
  const int thread_bits = 10;//mshadow::cuda::kBaseThreadBits;
  dim3 tiles(1, dst.size(1));
  // block size is a matrix column
  dim3 within_tile(num_threads);
  mshadow::utils::Check(dst.shape_ == src.shape_, "Softmax: shape mismatch");
  // mshadow::cuda::CheckLaunchParam(blockGridRows, threadBlockRows, "Softmax");
  cudaStream_t stream = mshadow::Stream<mshadow::gpu>::GetStream(dst.stream_);
  SoftmaxKernel<thread_bits, R>
      <<<tiles, within_tile, 0, stream>>>
      (mshadow::expr::MakePlan(dst),
       mshadow::expr::MakePlan(src),
       dst.size(0),
       temperature);
}

int main() {
    dali_init();
    auto bob = Mat<R>(5000, 5000, weights<R>::uniform(10, 20));
    auto bob_col_softmax = Mat<R>::zeros_like(bob);

    // set the computing streams
    softmax(bob_col_softmax.w().mutable_gpu_data(), bob.w().gpu_data());
    TensorOps::softmax(bob_col_softmax.w().mutable_gpu_data(), bob.w().gpu_data());

    int iter = 10;

    for (int i = 0; i < iter; i++) {
        {
            utils::Timer t1("Softmax col-wise (Dali)");
            // our softmax
            softmax(bob_col_softmax.w().mutable_gpu_data(), bob.w().gpu_data());
            cudaDeviceSynchronize();
        }
        {
            utils::Timer t2("Softmax col-wise (Thrust)");
            // thrust softmax
            TensorOps::softmax(bob_col_softmax.w().mutable_gpu_data(), bob.w().gpu_data());
            cudaDeviceSynchronize();
        }
        {
            utils::Timer t2("Softmax row-wise (mshadow)");
            // thrust softmax
            TensorOps::softmax_transpose(bob_col_softmax.w().mutable_gpu_data(), bob.w().gpu_data());
            cudaDeviceSynchronize();
        }
    }

    utils::Timer::report();
}
