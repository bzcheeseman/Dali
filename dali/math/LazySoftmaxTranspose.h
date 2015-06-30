#ifndef DALI_MATH_LAZY_TRANSPOSE_SOFTMAX_H
#define DALI_MATH_LAZY_TRANSPOSE_SOFTMAX_H

#include "mshadow/tensor.h"

namespace dali_expr {
    #ifdef DALI_USE_CUDA
    template<int x_bits, typename DType,  typename DstPlan, typename SrcPlan>
    __global__ void SoftmaxTransposeKernel(DstPlan dst, SrcPlan src, mshadow::index_t xmax) {
      const unsigned x_size = 1 << x_bits;
      const int y = blockIdx.y;
      __shared__ DType s_rec[x_size];
      // step 1: get max
      if (threadIdx.x < xmax) {
        s_rec[threadIdx.x] = src.Eval(y, threadIdx.x);
      }
      for (unsigned x = x_size; x < xmax; x += x_size) {
        if (x + threadIdx.x < xmax) {
          DType a = src.Eval(y, x + threadIdx.x);
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
      DType smax = s_rec[0];
      __syncthreads();
      s_rec[threadIdx.x] = 0.0f;
      __syncthreads();

      // calculate normalizer, with writeback
      for (unsigned x = 0; x < xmax; x += x_size) {
        if (x + threadIdx.x < xmax) {
          DType p = expf(src.Eval(y, x + threadIdx.x) - smax);
          s_rec[threadIdx.x] += p;
          // write back first, will fetch later
          dst.REval(y, x + threadIdx.x) = p;
        }
      }
      // calculate normalizer
      __syncthreads();
      mshadow::cuda::Reduce1D<mshadow::red::sum, x_bits>(s_rec);
      __syncthreads();
      DType ssum = s_rec[0];

      for (unsigned x = 0; x < xmax; x += x_size) {
        if (x + threadIdx.x < xmax) {
          dst.REval(y, x + threadIdx.x) /= ssum;
        }
      }
    }

    template<typename DType>
    inline void SoftmaxTransposeCuda(mshadow::Tensor<mshadow::gpu, 2, DType> &dst,
                        const mshadow::Tensor<mshadow::gpu, 2, DType> &src) {
      dim3 dimBlock(mshadow::cuda::kBaseThreadNum);
      dim3 dimGrid(dst.size(0));
      mshadow::utils::Check(dst.shape_ == src.shape_, "Softmax: shape mismatch");
      mshadow::cuda::CheckLaunchParam(dimGrid, dimBlock, "Softmax");
      auto stream = mshadow::Stream<mshadow::gpu>::GetStream(dst.stream_);
      SoftmaxTransposeKernel<mshadow::cuda::kBaseThreadBits, DType>
          <<<dimGrid, dimBlock, 0, stream>>>
          (mshadow::expr::MakePlan(dst),
           mshadow::expr::MakePlan(src),
           dst.size(0));
    }
    #endif

    // softmax on the transposed direction
    template<typename DType>
    inline void SoftmaxTranspose(mshadow::Tensor<mshadow::cpu, 2, DType> dst,
                        const mshadow::Tensor<mshadow::cpu, 2, DType> &energy) {
        mshadow::utils::Check(dst.shape_ == energy.shape_, "SoftmaxTranspose: shape mismatch");
        for (mshadow::index_t col = 0; col < dst.size(1); ++col) {
            DType mmax = energy[0][col];
            for (mshadow::index_t row = 1; row < dst.size(0); ++row) {
                if (mmax < energy[row][col]) mmax = energy[row][col];
            }
            DType sum = 0.0f;
            for (mshadow::index_t row = 0; row < dst.size(0); ++row) {
                dst[row][col] = std::exp(energy[row][col] - mmax);
                sum += dst[row][col];
            }
            for (mshadow::index_t row = 0; row < dst.size(0); ++row) {
                dst[row][col] /= sum;
            }
        }
    }

    #ifdef DALI_USE_CUDA
    template<typename DType>
    inline void SoftmaxTranspose(mshadow::Tensor<mshadow::gpu, 2, DType> dst,
                        const mshadow::Tensor<mshadow::gpu, 2, DType>& src) {
        SoftmaxTransposeCuda(dst, src);
    }
    #endif

    template<typename EType, typename DType>
    struct SoftmaxTransposeExpression: public mshadow::expr::Exp<SoftmaxTransposeExpression<EType, DType>,
                                      DType, mshadow::expr::type::kComplex> {
        const EType &exp;
        explicit SoftmaxTransposeExpression(const EType &e) : exp(e) {}
        inline auto T(void) const -> const mshadow::expr::TransposeExp<decltype(*this), DType> {
            return mshadow::expr::TransposeExp<decltype(*this), DType>(this->self());
        }
    };
}

namespace mshadow {
    namespace expr {
        template<typename SV, typename Device, typename DType, template <typename, int, typename> class tensor_t>
        struct ExpComplexEngine<SV,
                                tensor_t<Device, 2, DType>,
                                dali_expr::SoftmaxTransposeExpression< tensor_t<Device, 2, DType>, DType >,
                                DType > {
            inline static void Eval(tensor_t<Device, 2, DType> *dst,
                                    const dali_expr::SoftmaxTransposeExpression< tensor_t<Device, 2, DType>, DType > &exp) {
                dali_expr::SoftmaxTranspose(*dst, exp.exp);
            }
        };
    }
}

#endif
