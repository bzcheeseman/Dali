#ifndef DALI_MATH_LAZY_TRANSPOSE_SOFTMAX_H
#define DALI_MATH_LAZY_TRANSPOSE_SOFTMAX_H

#include "mshadow/tensor.h"

namespace dali_expr {
    #ifdef DALI_USE_CUDA
    // For some reason kernel does not write to any cell
    // except first cell [0,0]
    template<int y_bits, typename DType,  typename DstPlan, typename SrcPlan>
    __global__ void SoftmaxTransposeKernel(DstPlan dst, SrcPlan src, mshadow::index_t ymax) {
      const unsigned y_size = 1 << y_bits;
      const int x = blockIdx.y;
      __shared__ DType s_rec[y_size];
      // step 1: get max
      if (threadIdx.y < ymax) {
        s_rec[threadIdx.y] = src.Eval(threadIdx.y, x);
      }
      for (unsigned y = y_size; y < ymax; y += y_size) {
        if (y + threadIdx.y < ymax) {
          DType a = src.Eval(y + threadIdx.y, x);
          s_rec[threadIdx.y] = max(a, s_rec[threadIdx.y]);
        }
      }
      __syncthreads();
      if (threadIdx.y >= ymax) {
        s_rec[threadIdx.y] = s_rec[0];
      }
      __syncthreads();
      mshadow::cuda::Reduce1D<mshadow::red::maximum, y_bits>(s_rec);
      __syncthreads();
      DType smax = s_rec[0];
      __syncthreads();
      s_rec[threadIdx.y] = 0.0f;
      __syncthreads();

      // calculate normalizer, with writeback
      for (unsigned y = 0; y < ymax; y += y_size) {
        if (y + threadIdx.y < ymax) {
          DType p = expf(src.Eval(y + threadIdx.y, x) - smax);
          s_rec[threadIdx.y] += p;
          // write back first, will fetch later
          dst.REval(y + threadIdx.y, x) = p;
        }
      }
      // calculate normalizer
      __syncthreads();
      mshadow::cuda::Reduce1D<mshadow::red::sum, y_bits>(s_rec);
      __syncthreads();
      DType ssum = s_rec[0];

      for (unsigned y = 0; y < ymax; y += y_size) {
        if (y + threadIdx.y < ymax) {
          dst.REval(y + threadIdx.y, x) /= ssum;
        }
      }
    }

    template<typename DType>
    inline void SoftmaxTransposeCuda(mshadow::Tensor<mshadow::gpu, 2, DType> &dst,
                        const mshadow::Tensor<mshadow::gpu, 2, DType> &src) {
      dim3 dimBlock(mshadow::cuda::kBaseThreadNum);
      dim3 dimGrid(dst.size(1));
      mshadow::utils::Check(dst.shape_ == src.shape_, "SoftmaxTranspose: shape mismatch");
      mshadow::cuda::CheckLaunchParam(dimGrid, dimBlock, "SoftmaxTranspose");
      auto stream = mshadow::Stream<mshadow::gpu>::GetStream(dst.stream_);
      SoftmaxTransposeKernel<mshadow::cuda::kBaseThreadBits, DType>
          <<<dimGrid, dimBlock, 0, stream>>>
          (mshadow::expr::MakePlan(dst),
           mshadow::expr::MakePlan(src),
           dst.size(0)
          );
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
