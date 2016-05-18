#ifndef DALI_ARRAY_FUNCTION_ARGS_DALI_GEMM_ENGINE_EXP_H
#define DALI_ARRAY_FUNCTION_ARGS_DALI_GEMM_ENGINE_EXP_H

#include "dali/config.h"

#include <cassert>

#include <mshadow/tensor.h>
#include <mshadow/dot_engine-inl.h>
#include "dali/array/function/args/igemm.h"

namespace mshadow {
    namespace expr {
        template<typename Device>
        mshadow::Stream<Device>* get_default_gemm_stream() {
            assert(false);
        }

#ifdef DALI_USE_CUDA
        template<>
        mshadow::Stream<mshadow::gpu>* get_default_gemm_stream();
#endif

        template<>
        mshadow::Stream<mshadow::cpu>* get_default_gemm_stream();

        // new DotExpr that removes transpose templating
        // in lieu of bool templating
        template<typename DType, typename Device>
        struct DaliDotExp: public Exp<DaliDotExp<DType, Device>,
                                  DType, type::kComplex> {

          /*! \brief left operand */
          const Tensor<Device, 2, DType> &lhs_;
          /*! \brief right operand */
          const Tensor<Device, 2, DType> &rhs_;
          /*! \brief scale over result */
          DType scale_;

          bool ltransposed;
          bool rtransposed;

          /*! \brief constructor */
          explicit DaliDotExp(const Tensor<Device, 2, DType> &lhs,
                              const Tensor<Device, 2, DType> &rhs,
                              bool ltransposed_,
                              bool rtransposed_,
                              DType scale)
                : lhs_(lhs),
                  rhs_(rhs),
                  scale_(scale),
                  ltransposed(ltransposed_),
                  rtransposed(rtransposed_) {

          }
        };

        template<typename DType, typename Device>
        inline DaliDotExp<DType, Device> dali_gemm(
            const Tensor<Device, 2, DType> &lhs,
            const Tensor<Device, 2, DType> &rhs,
            bool ltransposed,
            bool rtransposed,
            DType scale) {
          return DaliDotExp<DType, Device>(lhs, rhs, ltransposed, rtransposed, scale);
        }


        // gemm call happens here:
        // dst = dot(lhs[.T], rhs[.T])
        template<typename SV, typename xpu, typename DType>
        struct GemmEngine {
          inline static void Eval(Tensor<xpu, 2, DType> *p_dst,
                                  const Tensor<xpu, 2, DType> &lhs,
                                  const Tensor<xpu, 2, DType> &rhs,
                                  bool transpose_left,
                                  bool transpose_right,
                                  DType scale) {
            Tensor<xpu, 2, DType> &dst = *p_dst;
            dst.set_stream(get_default_gemm_stream<xpu>());
            // set kernel stream
            // if there is no stream, crush
            BLASEngine<xpu, DType>::SetStream(dst.stream_);

            // use column major argument to compatible with most BLAS
            BLASEngine<xpu, DType>::gemm
                (dst.stream_,
                 transpose_right, transpose_left,
                 rhs.size(1),
                 lhs.size(0),
                 rhs.size(0),
                 DType(scale * SV::AlphaBLAS()),
                 rhs.dptr_, rhs.stride_,
                 lhs.dptr_, lhs.stride_,
                 DType(SV::BetaBLAS()),
                 dst.dptr_, dst.stride_);
          }
        };

        // expression system kicks off eval here upon assignment
        template<typename SV, typename Device, typename DType>
        struct ExpComplexEngine<SV,
                                Tensor<Device, 2, DType>,
                                DaliDotExp<DType, Device>,
                                DType> {
          inline static void Eval(Tensor<Device, 2, DType> *dst,
                                  const DaliDotExp<DType, Device> &exp) {
            GemmEngine<SV, Device, DType>::Eval(
                dst,
                exp.lhs_,
                exp.rhs_,
                exp.ltransposed,
                exp.rtransposed,
                exp.scale_
            );
          }
        };
    } // namespace expr
} // namespace mshadow


#endif
