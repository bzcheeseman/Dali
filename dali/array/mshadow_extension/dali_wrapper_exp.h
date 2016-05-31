#ifndef DALI_ARRAY_FUNCTION_ARGS_DALI_WRAPPER_EXP_H
#define DALI_ARRAY_FUNCTION_ARGS_DALI_WRAPPER_EXP_H

#include "dali/config.h"
#include <mshadow/tensor.h>

#include "dali/array/memory/device.h"
#include "dali/utils/assert2.h"

////////////////////////////////////////////////////////////////////////////////
//                           DALI_WRAPPER_EXP                                 //
//                                   ---                                      //
//  This expression is used to inject Dali striding information to mshadow    //
//  expression processor                                                      //
//  TODO(szymon): make this code more efficient:                              //
//     -> On CPU code is evaluated serially, so we can replace modulos with   //
//        if statements => MAD_EFFICIENT (also omp parallel?)                 //
//     -> On GPU we need to make sure that strides and shapes are in the      //
//        lowest level of cache possible. I think __shared__ needs to be used //
//        but injecting this into mshadow might cause insanity                //
////////////////////////////////////////////////////////////////////////////////


template<typename Device, int srcdim, typename DType>
struct DaliWrapperExp : public mshadow::TRValue<
            DaliWrapperExp<Device, srcdim, DType>,
            Device,
            srcdim,
            DType
        > {
    typedef mshadow::Tensor<Device, srcdim, DType> src_t;
    const src_t src_;
    mshadow::Shape<srcdim> shape_;
    mshadow::Stream<Device> * stream_;
    const Array array;

    DaliWrapperExp(const src_t& src, const Array& dali_src) :
            src_(src),
            stream_(src.stream_),
            array(dali_src) {
        ASSERT2(src_.shape_[srcdim - 1] == src_.stride_,
                "DaliWrapperExp should never reach that condition (only tensors should be passed as arguments).");
        ASSERT2(array.ndim() <= DALI_MAX_STRIDED_DIMENSION,
                "Striding only supported for Tensors up to DALI_MAX_STRIDED_DIMENSION dimensions.");

        this->shape_ = mshadow::expr::ShapeCheck<srcdim, src_t>::Check(src_);
    }

    MSHADOW_XINLINE mshadow::index_t size(mshadow::index_t idx) const {
        return shape_[idx];
    }

    template<typename E, int etype>
    inline DaliWrapperExp<Device, srcdim, DType>&
    operator=(const mshadow::expr::Exp<E, DType, etype>& exp) {
        return this->__assign(exp);
    }

    inline DaliWrapperExp<Device, srcdim, DType>&
    operator=(const DType& exp) {
        return this->__assign(exp);
    }

};

template<typename Device, int srcdim, typename DType>
inline DaliWrapperExp<Device, srcdim, DType>
MakeDaliWrapperExp(const mshadow::Tensor<Device, srcdim, DType> &src, const Array& dali_src) {
    return DaliWrapperExp<Device, srcdim, DType>(src.self(), dali_src);
}


namespace mshadow {
    namespace expr {

        template<typename Device, int srcdim, typename DType>
        struct ExpInfo<DaliWrapperExp<Device, srcdim, DType>> {
            static const int kDimSrc = ExpInfo<typename DaliWrapperExp<Device, srcdim, DType>::src_t>::kDim;
            static const int kDim = kDimSrc >= 0 ? srcdim : -1;
            static const int kDevMask = ExpInfo<typename DaliWrapperExp<Device, srcdim, DType>::src_t>::kDevMask;
        };

        template<typename Device, int srcdim, typename DType>
        struct ShapeCheck<srcdim, DaliWrapperExp<Device,srcdim,DType>> {
            inline static Shape<srcdim>
            Check(const DaliWrapperExp<Device, srcdim, DType> &t) {
                return t.shape_;
            }
        };

        template<typename Device, int srcdim, typename DType>
        struct Plan<DaliWrapperExp<Device, srcdim, DType>, DType> {
          public:
            explicit Plan(const DaliWrapperExp<Device, srcdim, DType> &e) :
                    src_(MakePlan(e.src_)),
                    ndim(e.array.ndim()),
                    has_strides(!e.array.strides().empty()),
                    src_trailing_dim(e.src_.stride_) {
                for (int i = 0; i < ndim; ++i) {
                    shape[i] = e.array.shape()[i];
                    if (has_strides) strides[i] = e.array.strides()[i];
                }
            }

            MSHADOW_XINLINE void map_indices_using_stride(index_t& new_i, index_t& new_j, index_t i, index_t j) const {
                j += i * src_trailing_dim;

                new_i = 0;
                new_j = 0;
                for (int dim_idx = ndim - 1; dim_idx >= 0; --dim_idx) {
                    new_j += (j % shape[dim_idx]) * strides[dim_idx];
                    j /=  shape[dim_idx];
                }

                int old_new_j = new_j;
                if (srcdim > 1) {
                    if (new_j >= 0) {
                        new_i = new_j / src_trailing_dim;
                        new_j = new_j % src_trailing_dim;
                    } else {
                        new_i = -((-new_j) / src_trailing_dim);
                        new_j = -((-new_j) % src_trailing_dim);
                    }
                }
            }

            MSHADOW_XINLINE DType& REval(index_t i, index_t j) {
                if (!has_strides) {
                    return src_.REval(i, j);
                } else {
                    index_t new_i, new_j;
                    map_indices_using_stride(new_i, new_j, i, j);
                    return src_.REval(new_i, new_j);
                }
            }

            MSHADOW_XINLINE const DType& Eval(index_t i, index_t j) const {
                if (!has_strides) {
                    return src_.Eval(i, j);
                } else {
                    index_t new_i, new_j;
                    map_indices_using_stride(new_i, new_j, i, j);

                    return src_.Eval(new_i, new_j);
                }
            }

          private:
            Plan<typename DaliWrapperExp<Device, srcdim, DType>::src_t, DType> src_;
            int ndim;
            int shape[DALI_MAX_STRIDED_DIMENSION];
            int strides[DALI_MAX_STRIDED_DIMENSION];
            int src_trailing_dim;
            const bool has_strides;
        };

        template<typename SV, typename Device, typename DType,
                 typename SrcExp, typename Reducer, int m_dimkeep>
        struct ExpComplexEngine<SV,
                                DaliWrapperExp<Device, 1, DType>,
                                ReduceTo1DExp<SrcExp, DType, Reducer, m_dimkeep>,
                                DType> {
            static const int dimkeep = ExpInfo<SrcExp>::kDim - m_dimkeep;
            inline static void Eval(DaliWrapperExp<Device, 1, DType> *dst,
                                    const ReduceTo1DExp<SrcExp, DType,
                                                        Reducer, m_dimkeep> &exp) {
                TypeCheckPass<m_dimkeep != 1>
                        ::Error_Expression_Does_Not_Meet_Dimension_Req();
                MapReduceKeepHighDim<SV, Reducer, dimkeep>(dst, exp.src_, exp.scale_);
            }
        };
        template<typename SV, typename Device, typename DType,
                 typename SrcExp, typename Reducer>
        struct ExpComplexEngine<SV,
                                DaliWrapperExp<Device, 1, DType>,
                                ReduceTo1DExp<SrcExp, DType, Reducer, 1>, DType> {
            inline static void Eval(DaliWrapperExp<Device, 1, DType> *dst,
                                    const ReduceTo1DExp<SrcExp, DType, Reducer, 1> &exp) {
                MapReduceKeepLowest<SV, Reducer>(dst, exp.src_, exp.scale_);
            }
        };


        template<typename Device, int srcdim, typename DType>
        struct StreamInfo<Device, DaliWrapperExp<Device, srcdim, DType> > {
            inline static Stream<Device> *Get(const DaliWrapperExp<Device, srcdim, DType>& t) {
                return t.src_.stream_;
            }
        };

    } //namespace expr
} // namespace mshadow

#endif // DALI_ARRAY_FUNCTION_ARGS_DALI_WRAPPER_EXP_H
