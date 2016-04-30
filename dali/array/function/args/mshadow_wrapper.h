#ifndef DALI_ARRAY_FUNCTION_ARGS_MSHADOW_WRAPPER_H
#define DALI_ARRAY_FUNCTION_ARGS_MSHADOW_WRAPPER_H

#include "dali/array/function/typed_array.h"
#include "dali/array/memory/device.h"

////////////////////////////////////////////////////////////////////////////////
//                           MSHADOW_WRAPPER_EXP                              //
//                                   ---                                      //
//  This expression is used to inject Dali striding information to mshadow    //
//  expression processor                                                      //
////////////////////////////////////////////////////////////////////////////////

namespace mshadow {
    namespace expr {
        template<typename SrcExp, typename DType, int srcdim>
        struct MshadowWrapperExp: public MakeTensorExp<
                                            MshadowWrapperExp<SrcExp, DType, srcdim>,
                                            SrcExp, srcdim, DType
                                         > {
            const SrcExp src_;

            MshadowWrapperExp(const SrcExp &src, const Array& dali_src) :
                    src_(src) {
                this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
            }
        };

        template<typename SrcExp, typename DType, int etype>
        inline MshadowWrapperExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
        MakeMshadowWrapperExp(const Exp<SrcExp, DType, etype> &src, const Array& dali_src) {
            return MshadowWrapperExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), dali_src);
        }

        template<typename SrcExp, typename DType, int srcdim>
        struct ExpInfo<MshadowWrapperExp<SrcExp, DType, srcdim> > {
            static const int kDimSrc = ExpInfo<SrcExp>::kDim;
            static const int kDim = kDimSrc >= 0 ? srcdim : -1;
            static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
        };

        template<typename SrcExp, typename DType, int srcdim>
        struct ShapeCheck<srcdim, MshadowWrapperExp<SrcExp, DType, srcdim> > {
            inline static Shape<srcdim>
            Check(const MshadowWrapperExp<SrcExp, DType, srcdim> &t) {
                return t.shape_;
            }
        };

        template<typename SrcExp, typename DType, int srcdim>
        struct Plan<MshadowWrapperExp<SrcExp, DType, srcdim>, DType> {
          public:
            explicit Plan(const MshadowWrapperExp<SrcExp, DType, srcdim> &e) :
                    src_(MakePlan(e.src_)) {
            }

            MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
                return src_.Eval(i, j);
            }

          private:
            Plan<SrcExp, DType> src_;
        };
    } //namespace expr
} // namespace mshadow
////////////////////////////////////////////////////////////////////////////////
//                             MSHADOW_WRAPPER                                //
//                                   ---                                      //
//  This class would not be needed at all if we defined to_mshadow_expr       //
//  function on Array. The reason not to do that is to hide all mshadow usage //
//  in cpp files whereever possible.                                          //
////////////////////////////////////////////////////////////////////////////////


template<int devT,typename T, typename ExprT>
struct MshadowWrapper {
    static inline auto wrap(const ExprT& sth, memory::Device device) ->
            decltype(sth.template to_mshadow_expr<devT,T>(device)) {
        return sth.template to_mshadow_expr<devT,T>(device);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,Array> {
    static inline auto wrap(const Array& array, memory::Device device) ->
            decltype(MakeMshadowWrapperExp(TypedArray<devT,T>(array, device).d1(), array)) {
        return MakeMshadowWrapperExp(TypedArray<devT,T>(array, device).d1(), array);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,float> {
    static inline T wrap(const float& scalar, memory::Device device) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,double> {
    static inline T wrap(const double& scalar, memory::Device device) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,int> {
    static inline T wrap(const int& scalar, memory::Device device) { return (T)scalar; }
};

#endif // DALI_ARRAY_FUNCTION_ARGS_MSHADOW_WRAPPER_H
