#ifndef DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H
#define DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H

#include <vector>
#define DALI_ARRAY_HIDE_LAZY 1
#include "dali/array/array.h"
#undef DALI_ARRAY_HIDE_LAZY
#include "dali/array/dtype.h"
#include "dali/array/function/expression.h"
#include "dali/array/function/typed_array.h"

////////////////////////////////////////////////////////////////////////////////
//                             MSHADOW_WRAPPER                                //
//                                   ---                                      //
//  This class would not be needed at all if we defined to_mshadow_expr       //
//  function on Array. The reason not to do that is to hide all mshadow usage //
//  in cpp files whereever possible.                                          //
////////////////////////////////////////////////////////////////////////////////

namespace lazy {
    template<int devT, typename T, int ndim>
    struct EvaluationSpec {
        EvaluationSpec(const bool& collapse_leading) : collapse_leading_(collapse_leading) {}
        EvaluationSpec(const EvaluationSpec& other) : collapse_leading_(other.collapse_leading_) {}
        EvaluationSpec(EvaluationSpec&& other) : collapse_leading_(other.collapse_leading_) {}
        EvaluationSpec() : collapse_leading_(true) {}

        bool collapse_leading_;
        auto fit_array_to_spec(const Array& array,
                               const memory::Device& device,
                               const std::vector<int>& output_shape) const ->
                decltype(TypedArray<devT,T>(array, device, output_shape).template d<ndim>()) {
            return TypedArray<devT,T>(array, device, output_shape).template d<ndim>(memory::AM_READONLY, collapse_leading_);
        }

        template<int ndim_dst>
        EvaluationSpec<devT, T, ndim_dst> d() const {
            return EvaluationSpec<devT, T, ndim_dst>{collapse_leading_};
        }

        template<int ndim_dst>
        EvaluationSpec<devT, T, ndim_dst> collapse_trailing_d() const {
            // TODO(jonathan): check that the incoming EvaluationSpec has the
            // same value for collapse leading or raise error
            return EvaluationSpec<devT, T, ndim_dst>{false};
        }

        template<int ndim_dst>
        EvaluationSpec<devT, T, ndim_dst> collapse_leading_d() const {
            // TODO(jonathan): check that the incoming EvaluationSpec has the
            // same value for collapse leading or raise error
            return EvaluationSpec<devT, T, ndim_dst>{true};
        }

        static EvaluationSpec<devT, T, ndim> collapse_trailing() {
            return EvaluationSpec<devT, T, ndim>(false);
        }

        static EvaluationSpec<devT, T, ndim> collapse_leading() {
            return EvaluationSpec<devT, T, ndim>(true);
        }
    };

}  // namespace lazy

template<int devT, typename T, typename ExprT>
struct MshadowWrapper {
    template<int ndim>
    static inline auto wrap(const ExprT& sth,
                            memory::Device device,
                            const std::vector<int>& output_shape,
                            const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) ->
            decltype(sth.template to_mshadow_expr<devT, T, ndim>(device, output_shape, wrap_array)) {
        // static_assert(mshadow::expr::ExpInfo<WrappedArrayT>::kDim != -1,
        //     "Mshadow expression for WrappedArrayT has no mshadow::expr::ExpInfo kdim defined. Specialize this classes ExpInfo to know the kDim.");
        // static_assert(mshadow::expr::ExpInfo<decltype(sth.template to_mshadow_expr<devT,T>(device, output_shape, wrap_array))>::kDim != -1,
        //     "Mshadow expression for wrapped lazy expression no mshadow::expr::ExpInfo kdim defined. Specialize this classes ExpInfo to know the kDim.");
        // static_assert(
        //     !(mshadow::expr::ExpInfo<decltype(sth.template to_mshadow_expr<devT,T>(device, output_shape, wrap_array))>::kDim == 1 &&
        //     mshadow::expr::ExpInfo<WrappedArrayT>::kDim == 2), "Input expression has ndim = 1 while wrap_array calls for ndim = 2");
        // static_assert(
        //     !(mshadow::expr::ExpInfo<decltype(sth.template to_mshadow_expr<devT,T>(device, output_shape, wrap_array))>::kDim == 2 &&
        //     mshadow::expr::ExpInfo<WrappedArrayT>::kDim == 1), "Input expression has ndim = 2 while wrap_array calls for ndim = 1");
        // static_assert(
        //     mshadow::expr::ExpInfo<decltype(sth.template to_mshadow_expr<devT,T>(device, output_shape, wrap_array))>::kDim ==
        //     mshadow::expr::ExpInfo<WrappedArrayT>::kDim, "Input expression to wrap has wrong dimensionality.");
        return sth.template to_mshadow_expr<devT, T, ndim>(device, output_shape, wrap_array);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,Array> {
    template<int ndim>
    static inline auto wrap(const Array& array,
                            memory::Device device,
                            const std::vector<int>& output_shape,
                            const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) ->
            decltype(wrap_array.fit_array_to_spec(array, device, output_shape)) {
        return wrap_array.fit_array_to_spec(array, device, output_shape);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,float> {
    template<int ndim>
    static inline T wrap(const float& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         const lazy::EvaluationSpec<devT, T, ndim>&) {
        return (T)scalar;
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,double> {
    template<int ndim>
    static inline T wrap(const double& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         const lazy::EvaluationSpec<devT, T, ndim>&) {
        return (T)scalar;
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,int> {
    template<int ndim>
    static inline T wrap(const int& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         const lazy::EvaluationSpec<devT, T, ndim>&) {
        return (T)scalar;
    }
};

////////////////////////////////////////////////////////////////////////////////
//                             LAZY_FUNCTION                                  //
////////////////////////////////////////////////////////////////////////////////


template<typename Class, typename... Args>
struct LazyFunction: public LazyExp<Class> {
    static const int  evaluation_dim;
    static const bool collapse_leading;
    std::vector<int> bshape_;
    DType dtype_;

    LazyFunction(Args... args);

    static std::vector<int> lazy_output_bshape(const Args&... args);

    static DType lazy_output_dtype(const Args&... args);

    const std::vector<int>& bshape() const;

    const DType& dtype() const;
};

#include "dali/array/function/lazy_function-impl.h"

#endif
