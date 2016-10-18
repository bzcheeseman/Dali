#ifndef DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H
#define DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H

#include <vector>
#define DALI_ARRAY_HIDE_LAZY 1
#include "dali/array/array.h"
#undef DALI_ARRAY_HIDE_LAZY
#include "dali/array/dtype.h"
#include "dali/array/function/expression.h"
#include "dali/array/function/typed_array.h"
#include "dali/array/function/evaluation_dim.h"

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
                            const lazy::EvaluationSpec<devT, T, ndim>& eval_spec) ->
            decltype(sth.template to_mshadow_expr<devT, T, ndim>(device, output_shape, eval_spec)) {
        typedef decltype(sth.template to_mshadow_expr<devT,T, ndim>(device, output_shape, eval_spec)) expected_return_t;

        static const int the_dim_that_was_requested          = ndim;
        static const int the_dim_that_expr_claims_it_returns = lazy::LazyEvaluationDim<ExprT>::value;
        static const int the_dim_that_expr_actually_returned = mshadow::expr::ExpInfo<expected_return_t>::kDim;

        static_assert(the_dim_that_expr_claims_it_returns != lazy::EVALUATION_DIM_ERROR,
                "Lazy expression encountered an error when computing evaluation dim.");


        static_assert(the_dim_that_expr_claims_it_returns == lazy::EVALUATION_DIM_ANY ||
                      the_dim_that_expr_claims_it_returns == the_dim_that_was_requested,
                  "Lazy expression is incapable of returing the requested dim.");

        const bool fulfilled_return_any = (the_dim_that_expr_claims_it_returns == lazy::EVALUATION_DIM_ANY &&
                                           the_dim_that_expr_actually_returned == the_dim_that_was_requested);

        const bool returned_what_it_advertised = the_dim_that_expr_claims_it_returns == the_dim_that_expr_actually_returned;

        static_assert(fulfilled_return_any || returned_what_it_advertised,
                "Lazy expression did not return the lazy dim it claimed it could.");

        return sth.template to_mshadow_expr<devT, T, ndim>(device, output_shape, eval_spec);
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
    static inline float wrap(const float& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         const lazy::EvaluationSpec<devT, T, ndim>&) {
        return scalar;
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,double> {
    template<int ndim>
    static inline double wrap(const double& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         const lazy::EvaluationSpec<devT, T, ndim>&) {
        return scalar;
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,int> {
    template<int ndim>
    static inline int wrap(const int& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         const lazy::EvaluationSpec<devT, T, ndim>&) {
        return scalar;
    }
};

////////////////////////////////////////////////////////////////////////////////
//                             LAZY_FUNCTION                                  //
////////////////////////////////////////////////////////////////////////////////


template<typename Class, typename... Args>
struct LazyFunction: public LazyExp<Class> {
    static const int  evaluation_dim;
    static const bool collapse_leading;
    const std::vector<int> bshape_;
    const DType dtype_;

    LazyFunction(Args... args);

    static std::vector<int> lazy_output_bshape(const Args&... args);

    static DType lazy_inputs_dtype(const Args&... args);

    const std::vector<int>& bshape() const;
    const DType& dtype() const;
};


namespace functor_helper {
    template<typename ExpT> struct ExtractDType {typedef typename ExpT::exp_dtype_t value;};
    template<>              struct ExtractDType<float> {typedef float value;};
    template<>              struct ExtractDType<int> {typedef int value;};
    template<>              struct ExtractDType<double> {typedef double value;};

    template<typename T1, typename T2>
    struct BinaryExtractDType {
        typedef typename ExtractDType<T1>::value left_t;
        typedef typename ExtractDType<T2>::value right_t;
        typedef decltype(left_t(0) * right_t(0)) value;
    };

    template<typename T>
    struct UnaryExtractDType {
        typedef typename ExtractDType<T>::value value;
    };
}  // namespace functor_helper

#include "dali/array/function/lazy_function-impl.h"

#endif
