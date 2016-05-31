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

template<typename WrappedArrayT>
using ArrayTransformerT = std::function<WrappedArrayT(const Array&,
                                                      const memory::Device&,
                                                      const std::vector<int>&)>;

template<int devT, typename T, int ndim>
auto transform_array(const Array& array,
                     const memory::Device& device,
                     const std::vector<int>& output_shape) ->
        decltype(TypedArray<devT,T>(array, device, output_shape).template d<ndim>()) {
    return TypedArray<devT,T>(array, device, output_shape).template d<ndim>();
};

template<int devT, typename T, int ndim>
auto make_transform_array() -> ArrayTransformerT<decltype(TypedArray<devT,T>(Array(), memory::Device::cpu(), std::vector<int>()).template d<ndim>())> {
    return &transform_array<devT,T,ndim>;
}

template<int devT, typename T, int ndim>
auto transform_array_collapse_trailing(const Array& array,
                     const memory::Device& device,
                     const std::vector<int>& output_shape) ->
        decltype(TypedArray<devT,T>(array, device, output_shape).template d<ndim>()) {
    return TypedArray<devT,T>(array, device, output_shape).template d<ndim>(memory::AM_READONLY, false);
};


template<int devT, typename T, int ndim>
auto make_transform_array_collapse_trailing() -> ArrayTransformerT<decltype(TypedArray<devT,T>(Array(), memory::Device::cpu(), std::vector<int>()).template d<ndim>())> {
    return &transform_array_collapse_trailing<devT,T,ndim>;
}


template<int devT, typename T, typename ExprT>
struct MshadowWrapper {
    template<typename WrappedArrayT>
    static inline auto wrap(const ExprT& sth,
                            memory::Device device,
                            const std::vector<int>& output_shape,
                            ArrayTransformerT<WrappedArrayT> wrap_array) ->
            decltype(sth.template to_mshadow_expr<devT,T>(device, output_shape, wrap_array)) {
        return sth.template to_mshadow_expr<devT,T>(device, output_shape, wrap_array);
    }

};

template<int devT,typename T>
struct MshadowWrapper<devT,T,Array> {
    template<typename WrappedArrayT>
    static inline WrappedArrayT wrap(const Array& array,
                                  memory::Device device,
                                  const std::vector<int>& output_shape,
                                  ArrayTransformerT<WrappedArrayT> wrap_array) {
        return wrap_array(array, device, output_shape);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,float> {
    template<typename WrappedArrayT>
    static inline T wrap(const float& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         ArrayTransformerT<WrappedArrayT>) {
        return (T)scalar;
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,double> {
    template<typename WrappedArrayT>
    static inline T wrap(const double& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         ArrayTransformerT<WrappedArrayT>) {
        return (T)scalar;
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,int> {
    template<typename WrappedArrayT>
    static inline T wrap(const int& scalar,
                         memory::Device,
                         const std::vector<int>&,
                         ArrayTransformerT<WrappedArrayT>) {
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
