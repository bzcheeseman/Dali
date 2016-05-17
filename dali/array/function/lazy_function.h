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


template<int devT,typename T, typename ExprT>
struct MshadowWrapper {
    static inline auto wrap(const ExprT& sth, memory::Device device, const std::vector<int>& output_shape) ->
            decltype(sth.template to_mshadow_expr<devT,T>(device, output_shape)) {
        return sth.template to_mshadow_expr<devT,T>(device, output_shape);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,Array> {
    static inline auto wrap(const Array& array, memory::Device device, const std::vector<int>& output_shape) ->
            decltype(TypedArray<devT,T>(array, device, output_shape).d2()) {
        return TypedArray<devT,T>(array, device, output_shape).d2();
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,float> {
    static inline T wrap(const float& scalar, memory::Device device, const std::vector<int>& output_shape) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,double> {
    static inline T wrap(const double& scalar, memory::Device device, const std::vector<int>& output_shape) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,int> {
    static inline T wrap(const int& scalar, memory::Device device, const std::vector<int>& output_shape) { return (T)scalar; }
};


////////////////////////////////////////////////////////////////////////////////
//                             LAZY_FUNCTION                                  //
////////////////////////////////////////////////////////////////////////////////


template<typename Class, typename... Args>
struct BaseLazyFunction: public LazyExp<Class> {
    static const int evaluation_dim;
    std::vector<int> bshape_;
    DType dtype_;

    BaseLazyFunction(Args... args);

    static std::vector<int> lazy_output_bshape(const Args&... args);

    static DType lazy_output_dtype(const Args&... args);

    const std::vector<int>& bshape() const;

    const DType& dtype() const;
};

template<typename Class, typename... Args>
struct LazyFunction : public BaseLazyFunction<Class, Args...> {
    using BaseLazyFunction<Class, Args...>::BaseLazyFunction;
};

template<typename Class, typename... Args>
struct LazyFunctionNonRecusive : public BaseLazyFunction<Class, Args...> {
    using BaseLazyFunction<Class, Args...>::BaseLazyFunction;
};


#include "dali/array/function/lazy_function-impl.h"

#endif
