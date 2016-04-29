#ifndef DALI_ARRAY_FUNCTION_ARGS_MSHADOW_WRAPPER_H
#define DALI_ARRAY_FUNCTION_ARGS_MSHADOW_WRAPPER_H

#include "dali/array/function/typed_array.h"
#include "dali/array/memory/device.h"

// This file would not be needed at all if we defined to_mshadow_expr function on Array.
//
// The reason not to do that is to hide all mshadow usage in cpp files whereever possible.

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
            decltype(TypedArray<devT,T>(array, device).d1()) {
        return TypedArray<devT,T>(array, device).d1();
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
