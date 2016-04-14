#ifndef DALI_ARRAY_FUNCTION_FUNCTION_H
#define DALI_ARRAY_FUNCTION_FUNCTION_H

#include "dali/config.h"

#include <cstdarg>
#include <string>
#include <mshadow/tensor.h>
#include <tuple>

#include "dali/array/dtype.h"
#include "dali/array/array.h"
#include "dali/array/getmshadow.h"
#include "dali/utils/print_utils.h"
#include "dali/array/function/reducer.h"
#include "dali/array/function/property_extractor.h"



////////////////////////////////////////////////////////////////////////////////
//                FUNCTION AND ITS SPECIALIZATIONS                            //
////////////////////////////////////////////////////////////////////////////////

template<int devT, typename T>
struct ArrayWrapper {
    template<typename X>
    static X wrap(const X& sth, memory::Device dev) {
        return sth;
    }

    static MArray<devT,T> wrap(const Array& a, memory::Device dev) {
        return MArray<devT,T>(a,dev);
    }
};


template<typename Class, typename Outtype, typename... Args>
struct Function {
    typedef Outtype Outtype_t;
    // In the future this struct will implement all of the beautiful
    // logic that allows dali to spread its wings across clusters
    // and rampage around numbers into the ether. Soon.
    //
    // static RpcRequest bundle_for_remote_execution(Args... args) {
    //     // create args bundle, that can be transfered over the interwebz.
    //     auto bundle = combine_bundled_args(bundle_arg(args)...);
    //     // the cool part. child class defines FUNCTION_ID.
    //     return RpcRequest(Class::FUNCTION_ID, bundle);
    // }

    static const bool disable_output_shape_check = false;
    static const bool disable_output_dtype_check = false;

    static std::vector<int> deduce_output_shape(const Args&... args) {
        return ReduceOverArgs<CommonPropertyExtractor<ShapeProperty>>::reduce(args...);
    }

    static DType deduce_output_dtype(const Args&... args) {
        return ReduceOverArgs<CommonPropertyExtractor<DTypeProperty>>::reduce(args...);
    }

    static memory::Device deduce_output_device(const Args&... args) {
        return ReduceOverArgs<DeviceReducer>::reduce(args...);
    }

    static memory::Device deduce_computation_device(const Outtype& out, const Args&... args) {
        return ReduceOverArgs<DeviceReducer>::reduce(out, args...);
    }

    static DType deduce_computation_dtype(const Outtype& out, const Args&... args) {
        return ReduceOverArgs<CommonPropertyExtractor<DTypeProperty>>::reduce(out, args...);
    }

    static void prepare_output(Outtype& out, const Args&... args) {
        auto common_shape = Class::deduce_output_shape(args...);
        auto common_dtype = Class::deduce_output_dtype(args...);

        if (out.is_stateless()) {
            out.initialize(common_shape, common_dtype, Class::deduce_output_device(args...));
        } else {
            ASSERT2(Class::disable_output_shape_check || out.shape() == common_shape,
                    utils::MS() << "Cannot assign result of shape " << common_shape << " to a location of shape " << out.shape() << ".");
            ASSERT2(Class::disable_output_dtype_check || out.dtype() == common_dtype,
                    utils::MS() << "Cannot assign result of dtype " << common_dtype << " to a location of dtype " << out.dtype() << ".");
        }
    }

    static AssignableArray run(const Args&... args) {
        return AssignableArray([args...](Outtype& out) {
            prepare_output(out, args...);
            untyped_eval(out, args...);
        });
    }

    static void untyped_eval(const Outtype& out, const Args&... args) {
        auto device = Class::deduce_computation_device(out, args...);
        auto dtype  = Class::deduce_computation_dtype(out, args...);
        if (device.type() == memory::DEVICE_T_CPU && dtype == DTYPE_FLOAT) {
            typedef ArrayWrapper<memory::DEVICE_T_CPU,float> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        } else if (device.type() == memory::DEVICE_T_CPU && dtype == DTYPE_DOUBLE) {
            typedef ArrayWrapper<memory::DEVICE_T_CPU,double> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        } else if (device.type() == memory::DEVICE_T_CPU && dtype == DTYPE_INT32) {
            typedef ArrayWrapper<memory::DEVICE_T_CPU,int> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        }
#ifdef DALI_USE_CUDA
        else if (device.type() == memory::DEVICE_T_GPU && dtype == DTYPE_FLOAT) {
            typedef ArrayWrapper<memory::DEVICE_T_GPU,float> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        } else if (device.type() == memory::DEVICE_T_GPU && dtype == DTYPE_DOUBLE) {
            typedef ArrayWrapper<memory::DEVICE_T_GPU,double> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        } else if (device.type() == memory::DEVICE_T_GPU && dtype == DTYPE_INT32) {
            typedef ArrayWrapper<memory::DEVICE_T_GPU,int> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        }
#endif
        else {
            ASSERT2(false, utils::MS() << "Best device must be either cpu or gpu, and dtype must be in " DALI_ACCEPTABLE_DTYPE_STR << " (got device: " << device.description() << ", dtype: " << dtype_to_name(dtype) <<  ")");
        }
    }
};

template<template<class> class Functor>
struct Elementwise : public Function<Elementwise<Functor>, Array, Array> {
    template<int devT, typename T>
    void typed_eval(const MArray<devT, T>& out, const MArray<devT,T>& input) {
        out.d1(memory::AM_OVERWRITE) = mshadow::expr::F<Functor<T>>(input.d1());
    }
};

template<template<class> class Functor>
struct BinaryElementwise : public Function<BinaryElementwise<Functor>, Array, Array, Array> {
    template<int devT, typename T>
    void typed_eval(const MArray<devT, T>& out, const MArray<devT,T>& left, const MArray<devT,T>& right) {
        out.d1(memory::AM_OVERWRITE) = mshadow::expr::F<Functor<T>>(left.d1(), right.d1());
    }
};

template<typename Class, typename Outtype, typename... Args>
struct NonArrayFunction : public Function<Class,Outtype*,Args...> {

    static void prepare_output(Outtype& out, const Args&... args) {
    }

    static Outtype run(const Args&... args) {
        Outtype out;
        Function<Class,Outtype*,Args...>::untyped_eval(&out, args...);
        return out;
    }
};


// special macro that allows function structs to
// dynamically catch/fail unsupported cases
#define FAIL_ON_OTHER_CASES(OP_NAME)     Outtype_t operator()(...) { \
    throw std::string("ERROR: Unsupported types/devices for OP_NAME"); \
}

#endif
