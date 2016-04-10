#ifndef DALI_ARRAY_ARRAY_FUNCTIONS_H
#define DALI_ARRAY_ARRAY_FUNCTIONS_H

#include "dali/config.h"

#include <cstdarg>
#include <string>
#include <mshadow/tensor.h>
#include <tuple>

#include "dali/array/dtype.h"
#include "dali/array/assignable_array.h"
#include "dali/array/getmshadow.h"
#include "dali/utils/print_utils.h"
#include "dali/runtime_config.h"

////////////////////////////////////////////////////////////////////////////////
//         HELPER FUNCTION FOR EXTRACTING VARIOUS INFO ABOUT ARRAYS           //
////////////////////////////////////////////////////////////////////////////////

template<typename T>
memory::Device extract_device(T sth) {
    return memory::Device::device_of_doom();
}

memory::Device extract_device(const Array& a);

struct MaybeDType {
    DType dtype;
    bool is_present;
};

template<typename T>
MaybeDType extract_dtype(T sth) {
    return MaybeDType{DTYPE_FLOAT, false};
}

MaybeDType extract_dtype(const Array& a);

template<int devT, typename T>
struct ArrayWrapper {
    template<typename X>
    static X wrap(X sth, memory::Device dev) {
        return sth;
    }

    static MArray<devT,T> wrap(const Array& a, memory::Device dev) {
        return MArray<devT,T>(a,dev);
    }
};



////////////////////////////////////////////////////////////////////////////////
//                          PREPAR_OUTPUT                                     //
////////////////////////////////////////////////////////////////////////////////


template<typename Child, typename Outtype, typename State>
struct Reducer {
    template<typename T>
    static std::tuple<Outtype, State> reduce(const std::tuple<Outtype, State>& candidate_and_state, T elem) {
        return candidate_and_state;
    }
};

template<typename Reducer>
struct ReduceOverArgs {
    typedef std::tuple<typename Reducer::outtype_t, typename Reducer::state_t> outtuple_t;

    template<typename... Args>
    static typename Reducer::outtype_t reduce(const Args&... args) {
        auto initial_tuple = outtuple_t();
        return std::get<0>(reduce_helper(initial_tuple, args...));
    }

    template<typename FirstArg, typename... Args>
    static outtuple_t reduce_helper(const outtuple_t& candidate_and_state, const FirstArg& arg, const Args&... args) {
        return reduce_helper(Reducer::reduce(candidate_and_state, arg), args...);
    }

    static outtuple_t reduce_helper(const outtuple_t& candidate_and_state) {
        return candidate_and_state;
    }
};


template<typename ArrayProperty>
struct CommonPropertyExtractor : Reducer<CommonPropertyExtractor<ArrayProperty>,std::vector<int>,bool> {
    typedef typename ArrayProperty::property_t outtype_t;
    typedef bool state_t;

    static std::tuple<outtype_t,state_t> reduce(const std::tuple<outtype_t, state_t>& candidate_and_state, const Array& arg) {
        outtype_t candidate;
        bool ready;
        auto arg_property = ArrayProperty::extract(arg);
        std::tie(candidate, ready) = candidate_and_state;
        if (ready) {
            ASSERT2(candidate == arg_property, utils::MS() << "All arguments should be of the same " << ArrayProperty::name << " (MISMATCH between "
                                                          << candidate << " and " << arg_property << ")");
            return candidate_and_state;
        } else {
            return std::make_tuple(arg_property, true);
        }
    }
};

struct ShapeProperty {
    typedef std::vector<int> property_t;
    static std::string name;
    static std::vector<int> extract(const Array& x) { return x.shape(); }
};

struct DTypeProperty {
    typedef DType property_t;
    static std::string name;
    static DType extract(const Array& x) { return x.dtype(); }
};


template<typename... Args>
void default_prepare_output(Array& out, const Args&... args) {
    auto common_shape = ReduceOverArgs<CommonPropertyExtractor<ShapeProperty>>::reduce(args...);
    auto common_dtype = ReduceOverArgs<CommonPropertyExtractor<DTypeProperty>>::reduce(args...);

    if (out.is_stateless()) {
        out.initialize(common_shape, common_dtype);
    } else {
        ASSERT2(out.shape() == common_shape,
                utils::MS() << "Cannot assign result of shape " << common_shape << " to a location of shape " << out.shape() << ".");
        ASSERT2(out.dtype() == common_dtype,
                utils::MS() << "Cannot assign result of dtype " << common_dtype << " to a location of dtype " << out.dtype() << ".");
    }
}

template<typename Outtype, typename... Args>
void default_prepare_output(Outtype& out, Args... args) {
    // assume all the input arrays are of the same shape and output as well.
}

////////////////////////////////////////////////////////////////////////////////
//                FUNCTION AND ITS SPECIALIZATIONS                            //
////////////////////////////////////////////////////////////////////////////////

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

    // TODO(szymon): fix this
    static memory::Device find_best_device(int num, ...) {
        va_list args;
        va_start(args, num);
        memory::Device common_device = memory::Device::device_of_doom();
        for (int i=0; i<num; ++i) {
            auto device = va_arg(args, memory::Device);
            if (device.type != memory::DEVICE_T_ERROR) {
                common_device = device;
            }
        }
        va_end(args);
        return default_preferred_device;
        ASSERT2(common_device.type != memory::DEVICE_T_ERROR, "Device of doom happened.");
        return common_device;
    }

    static DType find_best_dtype(int num, ...) {
        bool dtype_set = false;
        DType common_dtype;
        va_list args;
        va_start(args, num);
        for (int i=0; i<num; ++i) {
            auto maybe_dtype = va_arg(args, MaybeDType);
            if (maybe_dtype.is_present) {
                ASSERT2(!dtype_set || maybe_dtype.dtype == common_dtype, "Inconsistent dtype passed to Dali Function.");
                common_dtype = maybe_dtype.dtype;
                dtype_set = true;
            }
        }
        va_end(args);
        return common_dtype;
    }

    static void prepare_output(Outtype& out, const Args&... args) {
        default_prepare_output(out, args...);
    }

    static AssignableArray run(const Args&... args) {
        return AssignableArray([args...](Outtype& out) {
            prepare_output(out, args...);
            untyped_eval(out, args...);
        });
    }

    static void untyped_eval(const Outtype& out, const Args&... args) {
        const int size = sizeof...(Args);
        auto device = find_best_device(size, extract_device(out), extract_device(args)...);
        auto dtype  = find_best_dtype(size, extract_dtype(out), extract_dtype(args)...);
        if (device.type == memory::DEVICE_T_CPU && dtype == DTYPE_FLOAT) {
            typedef ArrayWrapper<memory::DEVICE_T_CPU,float> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        } else if (device.type == memory::DEVICE_T_CPU && dtype == DTYPE_DOUBLE) {
            typedef ArrayWrapper<memory::DEVICE_T_CPU,double> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        } else if (device.type == memory::DEVICE_T_CPU && dtype == DTYPE_INT32) {
            typedef ArrayWrapper<memory::DEVICE_T_CPU,int> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        }
#ifdef DALI_USE_CUDA
        else if (device.type == memory::DEVICE_T_GPU && dtype == DTYPE_FLOAT) {
            typedef ArrayWrapper<memory::DEVICE_T_GPU,float> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        } else if (device.type == memory::DEVICE_T_GPU && dtype == DTYPE_DOUBLE) {
            typedef ArrayWrapper<memory::DEVICE_T_GPU,double> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        } else if (device.type == memory::DEVICE_T_GPU && dtype == DTYPE_INT32) {
            typedef ArrayWrapper<memory::DEVICE_T_GPU,int> wrapper_t;
            Class().typed_eval(wrapper_t::wrap(out,device), wrapper_t::wrap(args, device)...);
        }
#endif
        else {
            ASSERT2(false, "Best device must be either cpu or gpu, and dtype must be in " DALI_ACCEPTABLE_DTYPE_STR);
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
