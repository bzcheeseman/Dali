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
//                          PREPARE OUTPUT                                    //
////////////////////////////////////////////////////////////////////////////////


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
struct CommonPropertyExtractor {
    typedef typename ArrayProperty::property_t outtype_t;
    typedef bool state_t;

    template<typename T>
    static std::tuple<outtype_t, state_t> reduce(const std::tuple<outtype_t, state_t>& candidate_and_state, T elem) {
        return candidate_and_state;
    }

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
//                               FIND COMMON DEVICE                           //
////////////////////////////////////////////////////////////////////////////////

struct DeviceReducerState {
    int args_read;
    memory::Device common_preferred_device;
};

struct DeviceReducer {
    // Finds best device to run computation on
    // based on the availability, freshness, preference, and position
    // of the Array arguments in a function call.
    typedef memory::Device outtype_t;
    typedef DeviceReducerState state_t;

    template<typename T>
    static std::tuple<outtype_t, state_t> reduce(const std::tuple<outtype_t, state_t>& candidate_and_state, T elem) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce(const std::tuple<outtype_t, state_t>& candidate_and_state, const Array& arg) {
        auto state = std::get<1>(candidate_and_state);

        // When state args_read <= 0, then reduction is in its first Array argument
        // while other non-Array arguments have been ignored by ReduceOverArgs<>::reduce_helper
        // [Note: output is also an Array argument]
        if (state.args_read <= 0) {
            // *** When considering the first Array ***
            auto mem = arg.memory();
            // If there's only 1 Array involved, we can safely consider
            // this Array's memory's preferred_device as a good option
            memory::Device best_device_for_me_myself_and_i = mem->preferred_device;
            // One caveat, we want preferred_device's memory to be fresh
            bool is_best_option_fresh = mem->is_fresh(mem->preferred_device);
            // Also we want to know whether any copy of memory is fresh
            bool is_some_other_option_fresh = mem->is_any_fresh();
            // if the preferred memory is not fresh, and there is
            // a fresh alternative use it:
            if (!is_best_option_fresh && is_some_other_option_fresh) {
                best_device_for_me_myself_and_i = mem->find_some_fresh_device();
            }// else, make the preferred device fresh
            return std::make_tuple(best_device_for_me_myself_and_i, DeviceReducerState{state.args_read + 1, mem->preferred_device});
        } else {

            if (arg.memory()->preferred_device != state.common_preferred_device) {
                // When considering other arguments, if the next argument prefers a different device,
                // then we fallback to the tie-breaker device
                return std::make_tuple(default_preferred_device, DeviceReducerState{state.args_read + 1, memory::Device::device_of_doom()});
            } else {
                // we can place the computation on the currently agreed device
                return std::make_tuple(arg.memory()->preferred_device, DeviceReducerState{state.args_read + 1, arg.memory()->preferred_device});
            }
        }
    }
};

template<typename T>
memory::Device extract_device(T sth) {
    return memory::Device::device_of_doom();
}

memory::Device extract_device(const Array& a);

struct MaybeDType {
    DType dtype;
    bool is_present;
};


////////////////////////////////////////////////////////////////////////////////
//                FUNCTION AND ITS SPECIALIZATIONS                            //
////////////////////////////////////////////////////////////////////////////////

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
        auto device = ReduceOverArgs<DeviceReducer>::reduce(out, args...);
        auto dtype  = ReduceOverArgs<CommonPropertyExtractor<DTypeProperty>>::reduce(out, args...);

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
