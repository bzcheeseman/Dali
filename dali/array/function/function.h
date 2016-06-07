#ifndef DALI_ARRAY_FUNCTION_FUNCTION_H
#define DALI_ARRAY_FUNCTION_FUNCTION_H

#include "dali/config.h"

#include <cstdarg>
#include <string>
#include <mshadow/tensor.h>
#include <tuple>

#define DALI_ARRAY_HIDE_LAZY 1
#include "dali/array/array.h"
#undef DALI_ARRAY_HIDE_LAZY
#include "dali/array/debug.h"
#include "dali/array/dtype.h"
#include "dali/array/function/args/reduce_over_args.h"
#include "dali/array/function/args/property_reducer.h"
#include "dali/array/function/typed_array.h"
#include "dali/array/function/operator.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/unpack_tuple.h"


////////////////////////////////////////////////////////////////////////////////
//                FUNCTION AND ITS SPECIALIZATIONS                            //
////////////////////////////////////////////////////////////////////////////////

template<int devT, typename T>
struct ArrayWrapper {
    template<typename X>
    static inline X wrap(const X& sth, memory::Device dev) {
        return sth;
    }

    static inline TypedArray<devT,T> wrap(const Array& a, memory::Device dev) {
        return TypedArray<devT,T>(a, dev, a.shape());
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

    static std::vector<int> deduce_output_bshape(const Args&... args) {
        return ReduceOverArgs<BShapeCompatibleForAllArrayArgsReducer>::reduce(args...);
    }

    static DType deduce_output_dtype(const Args&... args) {
        return ReduceOverArgs<DTypeEqualForAllArrayArgsReducer>::reduce(args...);
    }

    static memory::Device deduce_output_device(const Args&... args) {
        return ReduceOverArgs<DeviceReducer>::reduce(args...);
    }

    static memory::Device deduce_computation_device(const Outtype& out, const Args&... args) {
        return ReduceOverArgs<DeviceReducer>::reduce(out, args...);
    }

    static DType deduce_computation_dtype(const Outtype& out, const Args&... args) {
        return ReduceOverArgs<DTypeEqualForAllArrayArgsReducer>::reduce(out, args...);
    }

    static void initialize_output_array(Outtype& out,
                                        const DType& output_dtype,
                                        const memory::Device& output_device,
                                        std::vector<int>* output_bshape_ptr) {
        auto& output_bshape = *output_bshape_ptr;
        if (out.is_stateless()) {
            // when constructing a stateless
            // output, we decide what the output
            // shape will be. Broadcasted greater
            // than one dimensions are expanded
            // out:
            for (auto& dim : output_bshape) {
                if (dim < -1) {
                    dim = std::abs(dim);
                }
            }

            out.initialize_with_bshape(output_bshape,
                                       output_dtype,
                                       output_device);
        } else {
            if (!Class::disable_output_shape_check) {
                bool broadcast_reshaped_output = false;

                for (const int& dim_size: out.bshape()) {
                    if (dim_size < -1) {
                        broadcast_reshaped_output = true;
                        break;
                    }
                }

                ASSERT2(!broadcast_reshaped_output,
                        "Cannot assign to broadcasted output with broadcasted dimension"
                        " bigger than 1, because it results in many-to-one mappings.");


                bool output_bshape_compatible = out.ndim() == output_bshape.size();
                if (output_bshape_compatible) {
                    for (int i = 0; i < out.ndim(); ++i) {
                        if (output_bshape[i] != -1 && abs(output_bshape[i]) != out.shape()[i]) {
                            output_bshape_compatible = false;
                            break;
                        }
                    }
                }

                ASSERT2(output_bshape_compatible,
                        utils::MS() << "Cannot assign result of shape " << output_bshape << " to a location of shape " << out.shape() << ".");
            }
            ASSERT2(Class::disable_output_dtype_check || out.dtype() == output_dtype,
                    utils::MS() << "Cannot assign result of dtype " << dtype_to_name(output_dtype) << " to a location of dtype " << dtype_to_name(out.dtype()) << ".");
        }
    }

    static std::tuple<Outtype_t, Args...> prepare_output(const OPERATOR_T& operator_t, Outtype& out, const Args&... args) {
        auto output_bshape = Class::deduce_output_bshape(args...);
        auto output_dtype  = Class::deduce_output_dtype(args...);
        auto output_device = Class::deduce_output_device(args...);

        Class::initialize_output_array(out, output_dtype, output_device, &output_bshape);

        Class::verify(args...);

        return std::tuple<Outtype, Args...>(out, args...);
    }

    static void verify(const Args&... args) {
        //TODO(yupeng): remove this once all the functions respect it.
    }

    template<OPERATOR_T intented_operator_t>
    static AssignableArray run_with_operator(const Args&... args) {
        return AssignableArray([args...](Outtype& out, const OPERATOR_T& operator_t) {
            ASSERT2(operator_t == intented_operator_t,
                utils::MS() << "AssignableArray constructed for operator "
                            << operator_to_name(intented_operator_t)
                            << " but got " << operator_to_name(operator_t)
                            << " instead");

            auto prepped_args = Class::prepare_output(operator_t, out, args...);
            unpack_tuple(Class::template untyped_eval<intented_operator_t>, prepped_args);
            debug::dali_function_computed.activate(true);
        });
    }

    template<OPERATOR_T operator_t>
    static void untyped_eval(const Outtype& out,
                             const Args&... args) {
        auto device = Class::deduce_computation_device(out, args...);
        auto dtype  = Class::deduce_computation_dtype(out, args...);

        if (device.type() == memory::DEVICE_T_CPU && dtype == DTYPE_FLOAT) {
            Class().template compute<operator_t,memory::DEVICE_T_CPU,float>(out, device, args...);
        } else if (device.type() == memory::DEVICE_T_CPU && dtype == DTYPE_DOUBLE) {
            Class().template compute<operator_t,memory::DEVICE_T_CPU,double>(out, device, args...);
        } else if (device.type() == memory::DEVICE_T_CPU && dtype == DTYPE_INT32) {
            Class().template compute<operator_t,memory::DEVICE_T_CPU,int>(out, device, args...);
        }
#ifdef DALI_USE_CUDA
        else if (device.type() == memory::DEVICE_T_GPU && dtype == DTYPE_FLOAT) {
            Class().template compute<operator_t,memory::DEVICE_T_GPU,float>(out, device, args...);
        } else if (device.type() == memory::DEVICE_T_GPU && dtype == DTYPE_DOUBLE) {
            Class().template compute<operator_t,memory::DEVICE_T_GPU,double>(out, device, args...);
        } else if (device.type() == memory::DEVICE_T_GPU && dtype == DTYPE_INT32) {
            Class().template compute<operator_t,memory::DEVICE_T_GPU,int>(out, device, args...);
        }
#endif
        else {
            ASSERT2(false, utils::MS() << "Best device must be either cpu or gpu, and dtype must be in " DALI_ACCEPTABLE_DTYPE_STR << " (got device: " << device.description() << ", dtype: " << dtype_to_name(dtype) <<  ")");
        }
    }

    static AssignableArray run(const Args&... args) {
        return AssignableArray([args...](Outtype& out, const OPERATOR_T& operator_t) {
            auto prepped_args = Class::prepare_output(operator_t, out, args...);
            switch (operator_t) {
                case OPERATOR_T_EQL:
                    unpack_tuple(Class::template untyped_eval<OPERATOR_T_EQL>, prepped_args);
                    break;
                case OPERATOR_T_ADD:
                    unpack_tuple(Class::template untyped_eval<OPERATOR_T_ADD>, prepped_args);
                    break;
                case OPERATOR_T_SUB:
                    unpack_tuple(Class::template untyped_eval<OPERATOR_T_SUB>, prepped_args);
                    break;
                case OPERATOR_T_MUL:
                    unpack_tuple(Class::template untyped_eval<OPERATOR_T_MUL>, prepped_args);
                    break;
                case OPERATOR_T_DIV:
                    unpack_tuple(Class::template untyped_eval<OPERATOR_T_DIV>, prepped_args);
                    break;
                case OPERATOR_T_LSE:
                    unpack_tuple(Class::template untyped_eval<OPERATOR_T_LSE>, prepped_args);
                    break;
                default:
                    ASSERT2(false, "OPERATOR_T for assignment between AssignableArray and output must be one of =,-=,+=,*=,/=,<<=");
                    break;
            }
            debug::dali_function_computed.activate(true);
        });
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void compute(const Outtype& out, const memory::Device& device, const Args&... args) {
        typedef ArrayWrapper<devT,T> wrapper_t;
        ((Class*)this)->template typed_eval<operator_t>(
            wrapper_t::wrap(out, device),
            wrapper_t::wrap(args, device)...
        );
    }
};

template<template<class> class Functor>
struct Elementwise : public Function<Elementwise<Functor>, Array, Array> {
    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(const TypedArray<devT, T>& out, const TypedArray<devT,T>& input) {
        operator_assign<operator_t, 1>(out, mshadow::expr::F<Functor<T>>(input.d1()));
    }
};

template<template<class> class Functor>
struct BinaryElementwise : public Function<BinaryElementwise<Functor>, Array, Array, Array> {
    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(const TypedArray<devT, T>& out, const TypedArray<devT,T>& left, const TypedArray<devT,T>& right) {
        operator_assign<operator_t, 1>(out, mshadow::expr::F<Functor<T>>(left.d1(), right.d1()));
    }
};

template<typename Class, typename Outtype, typename... Args>
struct NonArrayFunction : public Function<Class,Outtype*,Args...> {

    static std::tuple<Outtype, Args...> prepare_output(const OPERATOR_T& operator_t, Outtype& out, const Args&... args) {
        return std::tuple<Outtype, Args...>(out, args...);
    }

    static Outtype run(const Args&... args) {
        typedef Function<Class,Outtype*,Args...> func_t;
        Outtype out;
        func_t::template untyped_eval <OPERATOR_T_EQL>( &out, args... );
        return out;
    }
};


// special macro that allows function structs to
// dynamically catch/fail unsupported cases
#define FAIL_ON_OTHER_CASES(OP_NAME)     Outtype_t operator()(...) { \
    throw std::string("ERROR: Unsupported types/devices for OP_NAME"); \
}

#endif
