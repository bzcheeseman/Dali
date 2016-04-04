#ifndef DALI_ARRAY_ARRAY_FUNCTIONS_H
#define DALI_ARRAY_ARRAY_FUNCTIONS_H

#include <cstdarg>
#include <mshadow/tensor.h>
#include <string>
#include <tuple>


#include "dali/array/dtype.h"

////////////////////////////////////////////////////////////////////////////////
//         HELPER FUNCTION FOR EXTRACTING VARIOUS INFO ABOUT ARRAYS           //
////////////////////////////////////////////////////////////////////////////////

template<typename T>
memory::Device extract_device(T sth) {
    return memory::Device::device_of_doom();
}

template<>
memory::Device extract_device(Array a) {
    return a.memory()->preferred_device;
}

struct MaybeDType {
    DType dtype;
    bool is_present;
};

template<typename T>
MaybeDType extract_dtype(T sth) {
    return MaybeDType{DTYPE_FLOAT, false};
}

template<>
MaybeDType extract_dtype(Array a) {
    return MaybeDType{a.dtype(), true};
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

    static Outtype eval(Args... args) {
        const int size = sizeof...(Args);
        auto device = find_best_device(size, extract_device(args)...);
        auto dtype  = find_best_dtype(size, extract_dtype(args)...);

        if (device.type == memory::DEVICE_T_CPU && dtype == DTYPE_FLOAT) {
            return Class().template run<memory::DEVICE_T_CPU,float>(args..., device);
        } else if (device.type == memory::DEVICE_T_CPU && dtype == DTYPE_DOUBLE) {
            return Class().template run<memory::DEVICE_T_CPU,double>(args..., device);
        } else if (device.type == memory::DEVICE_T_CPU && dtype == DTYPE_INT32) {
            return Class().template run<memory::DEVICE_T_CPU,int>(args..., device);
        }
#ifdef DALI_USE_CUDA
        else if (device.type == memory::DEVICE_T_GPU && dtype == DTYPE_FLOAT) {
            return Class().template run<memory::DEVICE_T_GPU,float>(args..., device);
        } else if (device.type == memory::DEVICE_T_GPU && dtype == DTYPE_DOUBLE) {
            return Class().template run<memory::DEVICE_T_GPU,double>(args..., device);
        } else if (device.type == memory::DEVICE_T_GPU && dtype == DTYPE_INT32) {
            return Class().template run<memory::DEVICE_T_GPU,int>(args..., device);
        }
#endif
        ASSERT2(false, "Should not get here.");
        return Outtype();
    }
};


// special macro that allows function structs to
// dynamically catch/fail unsupported cases
#define FAIL_ON_OTHER_CASES(OP_NAME)     Outtype_t operator()(...) { \
    throw std::string("ERROR: Unsupported types/devices for OP_NAME"); \
}



////////////////////////////////////////////////////////////////////////////////
//                    EXTRACTING MSHADOW FROM ARRAYS                          //
////////////////////////////////////////////////////////////////////////////////

template<int devT, typename T>
struct getmshadow {
    memory::Device d;
    void d1(Array a) {}
};

template<typename T>
struct getmshadow<memory::DEVICE_T_CPU, T> {
    memory::Device d;

    mshadow::Tensor<mshadow::cpu, 1, T> d1(Array a) {
        return mshadow::Tensor<mshadow::cpu, 1, T>((T*)(a.memory()->data(d)), mshadow::Shape1(a.number_of_elements()));
    }
};

#ifdef DALI_USE_CUDA
template<typename T>
struct getmshadow<memory::DEVICE_T_GPU, T> {
    memory::Device d;

    mshadow::Tensor<mshadow::gpu, 1, T> d1(Array a) {
        return mshadow::Tensor<mshadow::gpu, 1, T>((T*)(a.memory()->data(d)),  mshadow::Shape1(a.number_of_elements()));
    }
};
#endif



#endif
