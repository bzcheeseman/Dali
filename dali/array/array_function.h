#ifndef DALI_ARRAY_ARRAY_FUNCTIONS_H
#define DALI_ARRAY_ARRAY_FUNCTIONS_H

#include <string>
#include <variant.hpp>
#include "dali/array/dtype.h"

template< typename T >
struct always_false {
    enum { value = false };
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

    static Outtype eval(Args... args) {
        // TODO(szymon): return lambda instead of returning
        return mapbox::util::apply_visitor(Class(), args...);
    }
};

// special macro that allows function structs to
// dynamically catch/fail unsupported cases
#define FAIL_ON_OTHER_CASES(OP_NAME)     Outtype_t operator()(...) { \
    throw std::string("ERROR: Unsupported types/devices for OP_NAME"); \
} \

template<typename T>
std::string type_to_name() {
    static_assert(
        always_false<T>::value,
        "type_to_name only works for DALI_ACCEPTABLE_DTYPE_STR"
    );
    return "unknown";
}

template<> std::string type_to_name<int>(){return "int";}
template<> std::string type_to_name<float>(){return "float";}
template<> std::string type_to_name<double>(){return "double";}

#endif
