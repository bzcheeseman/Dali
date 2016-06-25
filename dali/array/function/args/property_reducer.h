#ifndef DALI_ARRAY_FUNCTION_ARGS_PROPERTY_REDUCER_H
#define DALI_ARRAY_FUNCTION_ARGS_PROPERTY_REDUCER_H

#include <string>
#include <tuple>
#include <vector>

#include "dali/array/dtype.h"
#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/array/memory/device.h"

class Array;
class ArraySubtensor;
class ArrayGather;

template<typename Property>
struct PropertyEqualForAllArgsReducer {
    typedef typename Property::property_t outtype_t;
    typedef bool state_t;

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const float& arg) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const double& arg) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const int& arg) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const std::vector<int>& arg) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const bool& arg) {
        return candidate_and_state;
    }

    template<typename T>
    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const T& arg) {
        outtype_t candidate;
        bool ready;
        auto arg_property = Property::extract(arg);
        std::tie(candidate, ready) = candidate_and_state;
        if (ready) {
            ASSERT2(candidate == arg_property, utils::MS() << "All arguments should be of the same " << Property::name << " (MISMATCH between "
                                                          << Property::pretty_name(candidate) << " and " << Property::pretty_name(arg_property) << ")");
            return candidate_and_state;
        } else {
            return std::make_tuple(arg_property, true);
        }
    }
};


// for extracting properties like shape or dtype
template<typename Property>
struct PropertyEqualForAllArrayArgsReducer {
    typedef typename Property::property_t outtype_t;
    typedef bool state_t;

    template<typename T>
    static std::tuple<outtype_t, state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, T elem) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const Array& arg) {
        outtype_t candidate;
        bool ready;
        auto arg_property = Property::extract(arg);
        std::tie(candidate, ready) = candidate_and_state;
        if (ready) {
            ASSERT2(candidate == arg_property, utils::MS() << "All arguments should be of the same " << Property::name << " (MISMATCH between "
                                                          << Property::pretty_name(candidate) << " and " << Property::pretty_name(arg_property) << ")");
            return candidate_and_state;
        } else {
            return std::make_tuple(arg_property, true);
        }
    }
};



struct BShapeCompatibleForAllArgsReducer {
    typedef std::vector<int> outtype_t;
    typedef bool        state_t;

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const float& arg) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const double& arg) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const int& arg) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const bool& arg) {
        return candidate_and_state;
    }

    template<typename T>
    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const T& arg) {
        outtype_t candidate;
        bool ready;
        std::tie(candidate, ready) = candidate_and_state;
        if (ready) {
            const auto& new_shape = arg.bshape();
            // candidate
            ASSERT2(candidate.size() == new_shape.size(),
                    utils::MS() << "All arguments must be of the same dimensionality" <<
                    " (MISMATCH between ndims="  << candidate.size() << " and ndims=" <<
                    new_shape.size() << ")");
            std::vector<int> combined_shape(candidate.size(), 0);
            for (int i = 0; i < candidate.size(); ++i) {
                if (candidate[i] < 0) {
                    combined_shape[i] = new_shape[i];
                } else if (new_shape[i] < 0) {
                    combined_shape[i] = candidate[i];
                } else {
                    ASSERT2(new_shape[i] == candidate[i],
                            utils::MS() << "Incompatible shape at dimension " << i << ": " << candidate << " VS " << new_shape << ".");
                    combined_shape[i] = new_shape[i];
                }
            }

            return std::make_tuple(combined_shape, true);
        } else {
            return std::make_tuple(arg.bshape(), true);
        }
    }
};

struct BShapeCompatibleForAllArrayArgsReducer {
    typedef std::vector<int> outtype_t;
    typedef bool        state_t;

    template<typename T>
    static std::tuple<outtype_t, state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, T elem) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const Array& arg);
};

struct ShapeProperty {
    typedef std::vector<int> property_t;
    static std::string name;

    template<typename T>
    static property_t extract(const T& x) {
        return x.shape();
    }

    static property_t pretty_name(const property_t&);
};

struct DTypeProperty {
    typedef DType property_t;
    static std::string name;

    template<typename T>
    static property_t extract(const T& x) {
        return x.dtype();
    }

    static std::string pretty_name(const property_t&);
};

typedef PropertyEqualForAllArgsReducer<DTypeProperty> DTypeEqualForAllArgsReducer;

typedef PropertyEqualForAllArrayArgsReducer<DTypeProperty> DTypeEqualForAllArrayArgsReducer;

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
    static std::tuple<outtype_t, state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const T& elem) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const Array& arg);
    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const ArraySubtensor& arg);
    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const ArrayGather& arg);
    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const std::vector<Array>& arg);
};

#endif
