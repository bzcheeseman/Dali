#ifndef DALI_ARRAY_FUNCTION_PROPERTY_EXTRACTOR_H
#define DALI_ARRAY_FUNCTION_PROPERTY_EXTRACTOR_H

#include <string>
#include <tuple>
#include <vector>

#include "dali/array/dtype.h"
#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"

class Array;

template<typename Property>
struct CommonPropertyExtractor {
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
                                                          << candidate << " and " << arg_property << ")");
            return candidate_and_state;
        } else {
            return std::make_tuple(arg_property, true);
        }
    }
};


template<typename Property>
struct LazyCommonPropertyExtractor {
    typedef typename Property::property_t property_t;

    template<typename T>
    static std::tuple<bool,property_t> extract_unary(const T& x) {
        return std::make_tuple(true, Property::extract(x));
    }
    static std::tuple<bool,property_t> extract_unary(const float& x) {
        return std::make_tuple(false, property_t());
    }
    static std::tuple<bool,property_t> extract_unary(const double& x) {
        return std::make_tuple(false, property_t());
    }
    static std::tuple<bool,property_t> extract_unary(const int& x) {
        return std::make_tuple(false, property_t());
    }

    template<typename T, typename T2>
    static property_t extract_binary(const T& left, const T2& right) {
        bool left_matters, right_matters;
        property_t left_property, right_property;
        std::tie(left_matters, left_property)   = extract_unary(left);
        std::tie(right_matters, right_property) = extract_unary(right);

        ASSERT2(left_matters || right_matters,
                utils::MS() << "deduce_binary called with two " << Property::name << "less entities.");

        if (left_matters) {
            if (right_matters) {
                ASSERT2(left_property == right_property,
                    utils::MS() << "Expressions of inconsistent " << Property::name << " passed to binary expression (" << left_property << " VS " << right_property <<  ")");
            }
            return left_property;
        } else if (right_matters) {
            return right_property;
        }
        return property_t();
    }

};

struct ShapeProperty {
    typedef std::vector<int> property_t;
    static std::string name;

    template<typename T>
    static property_t extract(const T& x) {
        return x.shape();
    }
};

struct DTypeProperty {
    typedef DType property_t;
    static std::string name;

    template<typename T>
    static property_t extract(const T& x) {
        return x.dtype();
    }
};

#endif
