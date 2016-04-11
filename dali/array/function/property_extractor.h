#ifndef DALI_ARRAY_FUNCTION_PROPERTY_EXTRACTOR_H
#define DALI_ARRAY_FUNCTION_PROPERTY_EXTRACTOR_H

#include <string>
#include <tuple>
#include <vector>

#include "dali/array/dtype.h"
#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"

class Array;

template<typename ArrayProperty>
struct CommonPropertyExtractor {
    typedef typename ArrayProperty::property_t outtype_t;
    typedef bool state_t;

    template<typename T>
    static std::tuple<outtype_t, state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, T elem) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce_step(const std::tuple<outtype_t, state_t>& candidate_and_state, const Array& arg) {
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


struct ShapeDeducer {
    template<typename T>
    static std::tuple<bool,std::vector<int>> deduce_unary(const T& x) {
        static_assert(!std::is_same<T,uint>::value, "blah");
        return std::make_tuple(true, x.shape());
    }
    static std::tuple<bool,std::vector<int>> deduce_unary(const float& x) {
        return std::make_tuple(false, std::vector<int>());
    }
    static std::tuple<bool,std::vector<int>> deduce_unary(const double& x) {
        return std::make_tuple(false, std::vector<int>());
    }
    static std::tuple<bool,std::vector<int>> deduce_unary(const int& x) {
        return std::make_tuple(false, std::vector<int>());
    }

    template<typename T, typename T2>
    static std::vector<int> deduce_binary(const T& left, const T2& right) {
        bool left_matters, right_matters;
        std::vector<int> left_shape, right_shape;
        std::tie(left_matters, left_shape)   = deduce_unary(left);
        std::tie(right_matters, right_shape) = deduce_unary(right);

        ASSERT2(left_matters || right_matters,
                "deduce_binary called with two shapeless entities.");

        if (left_matters) {
            if (right_matters) {
                ASSERT2(left_shape == right_shape,
                    utils::MS() << "Expressions of inconsistent shape passed to binary expression (" << left_shape << " VS " << right_shape <<  ")");
            }
            return left_shape;
        } else if (right_matters) {
            return right_shape;
        }
        return std::vector<int>();
    }

};

struct DtypeDeducer {
    template<typename T>
    static std::tuple<bool, DType> deduce_unary(const T& x) {
        return std::make_tuple(true, x.dtype());
    }
    static std::tuple<bool,DType> deduce_unary(const float& x) {
        return std::make_tuple(false, DTYPE_FLOAT);
    }
    static std::tuple<bool,DType> deduce_unary(const double& x) {
        return std::make_tuple(false, DTYPE_FLOAT);
    }
    static std::tuple<bool,DType> deduce_unary(const int& x) {
        return std::make_tuple(false, DTYPE_FLOAT);
    }

    template<typename T, typename T2>
    static DType deduce_binary(const T& left, const T2& right) {
        bool left_matters, right_matters;
        DType left_dtype, right_dtype;
        std::tie(left_matters, left_dtype)   = deduce_unary(left);
        std::tie(right_matters, right_dtype) = deduce_unary(right);

        ASSERT2(left_matters || right_matters,
                "deduce_binary called with two dtypeless entities.");

        if (left_matters) {
            if (right_matters) {
                ASSERT2(left_dtype == right_dtype,
                    utils::MS() << "Expressions of inconsistent dtype passed to binary expression (" << left_dtype << " VS " << right_dtype <<  ")");
            }
            return left_dtype;
        } else if (right_matters) {
            return right_dtype;
        }
        return DTYPE_FLOAT;
    }

};



struct ShapeProperty {
    typedef std::vector<int> property_t;
    static std::string name;
    static std::vector<int> extract(const Array& x);
};

struct DTypeProperty {
    typedef DType property_t;
    static std::string name;
    static DType extract(const Array& x);
};

#endif
