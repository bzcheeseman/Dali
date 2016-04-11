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


template<template<class>class Functor, typename LeftT, typename RightT>
struct Binary {
    typedef Binary<Functor,LeftT,RightT> self_t;
    LeftT  left;
    RightT right;

    Binary(const LeftT& _left, const RightT& _right) :
            left(_left), right(_right) {
    }

    template<int devT, typename T>
    inline auto to_mshadow_expr() -> decltype(
                              mshadow::expr::F<Functor<T>>(
                                   MshadowWrapper<devT,T>::to_expr(left),
                                   MshadowWrapper<devT,T>::to_expr(right)
                              )
                          ) {
        auto left_expr  = MshadowWrapper<devT,T>::to_expr(left);
        auto right_expr = MshadowWrapper<devT,T>::to_expr(right);

        return mshadow::expr::F<Functor<T>>(left_expr, right_expr);

    }

    operator AssignableArray() const {
        return Evaluator<self_t>::run(*this);
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
