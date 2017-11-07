#ifndef DALI_ARRAY_FUNCTION_ARGS_REDUCE_OVER_ARGS_H
#define DALI_ARRAY_FUNCTION_ARGS_REDUCE_OVER_ARGS_H

#include <tuple>

#include "dali/array/memory/device.h"

class Array;

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
        return reduce_helper(Reducer::reduce_step(candidate_and_state, arg), args...);
    }

    static outtuple_t reduce_helper(const outtuple_t& candidate_and_state) {
        return candidate_and_state;
    }
};

#endif
