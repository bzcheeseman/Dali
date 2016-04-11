#ifndef DALI_ARRAY_FUNCTION_REDUCER_H
#define DALI_ARRAY_FUNCTION_REDUCER_H


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
        return reduce_helper(Reducer::reduce(candidate_and_state, arg), args...);
    }

    static outtuple_t reduce_helper(const outtuple_t& candidate_and_state) {
        return candidate_and_state;
    }
};



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
    static std::tuple<outtype_t, state_t> reduce(const std::tuple<outtype_t, state_t>& candidate_and_state, const T& elem) {
        return candidate_and_state;
    }

    static std::tuple<outtype_t,state_t> reduce(const std::tuple<outtype_t, state_t>& candidate_and_state, const Array& arg);
};

#endif
