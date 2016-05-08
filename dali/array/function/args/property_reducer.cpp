#include "property_reducer.h"

#include "dali/array/array.h"
#include "dali/runtime_config.h"

auto BShapeCompatibleForAllArrayArgsReducer::reduce_step(
        const std::tuple<outtype_t, state_t>& candidate_and_state,
        const Array& arg) -> std::tuple<outtype_t, state_t> {
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




std::string ShapeProperty::name = "shape";
std::string DTypeProperty::name = "dtype";


std::tuple<DeviceReducer::outtype_t,DeviceReducer::state_t> DeviceReducer::reduce_step(
        const std::tuple<DeviceReducer::outtype_t, DeviceReducer::state_t>& candidate_and_state,
        const Array& arg) {
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
            return std::make_tuple(memory::default_preferred_device, DeviceReducerState{state.args_read + 1, memory::Device::device_of_doom()});
        } else {
            // we can place the computation on the currently agreed device
            return std::make_tuple(arg.memory()->preferred_device, DeviceReducerState{state.args_read + 1, arg.memory()->preferred_device});
        }
    }
}
