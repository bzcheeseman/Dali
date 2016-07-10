#include "dali/utils/assert2.h"

template<typename Container>
SlicingInProgress<Container>::SlicingInProgress(const Container& input_) :
        consumed_dims(0),
        slice({}),
        action({}),
        input(input_) {
}

template<typename Container>
SlicingInProgress<Container>::SlicingInProgress(const SlicingInProgress& other) :
        consumed_dims(other.consumed_dims),
        slice(other.slice),
        action(other.action),
        input(other.input) {
}

template<typename Container>
SlicingInProgress<Container>::SlicingInProgress() {}

template<typename Container>
SlicingInProgress<Container> SlicingInProgress<Container>::operator[](const Slice& s) {
    ASSERT2(consumed_dims < input.ndim(),
            "Slicing a scalar container is not allowed.");
    SlicingInProgress<Container> res(*this);
    res.slice.push_back(s);
    res.action.push_back(SLICE_RANGE);
    res.consumed_dims += 1;
    return res;
}

template<typename Container>
SlicingInProgress<Container> SlicingInProgress<Container>::operator[](const Broadcast& b) {
    SlicingInProgress<Container> res(*this);
    res.action.push_back(BROADCAST);
    // res.consumed_dims remains unchanged during broadcast.
    return res;
}

template<typename Container>
SlicingInProgress<Container> SlicingInProgress<Container>::operator[](const int& idx) {
    ASSERT2(consumed_dims < input.ndim(),
        "Slicing a scalar container is not allowed.");
    SlicingInProgress<Container> res(*this);
    res.slice.push_back(Slice(idx, idx+1));
    res.action.push_back(SLICE_IDX);
    res.consumed_dims += 1;
    return res;
}

template<typename Container>
SlicingInProgress<Container>::operator Container() {
    Container out = input;
    ASSERT2(consumed_dims <= input.ndim(),
            "Slicing consumed more dimensions that the input dimensionality.");
    auto next_slice = slice.begin();
    int output_depth = 0;
    for (const auto& a: action) {
        switch(a) {
            case SLICE_RANGE:
                out = out.pluck_axis(output_depth, *(next_slice++));
                output_depth += 1;
                break;
            case SLICE_IDX:
                out = out.pluck_axis(output_depth, *(next_slice++));
                out = out.squeeze(output_depth);
                break;
            case BROADCAST:
                out = out.insert_broadcast_axis(output_depth);
                output_depth += 1;
                break;
            default:
                ASSERT2(false, "Unsupported value for SliceAction.");
                break;
        }
    }
    return out;
}
