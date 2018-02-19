#ifndef DALI_ARRAY_SLICE_H
#define DALI_ARRAY_SLICE_H

#include <ostream>
#include <string>
#include <vector>

#include <dali/utils/optional.h>
#include "dali/utils/print_utils.h"
#include "dali/utils/assert2.h"

// Empty struct, used for informing a sliced container to insert
// a newaxis at the currently sliced dimensionality, e.g.:
//
// Tensor x = Tensor({3,4}, DTYPE_DOUBLE)[0][Broadcast()];
// x.shape() => {1, 1, 4}
//
// With the second dimension broadcasted
struct Broadcast {
};

struct Slice {
    typedef std::experimental::optional<int> optional_int_t;

    int start;
    optional_int_t end;
    int step;
    Slice();
    Slice(const optional_int_t& end);
    Slice(const optional_int_t& start, const optional_int_t& end);
    Slice(const optional_int_t& start,
          const optional_int_t& end,
          const optional_int_t& step);

    // converts negative indexes to positive indexes
    // and verifies that indexes are in range.
    Slice(const Slice& other);
    Slice(const Slice& other, const int& dim_size);
    // calls the constructor above.
    static Slice normalize_and_check(const Slice&, const int& dim_size);


    int size() const;
    bool contains(const int& index) const;
    operator std::string() const;
};

template<typename Container>
struct SlicingInProgress {
  private:
    enum SliceAction {
        SLICE_RANGE,
        SLICE_IDX,
        BROADCAST
    };

    int consumed_dims;
    std::vector<Slice>       slice;
    std::vector<SliceAction> action;
    Container input;

  public:
    SlicingInProgress();
    SlicingInProgress(const Container& input_);
    SlicingInProgress(const SlicingInProgress& other);
    SlicingInProgress<Container> operator[](const Slice& s);
    SlicingInProgress<Container> operator[](const Broadcast& b);
    SlicingInProgress<Container> operator[](const int& idx);
    operator Container();
    Container assign(const Container& other);
};

std::ostream& operator<<(std::ostream&, const Slice&);

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
        switch (a) {
            case SLICE_RANGE:
                out = out.pluck_axis(output_depth, *(next_slice++));
                output_depth += 1;
                break;
            case SLICE_IDX:
                out = out.pluck_axis(output_depth, *(next_slice++));
                out = out.squeeze(output_depth);
                break;
            case BROADCAST:
                out = out.expand_dims(output_depth);
                output_depth += 1;
                break;
            default:
                ASSERT2(false, "Unsupported value for SliceAction.");
                break;
        }
    }
    return out;
}

template<typename Container>
Container SlicingInProgress<Container>::assign(const Container& other) {
    Container sliced = *this;
    sliced.assign(other);
    return sliced;
}

#endif  // DALI_ARRAY_SLICE_H
