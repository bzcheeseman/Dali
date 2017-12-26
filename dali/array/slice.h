#ifndef DALI_ARRAY_SLICE_H
#define DALI_ARRAY_SLICE_H

#include <ostream>
#include <string>
#include <vector>

#include <dali/utils/optional.h>
#include "dali/utils/print_utils.h"


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
};


std::ostream& operator<<(std::ostream&, const Slice&);

#include "dali/array/slice-impl.h"

#endif  // DALI_ARRAY_SLICE_H
