#ifndef DALI_ARRAY_SLICE_H
#define DALI_ARRAY_SLICE_H

#include <ostream>
#include <string>
#include <vector>

struct Broadcast {
};

struct Slice {
    const int start;
    const int end;
    const int step;
    Slice(const int& start, const int& end, const int& step=1);

    // converts negative indexes to positive indexes
    // and verifies that indexes are in range.
    Slice(const Slice& other, const int& dim_size);
    // calls the constructor above.
    static Slice normalize_and_check(const Slice&, const int& dim_size);

    int size() const;
    bool contains(int index) const;
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
    std::vector<Slice>            slice;
    std::vector<SliceAction> action;
    Container input;

  public:
    SlicingInProgress(const Container& input_);
    SlicingInProgress(const SlicingInProgress& other);
    SlicingInProgress<Container> operator[](const Slice& s);
    SlicingInProgress<Container> operator[](const Broadcast& b);
    SlicingInProgress<Container> operator[](int idx);
    operator Container();
};


std::ostream& operator<<(std::ostream&, const Slice&);

#include "dali/array/slice-impl.h"

#endif  // DALI_ARRAY_SLICE_H
