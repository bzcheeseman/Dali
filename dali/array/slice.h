#ifndef DALI_ARRAY_SLICE_H
#define DALI_ARRAY_SLICE_H

#include <ostream>
#include <string>

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

std::ostream& operator<<(std::ostream&, const Slice&);

#endif  // DALI_ARRAY_SLICE_H
