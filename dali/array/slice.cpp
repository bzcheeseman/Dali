#include "slice.h"

#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"

//////////////////////////////// UTILS /////////////////////////////////////////
// return ceil(a/b);
inline int division_ceil(int a, int b) {
    return (a + b - 1) / b;
}

//////////////////////////////// SLICE /////////////////////////////////////////


Slice::Slice(const int& start_, const int& end_, const int& step_) :
    start(start_), end(end_), step(step_) {
    ASSERT2(start <= end,
            utils::MS() << "Slice start (" << start << ") must be less or equal than end (" << end << ")");
    ASSERT2(step_ != 0, "slice step cannot be zero");
}

Slice::Slice(const Slice& other, const int& dim_size) :
        start((other.start >= 0) ? other.start : other.start + dim_size),
        end((other.end >= 0)     ? other.end   : other.end + dim_size),
        step(other.step) {
    ASSERT2(0 <= start && start < dim_size,
            utils::MS() << "Index " << other.start << " is out of bounds for dimension with size " << dim_size);
    ASSERT2(0 <= end && end <= dim_size,
            utils::MS() << "Index " << other.end << " is out of bounds for dimension with size " << dim_size);
}

Slice Slice::normalize_and_check(const Slice& slice, const int& dim_size) {
    return Slice(slice, dim_size);
}


int Slice::size() const {
    ASSERT2(start >= 0 && end >= 0,
            "slice length can only be use with positive indexing.");
    return division_ceil(end - start, std::abs(step));
}
