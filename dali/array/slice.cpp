#include "slice.h"

#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"

//////////////////////////////// UTILS /////////////////////////////////////////
// return ceil(a/b);
inline int division_ceil(int a, int b) {
    return (a + b - 1) / b;
}

//////////////////////////////// SLICE /////////////////////////////////////////

Slice::Slice() :
        Slice({}, {}, {}) {
}
Slice::Slice(const optional_int_t& end_) :
        Slice({}, end_, {}) {
}

Slice::Slice(const optional_int_t& start_, const optional_int_t& end_) :
        Slice(start_, end_, {}) {
}

Slice::Slice(const optional_int_t& start_,
             const optional_int_t& end_,
             const optional_int_t& step_) :
        start(start_.value_or(0)),
        end(end_),
        step(step_.value_or(1)) {
    if (end && ((start >= 0 && end.value() >= 0) || (start < 0 && end.value() < 0))) {
        // start and end have to have the same sign.
        ASSERT2(start < end.value(), utils::make_message(
            "Slice start (", start, ") must be less than end (", end.value(), ")"));

    }
    ASSERT2(step_ != 0, "slice step cannot be zero");
}

Slice::Slice(const Slice& other) :
    start(other.start),
    end(other.end),
    step(other.step) {
}


Slice::Slice(const Slice& other, const int& dim_size) :
        start((other.start >= 0) ? other.start : other.start + dim_size),
        end((other.end.value_or(dim_size) >= 0) ?
             other.end.value_or(dim_size)       :
             other.end.value_or(dim_size) + dim_size),
        step(other.step) {
    ASSERT2(0 <= start && start < dim_size, utils::make_message(
        "Index ", other.start, " is out of bounds for dimension with size ", dim_size, "."));
    ASSERT2(0 <= end.value() && end.value() <= dim_size, utils::make_message(
        "Index ", other.end.value(), " is out of bounds for dimension with size ", dim_size, "."));
    ASSERT2(start < end.value(), utils::make_message(
        "Slice start (", start, ") must be less than end (", end.value(), ")."));
}

Slice Slice::normalize_and_check(const Slice& slice, const int& dim_size) {
    return Slice(slice, dim_size);
}

Slice::operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

int Slice::size() const {
    ASSERT2(end,
            "can only compute size of slice if end index is specified.");
    ASSERT2(start >= 0 && end.value() >= 0,
            "slice length can only be use with positive indexing.");
    return division_ceil(end.value() - start, std::abs(step));
}

bool Slice::contains(const int& index) const {
    ASSERT2(end,
            "can only compute size of slice if end index is specified.");
    ASSERT2(start >= 0 && end.value() >= 0,
            "slice length can only be use with positive indexing.");
    bool in_range = start <= index && index < end.value();
    bool good_jump = false;
    if (step > 0) {
        good_jump = (index - start) % step == 0;
    } else {
        good_jump = (end.value() - 1 - index) % (-step) == 0;
    }
    return in_range && good_jump;
}

std::ostream& operator<<(std::ostream& out, const Slice& slice) {
    std::string end_str = "undefined";
    if (slice.end) {
        end_str = std::to_string(slice.end.value());
    }
    out << "slice(" << slice.start << "," << end_str;
    if (slice.step != 1) {
        out << "," << slice.step;
    }
    out << ")";
    return out;
}
