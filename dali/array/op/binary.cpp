#include "binary.h"

namespace op {
    Array all_equals(Array left, Array right) {
        // TODO(jonathan): replace with RTC
        Array out({}, DTYPE_INT32);
        if (left.shape() != right.shape() || left.dtype() != right.dtype()) {
            out(0) = 0;
            return out;
        } else {
            for (int i = 0; i < left.number_of_elements(); i++) {
                if (left.dtype() == DTYPE_INT32) {
                    if (int(left(i)) != int(right(i))) {
                        out(0) = 0;
                        return out;
                    }
                } else if (left.dtype() == DTYPE_FLOAT) {
                    if (float(left(i)) != float(right(i))) {
                        out(0) = 0;
                        return out;
                    }
                } else if (left.dtype() == DTYPE_DOUBLE) {
                    if (double(left(i)) != double(right(i))) {
                        out(0) = 0;
                        return out;
                    }
                }
            }
        }
        out(0) = 1;
        return out;
    }

    Array add(Array left, Array right) {
        throw std::runtime_error("add is not implemented.");
    }
    Array subtract(Array left, Array right) {
        throw std::runtime_error("subtract is not implemented.");
    }
    Array eltmul(Array left, Array right) {
        throw std::runtime_error("eltmul is not implemented.");
    }
    Array eltdiv(Array left, Array right) {
        throw std::runtime_error("eltdiv is not implemented.");
    }
}
