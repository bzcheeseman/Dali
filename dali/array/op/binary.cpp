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
}
