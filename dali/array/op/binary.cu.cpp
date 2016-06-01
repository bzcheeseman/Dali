#include "binary.h"

#include "dali/array/array.h"
#include "dali/array/lazy/binary.h"
#include "dali/array/op/unary.h"

namespace op {
    AssignableArray add(const Array& a, const Array& b) {
        return lazy::add(a, b);
    }

    AssignableArray add(const std::vector<Array>& arrays) {
        ASSERT2(arrays.size() > 0, "op::add takes requires at least 1 array");
        if (arrays.size() == 1) {
            return op::identity(arrays[0], false);
        } else if (arrays.size() == 2) {
            return lazy::add(arrays[0], arrays[1]);
        } else {
            return AssignableArray([arrays](Array& out, const OPERATOR_T& operator_t) {
                Array res = arrays[0];
                for (int i = 1; i < arrays.size(); i += 4) {
                    Array newres;
                    if (i + 3 < arrays.size()) {
                        // do 4 additions at once
                        newres = lazy::add(
                            lazy::add(
                                lazy::add(
                                    lazy::add(
                                        res,
                                        arrays[i]
                                    ),
                                    arrays[i+1]
                                ),
                                arrays[i+2]
                            ),
                            arrays[i+3]
                        );
                    } else if (i + 2 < arrays.size()) {
                        // do 3 additions at once
                        newres = lazy::add(
                            lazy::add(
                                lazy::add(
                                    res,
                                    arrays[i]
                                ),
                                arrays[i+1]
                            ),
                            arrays[i+2]
                        );
                    } else if (i + 1 < arrays.size()) {
                    // do 2 additions at once
                        newres = lazy::add(lazy::add(res, arrays[i]), arrays[i+1]);
                    } else {
                    // do 1 addition
                        newres = lazy::add(res, arrays[i]);
                    }
                    res.reset();
                    res = newres;
                }
                op::identity(res, false).assign_to(out, operator_t);
            });
        }
    }

    AssignableArray sub(const Array& a, const Array& b) {
        return lazy::sub(a, b);
    }

    AssignableArray eltmul(const Array& a, const Array& b) {
        return lazy::eltmul(a, b);
    }

    AssignableArray eltdiv(const Array& a, const Array& b) {
        return lazy::eltdiv(a, b);
    }

    AssignableArray pow(const Array& a, const Array& b) {
        return lazy::pow(a, b);
    }
}  // namespace op
