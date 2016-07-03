#include "binary.h"

#include "dali/array/array.h"
#include "dali/array/lazy/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/lazy/circular_convolution.h"

namespace op {
    Assignable<Array> add(const Array& a, const Array& b) {
        return lazy::add(a, b);
    }

    Assignable<Array> add(const std::vector<Array>& arrays) {
        ASSERT2(arrays.size() > 0, "op::add takes requires at least 1 array");
        if (arrays.size() == 1) {
            return op::identity(arrays[0], false);
        } else if (arrays.size() == 2) {
            return lazy::add(arrays[0], arrays[1]);
        } else {
            return Assignable<Array>([arrays](Array& out, const OPERATOR_T& operator_t) {
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

    Assignable<Array> sub(const Array& a, const Array& b) {
        return lazy::sub(a, b);
    }

    Assignable<Array> eltmul(const Array& a, const Array& b) {
        return lazy::eltmul(a, b);
    }

    Assignable<Array> eltdiv(const Array& a, const Array& b) {
        return lazy::eltdiv(a, b);
    }

    Assignable<Array> pow(const Array& a, const Array& b) {
        return lazy::pow(a, b);
    }

    Assignable<Array> equals(const Array& left, const Array& right) {
        return lazy::equals(left, right);
    }

    Assignable<Array> circular_convolution(const Array& content, const Array& shift) {
        return lazy::circular_convolution(content, shift);
    }

}  // namespace op
