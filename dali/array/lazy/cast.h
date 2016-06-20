#ifndef DALI_ARRAY_LAZY_CAST_H
#define DALI_ARRAY_LAZY_CAST_H

template<typename ExprT, typename NewType>
struct LazyCast;

namespace lazy {
    template<typename NewType, typename ExprT>
    LazyCast<ExprT, NewType> astype(const Exp<ExprT>& expr);
}

#include "dali/array/lazy/cast-impl.h"

#endif
