#ifndef DALI_ARRAY_OP_CAST_H
#define DALI_ARRAY_OP_CAST_H

#include "dali/array/dtype.h"

class Array;
template<typename OutType>
class Assignable;

namespace op {
	Assignable<Array> astype(const Array& a, DType dtype);
}

#endif
