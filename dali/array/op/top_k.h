#ifndef DALI_ARRAY_OP_TOP_K_H
#define DALI_ARRAY_OP_TOP_K_H

#include "dali/array/array.h"

namespace op {
	Array top_k(const Array& array, int k, bool sorted=true);
	Array bottom_k(const Array& array, int k, bool sorted=true);
	Array argsort(const Array& array);
	Array argsort(const Array& array, int axis);
}

#endif
