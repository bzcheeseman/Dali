#ifndef DALI_ARRAY_JIT_RESHAPE_H
#define DALI_ARRAY_JIT_RESHAPE_H

#include "dali/array/array.h"

namespace op {
	namespace jit {
    	Array jit_reshape(const Array& array, const std::vector<int>& shape);
    }
}

#endif
