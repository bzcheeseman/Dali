#include "reducers.h"
#include "dali/array/op2/fused_operation.h"

namespace op2 {
	FusedOperation sum(const FusedOperation& x) {
		return all_reduce(x, "reducers::sum");
    }
}
