#include "array_function.h"

////////////////////////////////////////////////////////////////////////////////
//         HELPER FUNCTION FOR EXTRACTING VARIOUS INFO ABOUT ARRAYS           //
////////////////////////////////////////////////////////////////////////////////

memory::Device extract_device(const Array& a) {
    return a.memory()->preferred_device;
}

MaybeDType extract_dtype(const Array& a) {
    return MaybeDType{a.dtype(), true};
}

std::vector<int> find_common_shape(bool ready, const std::vector<int>& candidate) {
    ASSERT2(ready, "Find common shape called with zero array arguments");
    return candidate;
}
