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
