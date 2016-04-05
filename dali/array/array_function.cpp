#include "array_function.h"

////////////////////////////////////////////////////////////////////////////////
//         HELPER FUNCTION FOR EXTRACTING VARIOUS INFO ABOUT ARRAYS           //
////////////////////////////////////////////////////////////////////////////////

memory::Device extract_device(Array a) {
    return a.memory()->preferred_device;
}

MaybeDType extract_dtype(Array a) {
    return MaybeDType{a.dtype(), true};
}
