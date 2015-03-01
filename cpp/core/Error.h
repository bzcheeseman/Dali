#ifndef CORE_ERROR_H
#define CORE_ERROR_H

#include "core/Seq.h"
#include "core/model/Model.h"

template<typename REAL_t>
class Error {
    public:
        static SHARED_MAT cross_entropy(GRAPH& G,
                                        const Seq<SHARED_MAT>& expected,
                                        const Seq<SHARED_MAT>& prediction);
};

#endif
