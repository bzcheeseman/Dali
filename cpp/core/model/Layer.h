#ifndef CORE_MODEL_MAP_H
#define CORE_MODEL_MAP_H

#include "core/model/Model.h"
#include "core/Mat.h"

namespace model {
    template<typename REAL_t>
    class Layer : public Model<REAL_t> {
        const int input_size;
        const int output_size;
        const bool use_bias;

        SHARED_MAT mult;
        SHARED_MAT bias;

        public:
            // Todo(szymon): We should rewrite Mat to take optional
            // weight initializer object, so that we don't have
            // 50 million constructors and the approach scales.
            Layer(int input_size, int output_size, bool use_bias=true, double bound=0.2);

            SHARED_MAT activate(GRAPH& G, SHARED_MAT input);

            std::vector<SHARED_MAT> parameters() const;
    };
}


#endif
