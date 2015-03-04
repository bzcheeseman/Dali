#include "core/model/Recurrent.h"

using std::make_shared;
using std::vector;

/* RNN - Recurrent Neural Network */

namespace model {
    template<typename REAL_t>
    RNN<REAL_t>::RNN(int input_size,
                     int output_size,
                     int memory_size,
                     double bound) : input_size(input_size),
                                     output_size(output_size),
                                     memory_size(memory_size),
                                     input_map(input_size, memory_size, bound),
                                     output_map(memory_size, output_size, bound),
                                     memory_map(memory_size, memory_size, bound) {
        first_memory = make_shared<MAT>(memory_size, 1, -bound/2.0, bound/2.0);
        reset();
    }

    template<typename REAL_t>
    void RNN<REAL_t>::reset() const {
        prev_memory = first_memory;
    }

    // TODO(szymon): implement activate_internal(input, hidden),
    // to give use more control.
    // output is in range 0, 1
    template<typename REAL_t>
    SHARED_MAT RNN<REAL_t>::activate(GRAPH& G, SHARED_MAT input) const {
        SHARED_MAT memory_in;
        memory_in = memory_map(G, prev_memory);
        SHARED_MAT input_in = input_map(G, input);

        SHARED_MAT memory = G.tanh(G.add(input_in, memory_in));

        prev_memory = memory;

        return G.sigmoid(output_map(G, memory));
    }

    template<typename REAL_t>
    vector<SHARED_MAT> RNN<REAL_t>::parameters() const {
        vector<SHARED_MAT> res;

        for(const model::Layer<REAL_t>& affine_map: {input_map, memory_map, output_map}) {
            const auto& params = affine_map.parameters();
            res.insert(res.end(), params.begin(), params.end());
        }

        return res;
    }

    template class RNN<float>;
    template class RNN<double>;
};
