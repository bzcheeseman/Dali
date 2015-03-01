#ifndef MODEL_MAP_H
#define MODEL_MAP_H

#include "core/model/Model.h"
#include "core/Mat.h"

namespace model {
    template<typename REAL_t>
    class Layer : public Model<REAL_t> {
        typedef typename Model<REAL_t>::mat mat;
        typedef typename Model<REAL_t>::shared_mat shared_mat;
        typedef typename Model<REAL_t>::graph_t graph_t;

        int input_size;
        int output_size;
        bool use_bias;
        shared_mat mult;
        shared_mat bias;

        public:
            // Todo(szymon): We should rewrite Mat to take optional
            // weight initializer object, so that we don't have
            // 50 million constructors and the approach scales.
            Layer(int input_size, int output_size, bool use_bias=true, double bound=0.2);

            shared_mat operator() (graph_t& G, shared_mat input);

            shared_mat activate(graph_t& G, shared_mat input);

            std::vector<shared_mat> parameters() const;
    };
}


#endif
