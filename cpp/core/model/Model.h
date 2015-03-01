#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>

#include "core/utils.h"
#include "core/Mat.h"
#include "core/Graph.h"



namespace model {
    template<typename REAL_t>
    class Model {
        public:
            typedef Mat<REAL_t> mat;
            typedef std::shared_ptr<mat> shared_mat;
            typedef Graph<REAL_t> graph_t;

            // Should append to parent parameters.
            virtual std::vector<shared_mat> parameters() const = 0;

            virtual shared_mat activate(graph_t& G, shared_mat input) = 0;

            // Save to file.
            void save(std::string path) const;

            // Load from file.
            void load(std::string path);
    };

    /*class RecursiveModel {
        // Must call parent first.
        void reset() = 0;
    };*/

}
#endif
