#ifndef CORE_MODEL_MODEL_H
#define CORE_MODEL_MODEL_H

#include <string>
#include <vector>

#include "core/Graph.h"
#include "core/Mat.h"
#include "core/Seq.h"
#include "core/utils.h"

#define MAT Mat<REAL_t>
#define SHARED_MAT std::shared_ptr<MAT>
#define GRAPH Graph<REAL_t>

namespace model {
    template<typename REAL_t>
    class Model {
        public:
            // Should append to parent parameters.
            virtual std::vector<SHARED_MAT> parameters() const = 0;

            virtual SHARED_MAT activate(GRAPH& G, SHARED_MAT input) = 0;

            virtual SHARED_MAT operator() (GRAPH& G, SHARED_MAT input);

            // Save to file.
            void save(std::string path) const;

            // Load from file.
            void load(std::string path);
    };

    template <typename REAL_t>
    class RecurrentModel: public Model<REAL_t> {
        public:
            // Must call parent first.
            virtual void reset() = 0;

            Seq<SHARED_MAT> operator() (GRAPH& G, const Seq<SHARED_MAT>& seq);

            virtual Seq<SHARED_MAT> activate_sequence(GRAPH& G, const Seq<SHARED_MAT>& seq);
    };
}
#endif
