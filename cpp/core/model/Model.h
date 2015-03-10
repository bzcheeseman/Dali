#ifndef CORE_MODEL_MODEL_H
#define CORE_MODEL_MODEL_H

#include <string>
#include <vector>

#include "core/Graph.h"
#include "core/Mat.h"
#include "core/Seq.h"
#include "core/utils.h"

namespace model {
    template<typename REAL_t>
    class Model {
        public:
            // Should append to parent parameters.
            virtual std::vector<SHARED_MAT> parameters() const = 0;

            virtual SHARED_MAT activate(GRAPH& G, SHARED_MAT input) const = 0;

            virtual SHARED_MAT operator() (GRAPH& G, SHARED_MAT input) const;

            // Save to file.
            void save(std::string path) const;

            // Load from file.
            void load(std::string path);
    };

    template <typename REAL_t>
    class RecurrentModel: public Model<REAL_t> {
        public:
            // Must call parent first.
            virtual void reset() const = 0;

            Seq<SHARED_MAT> operator() (GRAPH& G, const Seq<SHARED_MAT>& seq) const;

            virtual Seq<SHARED_MAT> activate_sequence(GRAPH& G, const Seq<SHARED_MAT>& seq) const;
    };
}
#endif
