#ifndef CORE_STACKED_GATED_MAT_H
#define CORE_STACKED_GATED_MAT_H

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

#include "core/CrossEntropy.h"
#include "core/Layers.h"
#include "core/Mat.h"
#include "core/Softmax.h"
#include "core/StackedModel.h"
#include "core/utils.h"

DECLARE_double(memory_penalty);

/**
StackedGatedModel
-----------------

A Model for making sequence predictions using stacked LSTM cells.

The input is gated using a sigmoid linear regression that takes
as input the last hidden cell's activation and the input to the network.

The network uses an embedding layer, and can reconstruct a sequence.

The objective function is built using masked cross entropy (only certain
input channels collect error over small intervals), and L1 loss on the
total memory used (the input gate's total activation).

**/


template<typename R>
class StackedGatedModel : public StackedModel<R> {
    typedef LSTM<R>                                             lstm;
    typedef Layer<R>                                    classifier_t;
    typedef GatedInput<R>                                     gate_t;
    typedef std::map<std::string, std::vector<std::string>> config_t;

    public:
        typedef Mat<R> mat;
        typedef std::tuple<std::vector<mat>, std::vector<mat>> state_type;
        typedef std::tuple<state_type, mat, mat> activation_t;
        typedef R value_t;
        typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
        typedef std::shared_ptr< index_mat > shared_index_mat;

        const gate_t gate;
        R memory_penalty;
        virtual std::vector<mat> parameters() const;
        virtual config_t configuration() const;
        static StackedGatedModel<R> load(std::string);
        static StackedGatedModel<R> build_from_CLI(std::string load_location,
                                                   int vocab_size,
                                                   int output_size,
                                                   bool verbose);
        StackedGatedModel(int, int, int, int, int, bool use_shortcut = false, R _memory_penalty = 0.3);
        StackedGatedModel(int, int, int, std::vector<int>&, bool use_shortcut = false, R _memory_penalty = 0.3);
        StackedGatedModel(const config_t&);
        StackedGatedModel(const StackedGatedModel<R>&, bool, bool);
        std::tuple<R, R> masked_predict_cost(
            shared_index_mat,
            shared_index_mat,
            shared_eigen_index_vector,
            shared_eigen_index_vector,
            uint offset=0,
            R drop_prob = 0.0);
        std::tuple<R,R> masked_predict_cost(
            shared_index_mat,
            shared_index_mat,
            uint,
            shared_eigen_index_vector,
            uint offset=0,
            R drop_prob = 0.0);

        virtual std::vector<int> reconstruct(Indexing::Index, int, int symbol_offset = 0) const;
        state_type get_final_activation(Indexing::Index, R drop_prob=0.0) const;

        activation_t activate(state_type&, const uint&) const;
        activation_t activate(state_type&, const eigen_index_block) const;

        virtual std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(
            Indexing::Index,
            utils::OntologyBranch::shared_branch,
            int) const;

        StackedGatedModel<R> shallow_copy() const;

};

#endif
