#ifndef CORE_STACKED_GATED_MAT_H
#define CORE_STACKED_GATED_MAT_H

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

#include "dali/mat/CrossEntropy.h"
#include "dali/mat/Softmax.h"
#include "dali/models/StackedModel.h"
#include "dali/core.h"
#include "dali/utils.h"

DECLARE_double(memory_penalty);

/**
StackedGatedModel
-----------------

A Model for making sequence predictions using stacked LSTM cells,
that constructs an embedding matrix as a convenience
and also passes the inputs through a gate for pre-filtering.

The input is gated using a sigmoid linear regression that takes
as input the last hidden cell's activation and the input to the network.

The network uses an embedding layer, and can reconstruct a sequence.

The objective function is built using masked cross entropy (only certain
input channels collect error over small intervals), and L1 loss on the
total memory used (the input gate's total activation).

**/


template<typename Z>
class StackedGatedModel : public StackedModel<Z> {
    typedef LSTM<Z>                                             lstm;
    typedef Layer<Z>                                    classifier_t;
    typedef GatedInput<Z>                                     gate_t;
    typedef std::map<std::string, std::vector<std::string>> config_t;

    public:
        typedef Mat<Z> mat;
        typedef std::vector< typename LSTM<Z>::State > state_type;
        typedef std::tuple<state_type, mat, mat> activation_t;
        typedef Z value_t;
        typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
        typedef std::shared_ptr< index_mat > shared_index_mat;

        const gate_t gate;
        Z memory_penalty;
        virtual std::vector<mat> parameters() const;
        virtual config_t configuration() const;
        static StackedGatedModel<Z> load(std::string);
        static StackedGatedModel<Z> build_from_CLI(std::string load_location,
                                                   int vocab_size,
                                                   int output_size,
                                                   bool verbose);
        StackedGatedModel(
            int vocabulary_size,
            int input_size,
            int hidden_size,
            int stack_size,
            int output_size,
            bool use_shortcut,
            bool memory_feeds_gates,
            Z _memory_penalty);
        StackedGatedModel(
            int vocabulary_size,
            int input_size,
            int output_size,
            std::vector<int>& hiddens_sizes,
            bool use_shortcut,
            bool memory_feeds_gates,
            Z _memory_penalty);
        StackedGatedModel(const config_t&);
        StackedGatedModel(const StackedGatedModel<Z>&, bool, bool);
        std::tuple<Z, Z> masked_predict_cost(
            shared_index_mat,
            shared_index_mat,
            shared_eigen_index_vector,
            shared_eigen_index_vector,
            uint offset=0,
            Z drop_prob = 0.0);
        std::tuple<Z, Z> masked_predict_cost(
            shared_index_mat,
            shared_index_mat,
            uint,
            shared_eigen_index_vector,
            uint offset=0,
            Z drop_prob = 0.0);

        virtual std::vector<int> reconstruct(Indexing::Index, int, int symbol_offset = 0) const;
        state_type get_final_activation(Indexing::Index, Z drop_prob=0.0) const;

        activation_t activate(state_type&, const uint&) const;
        activation_t activate(state_type&, const eigen_index_block) const;

        virtual std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(
            Indexing::Index,
            utils::OntologyBranch::shared_branch,
            int) const;

        StackedGatedModel<Z> shallow_copy() const;

};

#endif
