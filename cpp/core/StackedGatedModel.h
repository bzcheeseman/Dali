#ifndef STACKEDGATED_MAT_H
#define STACKEDGATED_MAT_H

#include <iostream>
#include <fstream>
#include <sstream>
#include "utils.h"
#include <map>
#include <unordered_map>
#include "Mat.h"
#include "Layers.h"
#include "Softmax.h"
#include "CrossEntropy.h"
#include "StackedModel.h"
#include "core/RecurrentEmbeddingModel.h"


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


template<typename T>
class StackedGatedModel : public RecurrentEmbeddingModel<T> {
        typedef LSTM<T>                    lstm;
        typedef Layer<T>           classifier_t;
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Graph<T>                graph_t;
        typedef GatedInput<T>            gate_t;
        typedef std::map<std::string, std::vector<std::string>> config_t;

        inline void name_parameters();
        inline void construct_LSTM_cells();
        inline void construct_LSTM_cells(const std::vector<lstm>&, bool, bool);

        public:
                typedef std::pair<std::vector<shared_mat>, std::vector<shared_mat>> state_type;
                typedef std::tuple<state_type, shared_mat, shared_mat> activation_t;
                typedef T value_t;

                std::vector<lstm> cells;
                shared_mat    embedding;
                typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
                typedef std::shared_ptr< index_mat > shared_index_mat;

                const gate_t gate;
                const classifier_t decoder;
                T memory_penalty;
                virtual std::vector<shared_mat> parameters() const;
                virtual config_t configuration() const;
                void save(std::string) const;
                static StackedGatedModel<T> load(std::string);
                static StackedGatedModel<T> build_from_CLI(std::string load_location,
                                                                                           int vocab_size,
                                                                                           int output_size,
                                                                                           bool verbose);
                StackedGatedModel(int, int, int, int, int, T _memory_penalty = 0.3);
                StackedGatedModel(int, int, int, std::vector<int>&, T _memory_penalty = 0.3);
                StackedGatedModel(const config_t&);
                StackedGatedModel(const StackedGatedModel<T>&, bool, bool);
                std::tuple<T, T> masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, shared_eigen_index_vector, shared_eigen_index_vector, uint offset=0, T drop_prob = 0.0);
                std::tuple<T, T> masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, uint, shared_eigen_index_vector, uint offset=0, T drop_prob = 0.0);

                std::vector<int> reconstruct(Indexing::Index, int, int symbol_offset = 0) const;
                state_type get_final_activation(graph_t&, Indexing::Index, T drop_prob=0.0) const;

                activation_t activate(graph_t&, state_type&, const uint&) const;
                activation_t activate(graph_t&, state_type&, const eigen_index_block) const;

                std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(Indexing::Index, utils::OntologyBranch::shared_branch, int) const;

                StackedGatedModel<T> shallow_copy() const;

};

#endif
