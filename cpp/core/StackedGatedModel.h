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
class StackedGatedModel {
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

		typedef std::pair<std::vector<shared_mat>, std::vector<shared_mat>> lstm_activation_t;
		typedef std::tuple<lstm_activation_t, shared_mat, T> activation_t;

		std::vector<lstm> cells;
		shared_mat    embedding;
		typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
		typedef std::shared_ptr< index_mat > shared_index_mat;

		int vocabulary_size;
		const int output_size;
		const int stack_size;
		const int input_size;
		const gate_t gate;
		const classifier_t decoder;
		std::vector<int> hidden_sizes;
		T memory_penalty;
		std::vector<shared_mat> parameters();
		config_t configuration();
		void save_configuration(std::string);
		void save(std::string);
		static StackedGatedModel<T> load(std::string);
		static StackedGatedModel<T> build_from_CLI(int, int, bool verbose=true);
		StackedGatedModel(int, int, int, int, int, T _memory_penalty = 0.3);
		StackedGatedModel(int, int, int, std::vector<int>&, T _memory_penalty = 0.3);
		StackedGatedModel(const config_t&);
		StackedGatedModel(const StackedGatedModel<T>&, bool, bool);
		std::tuple<T, T> masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, shared_eigen_index_vector, shared_eigen_index_vector, uint offset=0);
		std::tuple<T, T> masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, uint, shared_eigen_index_vector, uint offset=0);
		template<typename K>
		std::vector<int> reconstruct(K, int, int symbol_offset = 0);

		template<typename K>
		std::string reconstruct_string(K, const utils::Vocab&, int, int symbol_offset = 0);

		template<typename K>
		lstm_activation_t get_final_activation(graph_t&, const K&);

		activation_t activate(graph_t&, lstm_activation_t&, const uint&);

		template<typename K>
		std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(K, utils::OntologyBranch::shared_branch, int);

		template<typename K>
		std::string reconstruct_lattice_string(K, utils::OntologyBranch::shared_branch, int);

		StackedGatedModel<T> shallow_copy() const;

};

#endif
