#ifndef STACKED_MAT_H
#define STACKED_MAT_H

#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

#include "CrossEntropy.h"
#include "Layers.h"
#include "Mat.h"
#include "Softmax.h"
#include "utils.h"
/**
StackedModel
-----------------

A Model for making sequence predictions using stacked LSTM cells.

The network uses an embedding layer, and can reconstruct a sequence.

The objective function is built using masked cross entropy (only certain
input channels collect error over small intervals).

**/

DECLARE_int32(stack_size);
DECLARE_int32(input_size);
DECLARE_int32(hidden);
DECLARE_double(decay_rate);
DECLARE_double(rho);
DECLARE_string(save);
DECLARE_string(load);


template<typename T>
class StackedModel {
	typedef LSTM<T>                    lstm;
	typedef Layer<T>           classifier_t;
	typedef Mat<T>                      mat;
	typedef std::shared_ptr<mat> shared_mat;
	typedef Graph<T>                graph_t;
	typedef std::map<std::string, std::vector<std::string>> config_t;




	inline void name_parameters();
	inline void construct_LSTM_cells();
	inline void construct_LSTM_cells(const std::vector<LSTM<T>>&, bool, bool);

	public:

		typedef std::pair<std::vector<shared_mat>, std::vector<shared_mat>> lstm_activation_t;
		typedef std::pair<lstm_activation_t, shared_mat > activation_t;


		std::vector<lstm> cells;
		shared_mat    embedding;
		typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
		typedef std::shared_ptr< index_mat > shared_index_mat;

		int vocabulary_size;
		const int output_size;
		const int stack_size;
		const int input_size;
		const classifier_t decoder;
		std::vector<int> hidden_sizes;
		std::vector<shared_mat> parameters();
		config_t configuration();
		void save_configuration(std::string);
		void save(std::string);
		static StackedModel<T> load(std::string);
		static StackedModel<T> build_from_CLI(int, int, bool verbose=true);
		StackedModel(int, int, int, int, int);
		StackedModel(int, int, int, std::vector<int>&);
		StackedModel(const config_t&);
		StackedModel(const StackedModel<T>&, bool, bool);
		T masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, shared_eigen_index_vector, shared_eigen_index_vector, uint offset=0);
		T masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, uint, shared_eigen_index_vector, uint offset=0);
		template<typename K>
		std::vector<int> reconstruct(K, int, int symbol_offset = 0);

		template<typename K>
		lstm_activation_t get_final_activation(graph_t&, const K&);

		activation_t activate(graph_t&, lstm_activation_t&, const uint& );

		template<typename K>
		std::string reconstruct_string(K, const utils::Vocab&, int, int symbol_offset = 0);

		template<typename K>
		std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(K, utils::OntologyBranch::shared_branch, int);

		template<typename K>
		std::string reconstruct_lattice_string(K, utils::OntologyBranch::shared_branch, int);

		StackedModel<T> shallow_copy() const;
};

#endif
