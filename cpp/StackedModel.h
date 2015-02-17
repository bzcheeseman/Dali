#ifndef STACKED_MAT_H
#define STACKED_MAT_H

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
#include "OptionParser/OptionParser.h"
/**
StackedModel
-----------------

A Model for making sequence predictions using stacked LSTM cells.

The network uses an embedding layer, and can reconstruct a sequence.

The objective function is built using masked cross entropy (only certain
input channels collect error over small intervals).

**/
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

	public:
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
		static StackedModel<T> build_from_CLI(optparse::Values&, int, int, bool verbose=true);
		StackedModel(int, int, int, int, int);
		StackedModel(int, int, int, std::vector<int>&);
		StackedModel(const config_t&);
		static void add_options_to_CLI(optparse::OptionParser&);
		T masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, shared_eigen_index_vector, shared_eigen_index_vector, uint offset=0);
		T masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, uint, shared_eigen_index_vector, uint offset=0);
		template<typename K>
		std::vector<int> reconstruct(K, int, int symbol_offset = 0);

		template<typename K>
		std::string reconstruct_string(K, const utils::Vocab&, int, int symbol_offset = 0);

		template<typename K>
		std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(K, utils::OntologyBranch::shared_branch, int);

		template<typename K>
		std::string reconstruct_lattice_string(K, utils::OntologyBranch::shared_branch, int);
};

#endif
