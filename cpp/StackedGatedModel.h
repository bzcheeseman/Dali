#ifndef STACKEDGATED_MAT_H
#define STACKEDGATED_MAT_H

#include "Mat.h"
#include "Layers.h"
#include "Softmax.h"
#include "CrossEntropy.h"
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
	typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
	typedef std::shared_ptr< index_mat > shared_index_mat;
	typedef LSTM<T>                    lstm;
	typedef Layer<T>           classifier_t;
	typedef Mat<T>                      mat;
	typedef std::shared_ptr<mat> shared_mat;
	typedef Graph<T>                graph_t;
	typedef GatedInput<T>            gate_t;
	typedef std::unordered_map<std::string, std::vector<std::string>> config_t;

	std::vector<lstm> cells;
	shared_mat    embedding;

	inline void name_parameters();
	inline void construct_LSTM_cells();

	public:
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
		StackedGatedModel(int, int, int, int, int, T _memory_penalty = 0.3);
		StackedGatedModel(int, int, int, std::vector<int>&, T _memory_penalty = 0.3);
		StackedGatedModel(const config_t&);
		std::tuple<T, T> cost_fun(graph_t&, std::shared_ptr< index_mat >, shared_eigen_index_vector, shared_eigen_index_vector, uint offset=0);
		
		template<typename K>
		std::vector<int> reconstruct_fun(K, utils::Vocab&, int, int symbol_offset = 0);
};

#endif