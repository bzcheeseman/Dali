#ifndef GRAPH_MAT_H
#define GRAPH_MAT_H

#include "Mat.h"
#include "Backward.h"
#include <memory>
#include <sstream>
#include <string>

template<typename T> class Graph {

	std::vector<Backward<T>>       backprop;
	typedef Mat<T>                      mat;
	typedef std::shared_ptr<mat> shared_mat;
	public:
		bool                 needs_backprop;
		Graph (bool);
		Graph ();
		void backward ();
		shared_mat eltmul_broadcast(shared_mat, shared_mat);
		shared_mat eltmul(shared_mat, shared_mat);
		shared_mat eltmul_broadcast_rowwise(shared_mat, shared_mat);
		shared_mat eltmul_rowwise(shared_mat, shared_mat);
		shared_mat mul_with_bias(shared_mat, shared_mat, shared_mat);
		// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
		shared_mat mul_add_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
		shared_mat mul_add_mul_with_bias(std::initializer_list<shared_mat>);
		// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
		// and with caveat that x is actually a column, and should be broadcasted
		shared_mat mul_add_broadcast_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
		shared_mat add_broadcast(shared_mat, shared_mat);
		shared_mat add(shared_mat, shared_mat);
		shared_mat add(std::initializer_list<shared_mat>);
		shared_mat sigmoid(shared_mat);
		shared_mat transpose(shared_mat);
		shared_mat tanh(shared_mat);
		shared_mat relu(shared_mat);
		shared_mat mul(shared_mat, shared_mat);
		shared_mat rows_pluck(shared_mat, index_std_vector&);
		shared_mat rows_pluck(shared_mat, eigen_index_block);
		shared_mat row_pluck(shared_mat, int);
};

#endif