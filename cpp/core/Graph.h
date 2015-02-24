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
		void backward (T);
		shared_mat eltmul_broadcast(shared_mat, shared_mat);
		shared_mat eltmul(shared_mat, shared_mat);
		/**
		Element Multiplication Broadcast Rowwise
		----------------------------------------

		To treat the special case of a row vector that must be multiplied
		with a matrix, rowwise, the we ensure that the row_vector has only
		one row, and the number of columns of this row vector is equal to
		the number of rows of matrix1.

		Inputs
		------

		shared_mat matrix1    : the matrix to multiply row wise
		shared_mat row_vector : the row vector to multiply with each row
		                        of matrix1 individually.

		Outputs
		-------

		shared_mat out : the rowwise multiply of matrix1 with row_vector.
		**/
		shared_mat eltmul_broadcast_rowwise(shared_mat, shared_mat);
		/**
		Element Multiplication Rowwise
		------------------------------

		The more general case is the element wise multiplication of two
		matrices A and B, with B transposed:

		> out = A * B^T

		Inputs
		------

		shared_mat matrix1    : the matrix to multiply
		shared_mat matrix2    : the matrix to multiply after transposing

		Outputs
		-------

		shared_mat out : the element wise product of matrix1 and matrix2^T

		**/
		shared_mat eltmul_rowwise(shared_mat, shared_mat);
		shared_mat mul_with_bias(shared_mat, shared_mat, shared_mat);
		// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
		shared_mat mul_add_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
		shared_mat mul_add_mul_with_bias(std::initializer_list<shared_mat>);
		shared_mat mul_add_mul_with_bias(const std::vector<shared_mat>&);
		// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
		// and with caveat that x is actually a column, and should be broadcasted
		shared_mat mul_add_broadcast_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
		shared_mat add_broadcast(shared_mat, shared_mat);
		/**
		Graph<T>::add
		-------------

		Add a 2 matrices together. Broadcasts the sum if
		one of them is actually a vector (number of
		columns = d = 1)

		Inputs
		------

		std::shared_ptr<Mat<T>> matrix1 : matrix to add
		std::shared_ptr<Mat<T>> matrix2 : matrix to add

		Outputs
		-------

		std::shared_ptr<Mat<T>> out : the sum of the matrices

		**/
		shared_mat add(shared_mat, shared_mat);
		shared_mat sub(shared_mat, shared_mat);


		/**
		Graph<T>::add
		-------------

		Add a list of matrices together, but does not perform any
		broadcasting (yet)

		Inputs
		------

		std::initializer_list<std::shared_ptr<Mat<T>>> matrices : matrices to add

		Outputs
		-------

		std::shared_ptr<Mat<T>> out : the sum of the matrices
		**/
		shared_mat add(std::initializer_list<shared_mat>);
		shared_mat square(shared_mat);
		/**
		Graph<T>::sum
		-------------

		Sum the elements of a matrix into a 1x1 matrix.

		Inputs
		------

		std::shared_ptr<Mat<T>> matrix1 : matrix to sum

		Outputs
		-------

		std::shared_ptr<Mat<T>> out : matrix sum

		**/
		shared_mat sum(shared_mat);
		/**
		Graph<T>::mean
		-------------

		Average the elements of a matrix into a 1x1 matrix.

		Inputs
		------

		std::shared_ptr<Mat<T>> matrix1 : matrix to average

		Outputs
		-------

		std::shared_ptr<Mat<T>> out : matrix average

		**/
		shared_mat mean(shared_mat);
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
