#ifndef BACKWARD_MAT_H
#define BACKWARD_MAT_H

#include "Mat.h"
#include <iostream>
#include <initializer_list>
#include <string>
#include <sstream>

template<typename T> class Backward {
	int ix;
	uint* indices;
	int num_indices;
	uint type;
	typedef Mat<T> mat;
	typedef std::shared_ptr<mat> shared_mat;
	void backward_rows_pluck();
	void backward_mul_add_mul_with_bias();
	public:
		std::vector<shared_mat> matrices;

		shared_mat out;
		Backward(shared_mat, shared_mat, uint);
		Backward(shared_mat, shared_mat, int, uint);
		Backward(shared_mat, shared_mat, index_std_vector&, uint);
		Backward(shared_mat, shared_mat, eigen_index_block, uint);
		Backward(std::initializer_list<shared_mat>, shared_mat, uint);

		operator std::string() const;
		std::string op_type () const;
		void operator ()();
};

template<typename T>
std::ostream& operator<<(std::ostream&, const Backward<T>&);

#endif