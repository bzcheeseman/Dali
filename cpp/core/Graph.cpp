#include "Graph.h"
using std::stringstream;
using std::vector;

template<typename T>
Graph<T>::Graph (bool _needs_backprop) : needs_backprop(_needs_backprop) {}
template<typename T>
Graph<T>::Graph () : needs_backprop(true) {}

template<typename T>
void Graph<T>::backward () {
	for (auto it = this->backprop.rbegin(); it != this->backprop.rend(); ++it)
		(*it)();
}

template<typename T>
void Graph<T>::backward (T clip_val) {
	for (auto it = this->backprop.rbegin(); it != this->backprop.rend(); ++it)
		(*it)(clip_val);
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul_broadcast(
	shared_mat matrix1,
	shared_mat matrix2) {
	if (matrix1->n != matrix2->n || matrix2->d != 1) {
		stringstream error_msg;
		error_msg << "Matrices " << *matrix1 << " and "
		                         << *matrix2
		          << " cannot be element multiplied with broadcast,"
		             " they do not have the same dimensions.";
		throw std::invalid_argument(error_msg.str());
	}
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.array().colwise() * matrix2->w.col(0).array()).matrix();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, matrix2}), out, utils::ops::eltmul_broadcast);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul(
	shared_mat matrix1,
	shared_mat matrix2) {
	if (matrix1->d != matrix2->d && (matrix1->d == 1 || matrix2->d == 1)) {
		if (matrix1->d == 1) {
			return eltmul_broadcast(matrix2, matrix1);
		}
		return eltmul_broadcast(matrix1, matrix2);
	}
	if (matrix1->n != matrix2->n || matrix1->d != matrix2->d)
		throw std::invalid_argument("Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.array() * matrix2->w.array()).matrix();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, matrix2}), out, utils::ops::eltmul);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul_broadcast_rowwise(
	shared_mat matrix1,
	shared_mat row_vector) {
	if (matrix1->d != row_vector->d || row_vector->n != 1)
		throw std::invalid_argument("Matrices A and B^T cannot be element multiplied with broadcast, they do not have the same dimensions.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.array().rowwise() * row_vector->w.row(0).array()).matrix();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, row_vector}), out, utils::ops::eltmul_broadcast_rowwise);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul_rowwise(
	shared_mat matrix1,
	shared_mat matrix2) {

	if (matrix1->n != matrix2->d || matrix1->d != matrix2->n)
		throw std::invalid_argument("Matrices A and B^T cannot be element-wise multiplied, they do not have the same dimensions.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.array() * matrix2->w.transpose().array()).matrix();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, matrix2}), out, utils::ops::eltmul_rowwise);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add(
		shared_mat matrix1,
		shared_mat matrix2) {
	if (matrix1->d != matrix2->d && (matrix1->d == 1 || matrix2->d == 1)) {
		if (matrix1->d == 1) {
			return add_broadcast(matrix2, matrix1);
		}
		return add_broadcast(matrix1, matrix2);
	}
	if (matrix1->n != matrix2->n || matrix1->d != matrix2->d)
		throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
	auto out = std::make_shared<Mat<T>>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w + matrix2->w;
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, matrix2}), out, utils::ops::add);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add_broadcast(shared_mat matrix1, shared_mat matrix2) {
	// broadcast matrix 2:
	if (matrix1->n != matrix2->n || matrix2->d != 1)
		throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
	auto out = std::make_shared<Mat<T>>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.colwise() + matrix2->w.col(0)).matrix();
	if (needs_backprop)
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, matrix2}), out, utils::ops::add_broadcast);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add(std::initializer_list<shared_mat> matrices) {
	auto out = std::make_shared<Mat<T>>(
		(*matrices.begin())->n,
		(*matrices.begin())->d,
		false);
	for (auto& matrix : matrices) out->w += matrix->w;
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(matrices, out, utils::ops::add);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::sigmoid(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::sigmoid_operator<T>());
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(matrix1, out, utils::ops::sigmoid);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::transpose(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->d,
		matrix1->n,
		true);
	out->w = matrix1->w.transpose();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(matrix1, out, utils::ops::transpose);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::tanh(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::tanh_operator<T>());
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(matrix1, out, utils::ops::tanh);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::relu(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::relu_operator<T>());
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, utils::ops::relu);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul(
	shared_mat matrix1,
	shared_mat matrix2) {
	if (matrix1->d != matrix2->n)
		throw std::invalid_argument("matmul dimensions misaligned.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix2->d,
		true);
	out->w = matrix1->w * matrix2->w;
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, matrix2}), out, utils::ops::mul);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_with_bias(
	shared_mat matrix1,
	shared_mat matrix2,
	shared_mat bias) {
	if (matrix1->d != matrix2->n)
		throw std::invalid_argument("matmul dimensions misaligned.");
	if (matrix1->n != bias->n || bias->d != 1)
		throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix2->d,
		true);
	out->w = ((matrix1->w * matrix2->w).colwise() + bias->w.col(0)).matrix();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, matrix2, bias}), out, utils::ops::mul_with_bias);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_add_broadcast_mul_with_bias(
	shared_mat matrix1,
	shared_mat input_to_1,
	shared_mat matrix2,
	shared_mat input_to_2,
	shared_mat bias) {
	if (matrix1->d != input_to_1->n)
		throw std::invalid_argument("matmul 1 dimensions misaligned.");
	if (matrix2->d != input_to_2->n)
		throw std::invalid_argument("matmul 2 dimensions misaligned.");
	if (matrix2->n != bias->n || matrix1->n != bias->n || input_to_1->d != 1 || bias->d != 1)
		throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		input_to_2->d,
		true);
	// both input to 1 and bias are columns,
	// so we add both of those before adding the true matrix
	// product in broadcasted form
	out->w = (
		          (
		              (
		                  (matrix2->w * input_to_2->w)
		              )
		          ).colwise() + (bias->w + (matrix1->w * input_to_1->w)).col(0)
		      ).matrix();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, input_to_1, matrix2, input_to_2, bias}), out, utils::ops::mul_add_broadcast_mul_with_bias);
	return out;
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_add_mul_with_bias(std::initializer_list<shared_mat> matrices) {
	auto out = std::make_shared<mat>(
		(*matrices.begin())->n,
		(*(matrices.begin() + 1))->d,
		false);
	auto matrices_ptr = matrices.begin();
	while (matrices_ptr != (matrices.end() - 1)) {
		out->w += (*matrices_ptr)->w * (*(matrices_ptr + 1))->w;
		matrices_ptr+=2;
	}
	out->w.colwise() += (*(matrices.begin() + matrices.size() - 1))->w.col(0);
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(matrices, out, utils::ops::mul_add_mul_with_bias);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_add_mul_with_bias(const vector<shared_mat>& matrices) {
	auto out = std::make_shared<mat>(
		matrices[0]->n,
		matrices[1]->d,
		false);
	auto matrices_ptr = matrices.begin();
	while (matrices_ptr != (matrices.end() - 1)) {
		out->w += (*matrices_ptr)->w * (*(matrices_ptr + 1))->w;
		DEBUG_ASSERT_NOT_NAN((*matrices_ptr)->w);
		DEBUG_ASSERT_NOT_NAN((*(matrices_ptr + 1))->w);
		matrices_ptr+=2;
	}

	DEBUG_ASSERT_NOT_NAN((*(matrices.begin() + matrices.size() - 1))->w);
	out->w.colwise() += (*(matrices.begin() + matrices.size() - 1))->w.col(0);
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(matrices, out, utils::ops::mul_add_mul_with_bias);

	DEBUG_ASSERT_NOT_NAN(out->w);
	return out;
}

// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_add_mul_with_bias(
	shared_mat matrix1,
	shared_mat input_to_1,
	shared_mat matrix2,
	shared_mat input_to_2,
	shared_mat bias) {
	if (matrix1->d != input_to_1->n)
		throw std::invalid_argument("matmul 1 dimensions misaligned.");
	if (matrix2->d != input_to_2->n)
		throw std::invalid_argument("matmul 2 dimensions misaligned.");
	if (matrix2->n != bias->n || matrix1->n != bias->n || bias->d != 1)
		throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
	if (input_to_1->d != input_to_2->d) {
		if (input_to_1->d == 1) {
			return mul_add_broadcast_mul_with_bias(matrix1, input_to_1, matrix2, input_to_2, bias);
		}
		return mul_add_broadcast_mul_with_bias(matrix2, input_to_2, matrix1, input_to_1, bias);
	}
	auto out = std::make_shared<mat>(
		matrix1->n,
		input_to_1->d,
		true);
	out->w = (
		          (
		              (

                          (matrix1->w * input_to_1->w) +
                          (matrix2->w * input_to_2->w)
		              )
		          ).colwise() + bias->w.col(0)
		      ).matrix();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(std::initializer_list<shared_mat>({matrix1, input_to_1, matrix2, input_to_2, bias}), out, utils::ops::mul_add_mul_with_bias);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::rows_pluck(
	shared_mat x,
	index_std_vector& indices
	) {
	auto out = std::make_shared<mat>(
		x->d,
		indices.size(),
		true);
	int offset = 0;
	for (auto& i : indices)
		out->w.col(offset++) = x->w.row(i).transpose();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(x, out, indices, utils::ops::rows_pluck);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::rows_pluck(
	shared_mat x,
	eigen_index_block indices
	) {
	auto out = std::make_shared<mat>(
		x->d,
		indices.rows(),
		true);
	for (int offset = 0; offset < indices.rows(); ++offset)
		out->w.col(offset) = x->w.row(indices(offset)).transpose();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(x, out, indices, utils::ops::rows_pluck);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::row_pluck(
	shared_mat x,
	int row) {
	auto out = std::make_shared<mat>(
		x->d,
		1,
		true);
	out->w = x->w.row(row).transpose();
	if (needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		backprop.emplace_back(x, out, row, utils::ops::row_pluck);
	return out;
}

template class Graph<float>;
template class Graph<double>;