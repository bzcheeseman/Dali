#include "Mat.h"
#include "utils.h"

using namespace Eigen;

template<typename T>
Mat<T>::Mat (int _n, int _d) : n(_n), d(_d), random_id(utils::get_random_id()) {
    w = eigen_mat::Zero(n,d);
    dw = eigen_mat::Zero(n,d);
}
template<typename T>
Mat<T>::Mat (int _n, int _d, bool empty) : n(_n), d(_d), random_id(utils::get_random_id()) {
	w  = empty ? eigen_mat(n,d) : eigen_mat::Zero(n,d);
	dw = eigen_mat::Zero(n,d);
}

template<typename T>
void Mat<T>::print () {
	for (int i = 0; i < n ; ++i) {
		std::cout << (i == 0 ? "[" : " ");
		for (int j = 0; j < d; ++j) {
			std::cout << std::fixed
			          << std::setw( 7 ) // keep 7 digits
			          << std::setprecision( 3 ) // use 3 decimals
			          << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
			          << this->w(i,j) << " ";
		}
		std::cout << (i == n-1 ? "]" : "\n");
	}
	std::cout << std::endl;
}

template<typename T>
void Mat<T>::npy_save (std::string fname, std::string mode) {
	const unsigned int shape[] = {(unsigned int) n,(unsigned int) d};
	cnpy::npy_save(fname, w.data(), shape, 2, mode);
}

template<typename T>
Mat<T>::Mat (std::string fname) : random_id(utils::get_random_id()) {
	auto arr = cnpy::npy_load(fname);

	n = arr.shape[0];
	d = arr.shape.size() > 1 ? arr.shape[1] : 1;

	w  = eigen_mat(n,d);
    dw = eigen_mat::Zero(n,d);

	if (arr.word_size == sizeof(double)) {
		double* loaded_data_double = reinterpret_cast<double*>(arr.data);
		if (arr.fortran_order) {
			Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, n, d);
			w = wrapped_mat_double_ft.cast<T>();
		} else {
			Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, n, d);
			w = wrapped_mat_double.cast<T>();
		}
	} else if (arr.word_size == sizeof(float)) {
		float* loaded_data_float = reinterpret_cast<float*>(arr.data);
		if (arr.fortran_order) {
			Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, n, d);
			w = wrapped_mat_float_ft.cast<T>();
		} else {
			Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, n, d);
			w = wrapped_mat_float.cast<T>();
		}
	} else {
		std::stringstream error_msg;
		error_msg << "Could not load numpy matrix : \""
		   << fname << "\". File dtype (" << arr.word_size << ") not recognized as float or double.";
		throw std::invalid_argument(error_msg.str());
	}
	arr.destruct();
}

template<typename T>
Mat<T>::~Mat() {}

template<typename T>
Mat<T>::Mat (int _n, int _d, T std) : n(_n), d(_d), random_id(utils::get_random_id()) {
	std::default_random_engine generator;
	std::normal_distribution<T> distribution(0.0, std);
	std::random_device rd;
	generator.seed(rd());
	auto randn = [&] (int) {return distribution(generator);};
	this->w = eigen_mat::NullaryExpr(n,d, randn);
	this->dw = eigen_mat::Zero(n,d);
}

template<typename T>
Mat<T>::Mat (int _n, int _d, T lower, T upper) : n(_n), d(_d), random_id(utils::get_random_id()) {
	std::default_random_engine generator;
	std::uniform_real_distribution<T> distribution(lower, upper);
	std::random_device rd;
	generator.seed(rd());
	auto randn = [&] (int) {return distribution(generator);};
	this->w = eigen_mat::NullaryExpr(n,d, randn);
	this->dw = eigen_mat::Zero(n,d);
}

template<typename T>
Mat<T> Mat<T>::RandMat(int n, int d, T std) {
	// is in fact using C++ 11 's rvalue, move operator,
	// so no copy is made.
	return Mat(n, d, std);
}

template<typename T>
Mat<T> Mat<T>::Empty(int n, int d) {
	// use an empty matrix and modify
	// it so as to not incur the filling
	// with zeros cost.
	return Mat(n, d, true);
}

template<typename T>
std::ostream& operator<<(std::ostream& strm, const Mat<T>& a) {
	return strm << "<#Mat n=" << a.n << ", d=" << a.d << ">";
}

template std::ostream& operator<< <double>(std::ostream& strm, const Mat<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const Mat<float>& a);

template <typename T>
std::size_t std::hash<Mat<T>>::operator()(const Mat<T>& k) const {
	return k.random_id;
}

template <typename T>
static bool operator!=(const Mat<T>& matrix1, const Mat<T>& matrix2) {
    return matrix2.random_id != matrix1.random_id;
}

template <typename T>
static bool operator==(const Mat<T>& matrix1, const Mat<T>& matrix2) {

    return matrix2.random_id == matrix1.random_id;
}

template<typename T> Backward<T>::Backward (
	shared_mat _matrix1,
	shared_mat _out,
	uint _type) : type(_type), matrix1(_matrix1), out(_out) {
	matrix2 = NULL;
	matrix3 = NULL;
	matrix4 = NULL;
	matrix5 = NULL;
}

template<typename T> Backward<T>::Backward (
	shared_mat _matrix1,
	shared_mat _out,
	int index,
	uint _type) : type(_type), matrix1(_matrix1), out(_out), ix(index) {
	matrix2 = NULL;
	matrix3 = NULL;
	matrix4 = NULL;
	matrix5 = NULL;
}

template<typename T> Backward<T>::Backward (
	shared_mat _matrix1,
	shared_mat _out,
	index_std_vector& _indices,
	uint _type)  : type(_type), matrix1(_matrix1), out(_out), num_indices(_indices.size()) {
	matrix2 = NULL;
	matrix3 = NULL;
	matrix4 = NULL;
	matrix5 = NULL;
	indices = _indices.data();
}

template<typename T> Backward<T>::Backward (
	shared_mat _matrix1,
	shared_mat _out,
	eigen_index_block _indices,
	uint _type)  : type(_type), matrix1(_matrix1), out(_out), num_indices(_indices.rows()) {
	matrix2 = NULL;
	matrix3 = NULL;
	matrix4 = NULL;
	matrix5 = NULL;
	indices = _indices.data();
}


template<typename T> Backward<T>::Backward (
	shared_mat _matrix1,
	shared_mat _matrix2,
	shared_mat _out,
	uint _type)  : type(_type), matrix1(_matrix1), matrix2(_matrix2), out(_out) {
	matrix3 = NULL;
	matrix4 = NULL;
	matrix5 = NULL;
}

template<typename T> Backward<T>::Backward (
	shared_mat _matrix1,
	shared_mat _matrix2,
	shared_mat _matrix3,
	shared_mat _out,
	uint _type) : type(_type), matrix1(_matrix1), matrix2(_matrix2), matrix3(_matrix3), out(_out){
	matrix4 = NULL;
	matrix5 = NULL;
}

template<typename T> Backward<T>::Backward (
	shared_mat _matrix1,
	shared_mat _matrix2,
	shared_mat _matrix3,
	shared_mat _matrix4,
	shared_mat _out,
	uint _type) : type(_type), matrix1(_matrix1), matrix2(_matrix2), matrix3(_matrix3), matrix4(_matrix4), out(_out){
	matrix5 = NULL;
}

template<typename T> Backward<T>::Backward (
	shared_mat _matrix1,
	shared_mat _matrix2,
	shared_mat _matrix3,
	shared_mat _matrix4,
	shared_mat _matrix5,
	shared_mat _out,
	uint _type) : type(_type), matrix1(_matrix1), matrix2(_matrix2), matrix3(_matrix3), matrix4(_matrix4), matrix5(_matrix5), out(_out) {}

template<typename T>
std::ostream& operator<<(std::ostream& strm, const Backward<T>& a) {
	if (a.matrix2 != NULL) {
		return strm << "<#Backward matrix1=" << *(a.matrix1) << ", matrix2=" << *(a.matrix2) << ", out=" << *(a.out) << ", type=\""<< a.op_type() << "\">";
	}
	return strm << "<#Backward matrix1=" << *(a.matrix1) << ", out=" << *(a.out) << ", type=\""<< a.op_type() << "\">";
}

template std::ostream& operator<< <double>(std::ostream& strm, const Backward<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const Backward<float>& a);

template<typename T> 
std::string Backward<T>::op_type () const {
	switch(this->type) {
		case utils::ops::add:
			return "add";
		case utils::ops::eltmul:
			return "eltmul";
		case utils::ops::eltmul_rowwise:
			return "eltmul_rowwise";
		case utils::ops::tanh:
			return "tanh";
		case utils::ops::sigmoid:
			return "sigmoid";
		case utils::ops::relu:
			return "relu";
		case utils::ops::mul:
			return "mul";
		case utils::ops::row_pluck:
			return "row_pluck";
		case utils::ops::rows_pluck:
			return "rows_pluck";
		case utils::ops::add_broadcast:
			return "add_broadcast";
		case utils::ops::eltmul_broadcast:
			return "eltmul_broadcast";
		case utils::ops::eltmul_broadcast_rowwise:
			return "eltmul_broadcast_rowwise";
		case utils::ops::mul_with_bias:
			return "mul_with_bias";
		case utils::ops::mul_add_mul_with_bias:
			return "mul_add_mul_with_bias";
		case utils::ops::mul_add_broadcast_mul_with_bias:
			return "mul_add_broadcast_mul_with_bias";
		case utils::ops::transpose:
			return "transpose";
		default:
			return "?";
			break;
	}
}

template<typename T>
void Backward<T>::backward_rows_pluck() {
	auto index_ptr = indices;
	for (int i = 0; i < num_indices; ++i) {
		// for each row do the same operation as for row_pluck:
		matrix1->dw.row(*index_ptr).noalias() += out->dw.col(i).transpose();
		index_ptr++;
	}
}

template<typename T> 
void Backward<T>::operator ()() {
	switch(this->type) {
		case utils::ops::add:
			matrix1->dw.noalias() += out->dw;
			matrix2->dw.noalias() += out->dw;
			break;
		case utils::ops::add_broadcast:
			matrix1->dw.noalias() += out->dw;
			matrix2->dw.noalias() += out->dw.rowwise().sum();
			break;
		case utils::ops::eltmul:
			matrix1->dw.noalias() += ((matrix2->w).array() * (out->dw).array()).matrix();
			matrix2->dw.noalias() += ((matrix1->w).array() * (out->dw).array()).matrix();
			break;
		case utils::ops::eltmul_rowwise:
			matrix1->dw.noalias() += ((matrix2->w).transpose().array() * (out->dw).array()).matrix();
			matrix2->dw.noalias() += ((matrix1->w).array() * (out->dw).array()).matrix().transpose();
			break;
		case utils::ops::eltmul_broadcast:
			matrix1->dw.noalias() += ((out->dw).array().colwise() * (matrix2->w).col(0).array()).matrix();
			matrix2->dw.noalias() += ((matrix1->w).array() * (out->dw).array()).matrix().rowwise().sum();
			break;
		case utils::ops::eltmul_broadcast_rowwise:
			// matrix1->dw.noalias() += ((out->dw).array().rowwise() * (matrix2->w).row(0).array()).matrix();
			// matrix2->dw.noalias() += ((matrix1->w).array() * (out->dw).array()).matrix().colwise().sum();
			break;
		case utils::ops::sigmoid:
			matrix1->dw.noalias() += (((out->w).array() * (1.0 - out->w.array())) * out->dw.array()).matrix();
			break;
		case utils::ops::mul:
			matrix1->dw.noalias() += (out->dw) * ((matrix2->w).transpose());
			matrix2->dw.noalias() += matrix1->w.transpose() * (out->dw);
			break;
		case utils::ops::relu:
			matrix1->dw.noalias() += (out->w.unaryExpr(utils::sign_operator<T>()).array() * out->dw.array()).matrix();
			break;
		case utils::ops::tanh:
			matrix1->dw.noalias() += (out->w.unaryExpr(utils::dtanh_operator<T>()).array() * out->dw.array()).matrix();
			break;
		case utils::ops::row_pluck:
			matrix1->dw.row(ix).noalias() += out->dw.col(0).transpose();
			break;
		case utils::ops::rows_pluck:
			// number of rows:
			backward_rows_pluck();
			break;
		case utils::ops::mul_with_bias:
			matrix1->dw.noalias() += (out->dw) * ((matrix2->w).transpose());
			matrix2->dw.noalias() += matrix1->w.transpose() * (out->dw);
			matrix3->dw.noalias() += out->dw.rowwise().sum();
			break;
		case utils::ops::mul_add_mul_with_bias:
			// first multiply:
			matrix1->dw.noalias() += (out->dw) * ((matrix2->w).transpose());
			matrix2->dw.noalias() += matrix1->w.transpose() * (out->dw);
			// second multiply:
			matrix3->dw.noalias() += (out->dw) * ((matrix4->w).transpose());
			matrix4->dw.noalias() += matrix3->w.transpose() * (out->dw);
			// bias vector:
			matrix5->dw.noalias() += out->dw.rowwise().sum();
			break;
		case utils::ops::mul_add_broadcast_mul_with_bias:
			// first multiply:
			// broadcasting input means taking outer product here:
			matrix1->dw += ((out->dw).rowwise().sum() * ((matrix2->w).transpose()));
			// broadcasting output means sum after the reverse product here:
			matrix2->dw.noalias() += (matrix1->w.transpose() * (out->dw)).rowwise().sum();
			// second multiply:
			matrix3->dw.noalias() += (out->dw) * ((matrix4->w).transpose());

			matrix4->dw.noalias() += matrix3->w.transpose() * (out->dw);
			// bias vector:
			matrix5->dw.noalias() += out->dw.rowwise().sum();
			break;
		case utils::ops::transpose:
			matrix1->dw.noalias() += (out->dw).transpose();
			break;
		default:
			std::stringstream error_msg;
			error_msg << "NotImplemented: Do not know how to backpropagate for this type => "
			   << op_type() << " (" << type << ")";
			throw std::invalid_argument(error_msg.str());
			break;
	}
}

template<typename T>
std::shared_ptr<Mat<T>> softmax(std::shared_ptr<Mat<T>> matrix) {
	auto layer_max = matrix->w.rowwise().maxCoeff().array().matrix();
	auto exped_distributions = (matrix->w.colwise() - layer_max.col(0)).array().exp().matrix();
	
	auto out = std::make_shared<Mat<T>>(
		matrix->n,
		matrix->d,
		false);

	auto total_distribution = exped_distributions.rowwise().sum().array().matrix();
	out->w = (exped_distributions.array().colwise() / total_distribution.col(0).array());
	return out;
}

template<typename T>
T cross_entropy(std::shared_ptr<Mat<T>> logprobs, int& target) {
	std::shared_ptr<Mat<T>> probs = softmax(logprobs);
	T cost = -std::log(probs->w(target,0));

	logprobs->dw = probs->w;
	// write gradients into log probabilities
	logprobs->dw(target, 0) -= 1;
	return cost;
}

template<typename T, typename M>
T cross_entropy(std::shared_ptr<Mat<T>> logprobs, const M targets) {
	std::shared_ptr<Mat<T>> probs = softmax(logprobs);
	T cost = 0.0;

	logprobs->dw = probs->w;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < targets.rows(); i++) {
		cost -= std::log(probs->w(targets(i),i));
		logprobs->dw(targets(i), i) -= 1;
	}

	return cost;
}

template<typename Z, typename M, typename K, typename F>
Z masked_cross_entropy(std::shared_ptr<Mat<Z>> logprobs,
	uint& T,
	const K& loss_start,
	const F& codelens,
	const M targets) {
	std::shared_ptr<Mat<Z>> probs = softmax(logprobs);
	Z cost = 0.0;
	logprobs->dw = probs->w;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < targets.rows(); i++) {
		if (T >= loss_start(i) && (T < loss_start(i) - codelens(i) )) {
			cost -= std::log(probs->w(targets(i),i));
			logprobs->dw(targets(i), i) -= 1;
		}
	}
	return cost;
}

template<typename Z, typename M, typename F>
Z masked_cross_entropy(std::shared_ptr<Mat<Z>> logprobs,
	uint& T,
	shared_eigen_index_vector loss_start,
	const F& codelens,
	const M targets) {
	std::shared_ptr<Mat<Z>> probs = softmax(logprobs);
	Z cost = 0.0;
	logprobs->dw = probs->w;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < targets.rows(); i++) {
		if (T >= (*loss_start)(i) && (T < (*loss_start)(i) - codelens(i) )) {
			cost -= std::log(probs->w(targets(i),i));
			logprobs->dw(targets(i), i) -= 1;
		}
	}
	return cost;
}

template<typename Z, typename M, typename K>
Z masked_cross_entropy(std::shared_ptr<Mat<Z>> logprobs,
	uint& T,
	const K& loss_start,
	shared_eigen_index_vector codelens,
	const M targets) {
	std::shared_ptr<Mat<Z>> probs = softmax(logprobs);
	Z cost = 0.0;
	logprobs->dw = probs->w;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < targets.rows(); i++) {
		if (T >= loss_start(i) && (T < loss_start(i) - (*codelens)(i) )) {
			cost -= std::log(probs->w(targets(i),i));
			logprobs->dw(targets(i), i) -= 1;
		}
	}
	return cost;
}
/**
Masked Cross Entropy Loss
-------------------------

Given a probability distribution p at a time T, for k channels,
and k different targets, apply KL Divergence loss
on the channels that are have T >= loss_start[k] and
T < loss_start[k] + codelens[k] (so from T to T+codelen error
will be picked up by channel k).

Inputs
------

std::shared_ptr<Mat<Z>> logprobs : the log probabilities (unnormalized)
uint& T : the log probabilities (unnormalized)
shared_eigen_index_vector loss_start : where to start picking up errors for channel k
shared_eigen_index_vector codelens : how long does channel k pick up errors
const M targets : the labels at time T

Outputs
-------

Z cost : the total KL divergence at this time step for
         the relevant channels

*/
template<typename Z, typename M>
Z masked_cross_entropy(std::shared_ptr<Mat<Z>> logprobs,
	uint& T,
	shared_eigen_index_vector loss_start,
	shared_eigen_index_vector codelens,
	const M targets) {
	std::shared_ptr<Mat<Z>> probs = softmax(logprobs);
	Z cost = 0.0;
	logprobs->dw = probs->w;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < targets.rows(); i++) {
		if (T >= (*loss_start)(i) && (T < (*loss_start)(i) + (*codelens)(i) )) {
			cost -= std::log(probs->w(targets(i),i));
			logprobs->dw(targets(i), i) -= 1;
		}
	}
	return cost;
}

template float  cross_entropy(std::shared_ptr<Mat<float>>, eigen_index_block);
template double cross_entropy(std::shared_ptr<Mat<double>>, eigen_index_block);
template float  cross_entropy(std::shared_ptr<Mat<float>>, eigen_index_block_scalar);
template double cross_entropy(std::shared_ptr<Mat<double>>, eigen_index_block_scalar);

template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_block&, const eigen_index_block&, const eigen_index_block);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, const eigen_index_vector&, const eigen_index_block);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_block&, const eigen_index_vector&, const eigen_index_block);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, const eigen_index_block&, const eigen_index_block);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, shared_eigen_index_vector, const eigen_index_block&, const eigen_index_block);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, shared_eigen_index_vector, const eigen_index_block);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const eigen_index_block);

template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_block&, const eigen_index_block&, const eigen_index_block);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, const eigen_index_vector&, const eigen_index_block);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_block&, const eigen_index_vector&, const eigen_index_block);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, const eigen_index_block&, const eigen_index_block);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, shared_eigen_index_vector, const eigen_index_block&, const eigen_index_block);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, shared_eigen_index_vector, const eigen_index_block);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const eigen_index_block);

template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_block&, const eigen_index_block&, const eigen_index_block_scalar);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, const eigen_index_vector&, const eigen_index_block_scalar);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_block&, const eigen_index_vector&, const eigen_index_block_scalar);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, const eigen_index_block&, const eigen_index_block_scalar);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, shared_eigen_index_vector, const eigen_index_block&, const eigen_index_block_scalar);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, shared_eigen_index_vector, const eigen_index_block_scalar);
template double masked_cross_entropy(std::shared_ptr<Mat<double>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const eigen_index_block_scalar);

template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_block&, const eigen_index_block&, const eigen_index_block_scalar);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, const eigen_index_vector&, const eigen_index_block_scalar);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_block&, const eigen_index_vector&, const eigen_index_block_scalar);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, const eigen_index_block&, const eigen_index_block_scalar);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, shared_eigen_index_vector, const eigen_index_block&, const eigen_index_block_scalar);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, shared_eigen_index_vector, const eigen_index_block_scalar);
template float masked_cross_entropy(std::shared_ptr<Mat<float>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const eigen_index_block_scalar);

/**
Masked Sum
----------

Sum values[k] if timestep T
if within [loss_start[k], loss_start[k] + codelens[k]),
gradient is columnwise vector of 1s.

Inputs:
-------

std::shared_ptr<Mat<Z>> values : the data columns subject to summing.
uint& T : the log probabilities (unnormalized)
shared_eigen_index_vector loss_start : where to start picking up errors for channel k
shared_eigen_index_vector codelens : how long does channel k pick up errors

Outputs:
--------

Z cost : the total sum along the non-masked columns of values.

*/
template<typename Z>
Z masked_sum(std::shared_ptr<Mat<Z>> values,
	uint& T,
	shared_eigen_index_vector loss_start,
	shared_eigen_index_vector codelens,
	Z gparent) {

	Z cost = 0.0;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < values->d; i++) {
		if (T >= (*loss_start)(i) && (T < (*loss_start)(i) + (*codelens)(i) )) {
			cost += values->w.col(i).sum() * gparent;
			values->dw.col(i).fill(gparent);
		}
	}
	return cost;
}

template<typename Z, typename K, typename F>
Z masked_sum(std::shared_ptr<Mat<Z>> values,
	uint& T,
	const K& loss_start,
	const F& codelens,
	Z gparent) {
	Z cost = 0.0;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < values->d; i++) {
		if (T >= loss_start(i) && (T < loss_start(i) + codelens(i) )) {
			cost += values->w.col(i).sum() * gparent;
			values->dw.col(i).fill(gparent);
		}
	}
	return cost;
}

template<typename Z, typename K>
Z masked_sum(std::shared_ptr<Mat<Z>> values,
	uint& T,
	const K& loss_start,
	shared_eigen_index_vector codelens,
	Z gparent) {
	Z cost = 0.0;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < values->d; i++) {
		if (T >= loss_start(i) && (T < loss_start(i) + (*codelens)(i) )) {
			cost += values->w.col(i).sum() * gparent;
			values->dw.col(i).fill(gparent);
		}
	}
	return cost;
}

template<typename Z, typename F>
Z masked_sum(std::shared_ptr<Mat<Z>> values,
	uint& T,
	shared_eigen_index_vector loss_start,
	const F& codelens,
	Z gparent) {
	Z cost = 0.0;
	// get cost for each pair of target and datastream:
	for (size_t i = 0; i < values->d; i++) {
		if (T >= (*loss_start)(i) && (T < (*loss_start)(i) + codelens(i) )) {
			cost += values->w.col(i).sum() * gparent;
			values->dw.col(i).fill(gparent);
		}
	}
	return cost;
}

template double masked_sum(std::shared_ptr<Mat<double>>, uint&, const eigen_index_block&, const eigen_index_block&, double);
template double masked_sum(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, const eigen_index_vector&, double);
template double masked_sum(std::shared_ptr<Mat<double>>, uint&, const eigen_index_block&, const eigen_index_vector&, double);
template double masked_sum(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, const eigen_index_block&, double);
template double masked_sum(std::shared_ptr<Mat<double>>, uint&, shared_eigen_index_vector, const eigen_index_block&, double);
template double masked_sum(std::shared_ptr<Mat<double>>, uint&, const eigen_index_vector&, shared_eigen_index_vector, double);
template double masked_sum(std::shared_ptr<Mat<double>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, double);

template float masked_sum(std::shared_ptr<Mat<float>>, uint&, const eigen_index_block&, const eigen_index_block&, float);
template float masked_sum(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, const eigen_index_vector&, float);
template float masked_sum(std::shared_ptr<Mat<float>>, uint&, const eigen_index_block&, const eigen_index_vector&, float);
template float masked_sum(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, const eigen_index_block&, float);
template float masked_sum(std::shared_ptr<Mat<float>>, uint&, shared_eigen_index_vector, const eigen_index_block&, float);
template float masked_sum(std::shared_ptr<Mat<float>>, uint&, const eigen_index_vector&, shared_eigen_index_vector, float);
template float masked_sum(std::shared_ptr<Mat<float>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, float);


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
typename Graph<T>::shared_mat Graph<T>::eltmul_broadcast(
	shared_mat matrix1,
	shared_mat matrix2) {
	if (matrix1->n != matrix2->n || matrix2->d != 1)
		throw std::invalid_argument("Matrices cannot be element multiplied with broadcast, they do not have the same dimensions.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.array().colwise() * matrix2->w.col(0).array()).matrix();
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::eltmul_broadcast);
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
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::eltmul);
	return out;
}

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
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, row_vector, out, utils::ops::eltmul_broadcast_rowwise);
	return out;
}

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
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::eltmul_rowwise);
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
	if (this->needs_backprop)
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::add_broadcast);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add(
		shared_mat matrix1,
		shared_mat matrix2) {
	if (matrix1->d != matrix2->d && (matrix1->d == 1 || matrix2->d == 1)) {
		if (matrix1->d == 1)
			return add_broadcast(matrix2, matrix1);
		return add_broadcast(matrix1, matrix2);
	}
	if (matrix1->n != matrix2->n || matrix1->d != matrix2->d)
		throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
	auto out = std::make_shared<Mat<T>>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w + matrix2->w;
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::add);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::sigmoid(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::sigmoid_operator<T>());
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, utils::ops::sigmoid);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::transpose(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->d,
		matrix1->n,
		true);
	out->w = matrix1->w.transpose();
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, utils::ops::transpose);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::tanh(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::tanh_operator<T>());
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, utils::ops::tanh);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::relu(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::relu_operator<T>());
	if (this->needs_backprop)
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
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::mul);
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
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, bias, out, utils::ops::mul);
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
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, input_to_1, matrix2, input_to_2, bias, out, utils::ops::mul_add_broadcast_mul_with_bias);
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
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, input_to_1, matrix2, input_to_2, bias, out, utils::ops::mul_add_mul_with_bias);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::rows_pluck(
	shared_mat matrix1,
	index_std_vector& indices
	) {
	auto out = std::make_shared<mat>(
		matrix1->d,
		indices.size(),
		true);
	int offset = 0;
	for (auto& i : indices) {
		out->w.col(offset) = matrix1->w.row(i).transpose();
		++offset;
	}
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, indices, utils::ops::rows_pluck);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::rows_pluck(
	shared_mat matrix1,
	eigen_index_block indices
	) {
	auto out = std::make_shared<mat>(
		matrix1->d,
		indices.rows(),
		true);
	for (int offset = 0; offset < indices.rows(); ++offset) {
		out->w.col(offset) = matrix1->w.row(indices(offset)).transpose();
	}
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, indices, utils::ops::rows_pluck);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::row_pluck(
	shared_mat matrix1,
	int ix) {
	auto out = std::make_shared<mat>(
		matrix1->d,
		1,
		true);
	out->w = matrix1->w.row(ix).transpose();
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, ix, utils::ops::row_pluck);
	return out;
}


template<typename T>
Solver::SGD<T>::SGD (T _clipval) :
        clipval(_clipval) {};

template<typename T>
void Solver::SGD<T>::step (std::vector<typename Solver::SGD<T>::shared_mat>& parameters,
	T step_size,
	T regc
	) {
	for (auto& param : parameters) {
		// clip the gradient to prevent explosions:
		param->dw = (param->dw).array().min(clipval).max(-clipval).matrix();
		// update gradient using SGD rule
		param->w -= (step_size * param->dw.array()).matrix()  - (regc * param->w);
		// reset gradient
		param->dw.fill(0);
	}
}

template<typename T>
Solver::AdaDelta<T>::AdaDelta (
	T _rho,
	T _smooth_eps,
	T _clipval) :
		smooth_eps(_smooth_eps),
		rho(_rho),
        clipval(_clipval) {};

template<typename T>
Solver::AdaDelta<T>::AdaDelta (
	std::vector<typename Solver::AdaDelta<T>::shared_mat>& parameters,
	T _rho,
	T _smooth_eps,
	T _clipval) :
		smooth_eps(_smooth_eps),
		rho(_rho),
        clipval(_clipval) {
    create_gradient_caches(parameters);
};

template<typename T>
void Solver::AdaDelta<T>::create_gradient_caches(
	std::vector<typename Solver::AdaDelta<T>::shared_mat>& parameters) {
	for (auto& param : parameters) {
		// this operation should be run once unless
		// we expect the parameters of the model
		// to change online (probably not the case)
		if (!(gsums.count(*param) > 0)) {
			auto new_cache = this->gsums.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(*param),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);

			new_cache = this->xsums.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(*param),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);
		}
	}
}

template<typename T>
void Solver::AdaDelta<T>::step (std::vector<typename Solver::AdaDelta<T>::shared_mat>& parameters, T regc) {
	for (auto& param : parameters) {

		auto& gsum = gsums[*param];
		auto& xsum = xsums[*param];

		// clip the gradient to prevent explosions:
		param->dw = (param->dw).array().min(clipval).max(-clipval).matrix() + (regc * param->w);
		
		// update gradient cache using decay rule:
		gsum = gsum * rho + (1.0 - rho) * param->dw.array().square().matrix();

		auto dparam = -(((xsum.array() + smooth_eps) / (gsum.array() + smooth_eps).sqrt()) * param->dw.array()).matrix();

		xsum = xsum * rho + (1.0 - rho) * (dparam.array().square().matrix());
		// update gradient using AdaDelta rule
		param->w += dparam;
		// reset gradient
		param->dw.fill(0);
	}
}

template<typename T>
int argmax(std::shared_ptr<Mat<T>> A) {
	int i = 0;
	T current_max = -std::numeric_limits<T>::infinity();
	auto ptr = A->w.data();
	for (int j = 0; j < A->n * A->d; j++) {
		if (*ptr > current_max) {
			current_max = *ptr;
			i = j;
		}
		ptr++;
	}
	return i;
}

template int argmax(std::shared_ptr<Mat<float>>);
template int argmax(std::shared_ptr<Mat<double>>);

template<typename T>
Solver::AdaGrad<T>::AdaGrad (T _smooth_eps, T _clipval) : smooth_eps(_smooth_eps), clipval(_clipval) {}

template<typename T>
Solver::AdaGrad<T>::AdaGrad (std::vector<typename Solver::AdaGrad<T>::shared_mat>& parameters,
	T _smooth_eps,
	T _clipval) : smooth_eps(_smooth_eps), clipval(_clipval) {
	create_gradient_caches(parameters);
}

template<typename T>
void Solver::AdaGrad<T>::step(
	std::vector<typename Solver::AdaGrad<T>::shared_mat>& parameters,
	T step_size,
	T regc
	) {
	for (auto& param : parameters) {
		auto& s = gsums[*param];

		param->dw = (param->dw).array().min(clipval).max(-clipval).matrix() + (regc * param->w);
		// update gradient cache using decay rule:
		s += (param->dw).array().square().matrix();
		// clip the gradient to prevent explosions:
		// update gradient using RMSprop rule
		param->w -= step_size * (param->dw.array() / (s.array() + this->smooth_eps).sqrt() ).matrix();
		// reset gradient
		param->dw.fill(0);
	}
}

template<typename T>
void Solver::AdaGrad<T>::reset_caches(
	std::vector<typename Solver::AdaGrad<T>::shared_mat>& parameters) {
	for (auto& param : parameters) {
		auto& s = gsums[*param];
		s.fill(0);
	}
}

template<typename T>
void Solver::AdaGrad<T>::create_gradient_caches(
	std::vector<typename Solver::AdaGrad<T>::shared_mat>& parameters) {
	for (auto& param : parameters) {
		// this operation should be run once unless
		// we expect the parameters of the model
		// to change online (probably not the case)
		if (!(gsums.count(*param) > 0)) {
			auto new_cache = this->gsums.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(*param),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);
		}
	}
}

template<typename T>
Solver::RMSProp<T>::RMSProp (
            T _decay_rate,
            T _smooth_eps,
            T _clipval) :
        decay_rate(_decay_rate),
        smooth_eps(_smooth_eps),
        clipval(_clipval) {};

template<typename T>
Solver::RMSProp<T>::RMSProp (
			std::vector<typename Solver::RMSProp<T>::shared_mat>& parameters,
            T _decay_rate,
            T _smooth_eps,
            T _clipval) :
        decay_rate(_decay_rate),
        smooth_eps(_smooth_eps),
        clipval(_clipval)
        {
    create_gradient_caches(parameters);
};

template<typename T>
void Solver::RMSProp<T>::create_gradient_caches(
	std::vector<typename Solver::RMSProp<T>::shared_mat>& parameters) {
	for (auto& param : parameters) {
		// this operation should be run once unless
		// we expect the parameters of the model
		// to change online (probably not the case)
		if (!(gsums.count(*param) > 0)) {
			auto new_cache = this->gsums.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(*param),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);
		}
	}
}

template<typename T>
void Solver::RMSProp<T>::step(
	std::vector<typename Solver::RMSProp<T>::shared_mat>& parameters,
	T step_size,
	T regc
	) {
	for (auto& param : parameters) {
		auto& s = gsums[*param];
		// update gradient cache using decay rule:
		s = s * this->decay_rate + (1.0 - this->decay_rate) * param->dw.array().square().matrix();
		// clip the gradient to prevent explosions:
		param->dw = (param->dw).array().min(clipval).max(-clipval).matrix();
		// update gradient using RMSprop rule
		param->w -= step_size * (param->dw.array() / (s.array() + this->smooth_eps).sqrt() ).matrix()  - (regc * param->w);
		// reset gradient
		param->dw.fill(0);
	}
}

template class Mat<float>;
template class Mat<double>;

template class Backward<float>;
template class Backward<double>;

template class Graph<float>;
template class Graph<double>;

template class Solver::SGD<float>;
template class Solver::SGD<double>;

template class Solver::AdaGrad<float>;
template class Solver::AdaGrad<double>;

template class Solver::AdaDelta<float>;
template class Solver::AdaDelta<double>;

template class Solver::RMSProp<float>;
template class Solver::RMSProp<double>;

template std::shared_ptr<Mat<float>> softmax(std::shared_ptr<Mat<float>>);
template std::shared_ptr<Mat<double>> softmax(std::shared_ptr<Mat<double>>);

template float cross_entropy(std::shared_ptr<Mat<float>>, int&);
template double cross_entropy(std::shared_ptr<Mat<double>>, int&);
