#ifndef RECURRENT_MAT_H
#define RECURRENT_MAT_H

#include "utils.hpp"
#include <Eigen>

using namespace Eigen;

template<typename T> class Mat {
	typedef Matrix<T, Dynamic, Dynamic> eigen_mat;
	public:
		int n; int d;
		eigen_mat w;
		eigen_mat dw;
		const std::size_t random_id;
		Mat (int _n, int _d) : n(_n), d(_d), random_id(utils::get_random_id()) {
			// this->random_id = utils::get_random_id();
			this->w = eigen_mat::Zero(n,d);
			this->dw = eigen_mat::Zero(n,d);
		}
		Mat (int _n, int _d, bool empty) : n(_n), d(_d), random_id(utils::get_random_id()) {
			// this->random_id = utils::get_random_id();
			this->w  = empty ? eigen_mat(n,d) : eigen_mat::Zero(n,d);
			this->dw = eigen_mat::Zero(n,d);
		}
		Mat (int _n, int _d, T* w) : n(_n), d(_d), random_id(utils::get_random_id()) {
			// this->random_id = utils::get_random_id();
			this->w = w;
			this->dw = eigen_mat::Zero(n,d);
		}
		void print () {
			utils::print_matrix(this->n, this->d, &(this->w(0)));// std::cout << this->w << std::endl;
		}
		~Mat() {}
		// random matrices:
		Mat (int _n, int _d, T std) : n(_n), d(_d), random_id(utils::get_random_id()) {
			// this->random_id = utils::get_random_id();
			std::default_random_engine generator;
			std::normal_distribution<T> distribution(0.0, std);
			std::random_device rd;
			generator.seed(rd());
			auto randn = [&] (int) {return distribution(generator);};
			this->w = eigen_mat::NullaryExpr(n,d, randn);
			this->dw = eigen_mat(n,d);
		}
		static Mat RandMat(int n, int d, T std) {
			// is in fact using C++ 11 's rvalue, move operator,
			// so no copy is made.
			return Mat(n, d, std);
		}
		static Mat Empty(int n, int d) {
			// use an empty matrix and modify
			// it so as to not incur the filling
			// with zeros cost.
			return Mat(n, d, true);
		}
		friend std::ostream& operator<<(std::ostream& strm, const Mat& a) {
			return strm << "<#Mat n=" << a.n << ", d=" << a.d << ">";
		}
		operator std::string() const {
			return "<#Mat n=" << this->n << ", d=" << this->d << ">";
		}
};

// define hash code for matrices:
namespace std {
	template <typename T> struct hash<Mat<T>> {
		std::size_t operator()(const Mat<T>& k) const {
			return k.random_id;
		}
	};
}

template <typename T>
static bool operator!=(const Mat<T>& matrix1, const Mat<T>& matrix2) {
    return matrix2.random_id != matrix1.random_id;
}

template <typename T>
static bool operator==(const Mat<T>& matrix1, const Mat<T>& matrix2) {

    return matrix2.random_id == matrix1.random_id;
}

template<typename T> class Backward {
	int ix;
	uint type;
	typedef Mat<T> mat;
	typedef std::shared_ptr<mat> shared_mat;
	shared_mat matrix1;
	shared_mat matrix2;
	shared_mat out;
	public:
		Backward (
			shared_mat matrix1,
			shared_mat out,
			uint type) {
			this->matrix1 = matrix1;
			this->matrix2 = NULL;
			this->out = out;
			this->type = type;
		}
		Backward (
			shared_mat matrix1,
			shared_mat out,
			int index,
			uint type) {
			this->matrix1 = matrix1;
			this->matrix2 = NULL;
			this->out = out;
			this->type = type;
			this->ix = index;
		}
		Backward (
			shared_mat matrix1,
			shared_mat matrix2,
			shared_mat out,
			uint type) {
			this->matrix1 = matrix1;
			this->matrix2 = matrix2;
			this->out = out;
			this->type = type;
		}
		friend std::ostream& operator<<(std::ostream& strm, const Backward<T>& a) {
			if (a.matrix2 != NULL) {
				return strm << "<#Backward matrix1=" << *(a.matrix1) << ", matrix2=" << *(a.matrix2) << ", out=" << *(a.out) << ", type=\""<< a.op_type() << "\">";
			}
			return strm << "<#Backward matrix1=" << *(a.matrix1) << ", out=" << *(a.out) << ", type=\""<< a.op_type() << "\">";
		}
		operator std::string() const {
			if (this->matrix2 != NULL) {
				return "<#Backward matrix1=" << *(this->matrix1) << ", matrix2=" << *(this->matrix2) << ", out=" << *(this->out) << ", type =\""<< this->op_type() << "\">";
			}
			return "<#Backward matrix1=" << *(this->matrix1) << ", out=" << *(this->out) << ", type =\""<< this->op_type() << "\">";
		}
		std::string op_type () const {
			switch(this->type) {
				case utils::ops::add:
					return "add";
					break;
				case utils::ops::eltmul:
					return "eltmul";
					break;
				case utils::ops::tanh:
					return "tanh";
					break;
				case utils::ops::sigmoid:
					return "sigmoid";
					break;
				case utils::ops::relu:
					return "relu";
					break;
				case utils::ops::mul:
					return "mul";
					break;
				case utils::ops::row_pluck:
					return "row_pluck";
					break;
				default:
					return "?";
					break;
			}
		}
		void operator ()() {
			switch(this->type) {
				case utils::ops::add:
					this->matrix1->dw += this->out->dw;
					this->matrix2->dw += this->out->dw;
					break;
				case utils::ops::eltmul:
					this->matrix1->dw += ((this->matrix2->w).array() * (this->out->dw).array()).matrix();
					this->matrix2->dw += ((this->matrix1->w).array() * (this->out->dw).array()).matrix();
					break;
				case utils::ops::sigmoid:
					this->matrix1->dw += (((this->out->w).array() * (1.0 - this->out->w.array())) * this->out->dw.array()).matrix();
					break;
				case utils::ops::mul:
					this->matrix1->dw += (this->out->dw) * ((this->matrix2->w).transpose());
					this->matrix2->dw += this->matrix1->w.transpose() * (this->out->dw);
					break;
				case utils::ops::relu:
					this->matrix1->dw += (this->out->w.unaryExpr(utils::sign_operator<T>()).array() * this->out->dw.array()).matrix();
					break;
				case utils::ops::tanh:
					this->matrix1->dw += (this->out->w.unaryExpr(utils::dtanh_operator<T>()).array() * this->out->dw.array()).matrix();
					break;
				case utils::ops::row_pluck:
					this->matrix1->dw.row(this->ix) += (this->out->w.array() * this->out->dw.array()).matrix().col(0).transpose();
					break;
				default:
					throw std::invalid_argument("NotImplemented: Do not know how to backpropagate for this type");
					break;
			}
		}
};

template<typename T>
std::shared_ptr<Mat<T>> softmax(std::shared_ptr<Mat<T>> matrix) {
	T layer_max = matrix->w.array().maxCoeff();
	Array<T, Dynamic, Dynamic> exped_distributions = (matrix->w.array() - layer_max).unaryExpr(utils::exp_operator<T>());
	T total_distribution = exped_distributions.sum();
	auto out = std::make_shared<Mat<T>>(
		matrix->n,
		matrix->d,
		true);
	out->w = (exped_distributions / total_distribution).matrix();
	return out;
}

template<typename T> class Graph {
	bool                     needs_backprop;
	std::vector<Backward<T>>       backprop;
	typedef Mat<T>                      mat;
	typedef std::shared_ptr<mat> shared_mat;
	public:
		Graph (bool needs_backprop) {this->needs_backprop = needs_backprop;}
		Graph () {this->needs_backprop = true;}
		void backward () {
			std::cout << "Graph backprop vector contains (in reverse):\n";
			for (auto it = this->backprop.rbegin(); it != this->backprop.rend(); ++it) {
				std::cout << ' ' << *it << "\n";
				(*it)();
			}
		}
		shared_mat eltmul(
			shared_mat matrix1,
			shared_mat matrix2) {
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
		shared_mat add(
				shared_mat matrix1,
				shared_mat matrix2) {
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
		shared_mat sigmoid(shared_mat matrix1) {
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
		shared_mat tanh(shared_mat matrix1) {
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
		shared_mat relu(shared_mat matrix1) {
			auto out = std::make_shared<mat>(
				matrix1->n,
				matrix1->d,
				true);
			out->w = matrix1.w.unaryExpr(utils::relu_operator<T>());
			if (this->needs_backprop)
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(matrix1, out, utils::ops::relu);
			return out;
		}
		shared_mat mul(
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
		shared_mat row_pluck(
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
};

#include <unordered_map>
template<typename T> class Solver {
	T decay_rate;
	T smooth_eps;
	T clipval;
	typedef Mat<T>                      mat;
	typedef std::shared_ptr<mat> shared_mat;
	typedef Matrix<T, Dynamic, Dynamic> eigen_mat;
	std::unordered_map<mat, eigen_mat> step_cache;
	utils::clip_operator<T> clip_values;
	utils::squared_operator<T> square_values;
	public:
		Solver (
			T _decay_rate = 0.999,
			T _smooth_eps = 1e-8,
			T _clipval = 5.0) :
		decay_rate(_decay_rate),
		smooth_eps(_smooth_eps),
		clipval(_clipval),
		clip_values(-_clipval, _clipval) {};

		void step(
			std::vector<shared_mat>& model,
			T step_size,
			T regc
			) {
			for (auto& param : model) {
				// this operation should be run once unless
				// we expect the parameters of the model
				// to change online (probably not the case)
				if (!(this->step_cache.count(*param) > 0)) {
					auto new_cache = step_cache.emplace(
						std::piecewise_construct,
			              std::forward_as_tuple(*param),
			              std::forward_as_tuple(param->n, param->d));
					// initialize values for step cache to zero:
					new_cache.first->second.fill(0);
				}
				auto& s = step_cache[*param];
				// update gradient cache using decay rule:
				s = s * this->decay_rate + (1.0 - this->decay_rate) * param->dw.unaryExpr(square_values);
				// clip the gradient to prevent explosions:
				param->dw = (param->dw).array().unaryExpr(clip_values).matrix();
				// update gradient using RMSprop rule
				param->w += step_size * (param->dw.array() / (s.array() + this->smooth_eps).sqrt() ).matrix()  - (regc * param->w);
				// reset gradient
				param->dw.fill(0);
			}
		}
		~Solver () {};
};

#endif