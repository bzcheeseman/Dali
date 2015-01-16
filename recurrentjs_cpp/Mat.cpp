#include "utils.hpp"
#include <Eigen>
using namespace Eigen;
template<typename T> class Mat {
	public:
		int n; int d;
		Matrix<T, Dynamic, Dynamic> w;
		Matrix<T, Dynamic, Dynamic> dw;

		Mat (int _n, int _d) : n(_n), d(_d) {
			this->w = Matrix<T, Dynamic, Dynamic>::Zero(n,d);
			this->dw = Matrix<T, Dynamic, Dynamic>::Zero(n,d);
		}

		Mat (int _n, int _d, bool empty) : n(_n), d(_d) {
			if (empty) {
				this->w = Matrix<T, Dynamic, Dynamic>(n,d);
				this->dw = Matrix<T, Dynamic, Dynamic>(n,d);
			} else {
				this->w = Matrix<T, Dynamic, Dynamic>::Zero(n,d);
				this->dw = Matrix<T, Dynamic, Dynamic>::Zero(n,d);
			}
		}

		Mat (int _n, int _d, T* w) : n(_n), d(_d) {
			this->w = w;
			this->dw = Matrix<T, Dynamic, Dynamic>::Zero(n,d);
		}
		void print () {
			utils::print_matrix(this->n, this->d, &(this->w(0)));// std::cout << this->w << std::endl;
		}
		~Mat() {}
		// random matrices:
		Mat (int _n, int _d, T std) : n(_n), d(_d) {
			std::default_random_engine generator;
			std::normal_distribution<T> distribution(0.0, std);
			std::random_device rd;
			generator.seed(rd());
			auto randn = [&] (int) {return distribution(generator);};
			this->w = Matrix<T, Dynamic, Dynamic>::NullaryExpr(n,d, randn);
			this->dw = Matrix<T, Dynamic, Dynamic>(n,d);
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
			return strm << "<#Mat n=" << a.n << ", d=" << a.d << " >";
		}
		operator std::string() const {
			return "<#Mat n=" << this->n << ", d=" << this->d << " >";
		}
};

template<typename T> class Backward {
	std::shared_ptr< Mat<T> > matrix1;
	std::shared_ptr< Mat<T> > matrix2;
	std::shared_ptr< Mat<T> > out;
	int ix;
	uint type;
	public:
		Backward (
			std::shared_ptr< Mat<T> > matrix1,
			std::shared_ptr< Mat<T> > out,
			uint type) {
			this->matrix1 = matrix1;
			this->matrix2 = NULL;
			this->out = out;
			this->type = type;
		}
		Backward (
			std::shared_ptr< Mat<T> > matrix1,
			std::shared_ptr< Mat<T> > out,
			int index,
			uint type) {
			this->matrix1 = matrix1;
			this->matrix2 = NULL;
			this->out = out;
			this->type = type;
			this->ix = index;
		}
		Backward (
			std::shared_ptr< Mat<T> > matrix1,
			std::shared_ptr< Mat<T> > matrix2,
			std::shared_ptr< Mat<T> > out,
			uint type) {
			this->matrix1 = matrix1;
			this->matrix2 = matrix2;
			this->out = out;
			this->type = type;
		}

		friend std::ostream& operator<<(std::ostream& strm, const Backward<T>& a) {
			if (a.matrix2 != NULL) {
				return strm << "<#Backward matrix1 = " << *(a.matrix1) << ", matrix2 = " << *(a.matrix2) << ", out = " << *(a.out) << ", type = "<< a.op_type() << ">";
			} else {
				return strm << "<#Backward matrix = " << *(a.matrix1) << ", out = " << *(a.out) << ", type = "<< a.op_type() << ">";
			}
			
		}
		operator std::string() const {
			if (this->matrix2 != NULL) {
				return "<#Backward matrix = " << *(this->matrix1) << ", matrix2 = " << *(this->matrix2) << ", out = " << *(this->out) << ", type = "<< this->op_type() << ">";
			} else {
				return "<#Backward matrix = " << *(this->matrix1) << ", out = " << *(this->out) << ", type = "<< this->op_type() << ">";
			}
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
std::shared_ptr< Mat<T> > softmax(std::shared_ptr< Mat<T> > matrix) {
	T layer_max = matrix->w.array().maxCoeff();
	Array<T, Dynamic, Dynamic> exped_distributions = (matrix->w.array() - layer_max).unaryExpr(utils::exp_operator<T>());
	T total_distribution = exped_distributions.sum();
	std::shared_ptr< Mat<T> > out(new Mat<T>(
		matrix->n,
		matrix->d,
		true));
	out->w = (exped_distributions / total_distribution).matrix();
	return out;
}

template<typename T> class Graph {
	bool needs_backprop;
	std::vector<Backward<T> > backprop;

	public:
		Graph (bool needs_backprop) {this->needs_backprop = needs_backprop;}
		Graph () {this->needs_backprop = true;}
		void backward () {
			std::cout << "myvector contains (reverse):\n";
			for (typename std::vector<Backward<T> >::reverse_iterator it = this->backprop.rbegin(); it != this->backprop.rend(); ++it) {
				std::cout << ' ' << *it << "\n";
				(*it)();
			}
			std::cout << '\n';
		}
		std::shared_ptr< Mat<T> > eltmul(
			std::shared_ptr< Mat<T> > matrix1,
			std::shared_ptr< Mat<T> > matrix2) {
			if (matrix1->n != matrix2->n || matrix1->d != matrix2->d) {
				throw std::invalid_argument("Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
			}
			std::shared_ptr< Mat<T> > out( new Mat<T>(
				matrix1->n,
				matrix1->d,
				true));
			out->w = (matrix1->w.array() * matrix2->w.array()).matrix();
			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::eltmul);
			}
			return out;
		}
		std::shared_ptr< Mat<T> > add(
				std::shared_ptr< Mat<T> > matrix1,
				std::shared_ptr< Mat<T> > matrix2) {
			if (matrix1->n != matrix2->n || matrix1->d != matrix2->d) {
				throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
			}
			std::shared_ptr< Mat<T> > out(new Mat<T>(
				matrix1->n,
				matrix1->d,
				true));
			out->w = matrix1->w + matrix2->w;
			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::add);
			}
			return out;
		}
		std::shared_ptr< Mat<T> > sigmoid(std::shared_ptr< Mat<T> > matrix1) {
			std::shared_ptr< Mat<T> > out(new Mat<T>(matrix1->n, matrix1->d, true));
			out->w = matrix1->w.unaryExpr(utils::sigmoid_operator<T>());
			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(matrix1, out, utils::ops::sigmoid);
			}
			return out;
		}
		std::shared_ptr< Mat<T> > tanh(std::shared_ptr< Mat<T> > matrix1) {
			std::shared_ptr< Mat<T> > out(new Mat<T>(matrix1->n, matrix1->d, true));
			out->w = matrix1->w.unaryExpr(utils::tanh_operator<T>());
			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(matrix1, out, utils::ops::tanh);
			}
			return out;
		}
		std::shared_ptr< Mat<T> > relu(std::shared_ptr< Mat<T> > matrix1) {
			std::shared_ptr< Mat<T> > out(new Mat<T>(
				matrix1->n,
				matrix1->d,
				true));
			out->w = matrix1.w.unaryExpr(utils::relu_operator<T>());
			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(matrix1, out, utils::ops::relu);
			}
			return out;
		}
		std::shared_ptr< Mat<T> > mul(
			std::shared_ptr< Mat<T> > matrix1,
			std::shared_ptr< Mat<T> > matrix2) {
			if (matrix1->d != matrix2->n) {
				throw std::invalid_argument("matmul dimensions misaligned.");
			}
			std::shared_ptr< Mat<T> > out(new Mat<T>(
				matrix1->n,
				matrix2->d,
				true));
			out->w = matrix1->w * matrix2->w;
			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::mul);
			}
			return out;
		}
		std::shared_ptr< Mat<T> > row_pluck(
			std::shared_ptr< Mat<T> > matrix1,
			int ix) {
        
			std::shared_ptr< Mat<T> > out( new Mat<T>(
				matrix1->d,
				1,
				true));
			out->w = matrix1->w.row(ix).transpose();
			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(matrix1, out, ix, utils::ops::row_pluck);
			}
			return out;
		}
};

typedef double REAL_t;

template<typename T>
void forward_model(Graph<T>& graph,
	std::shared_ptr< Mat<T> > input,
	std::shared_ptr< Mat<T> > classifier) {
	std::shared_ptr< Mat<T> > prod( graph.mul(input, classifier) );
	std::shared_ptr< Mat<T> > activ( graph.tanh(prod) );
}

int main() {
	// build blank matrix of double type:
    std::shared_ptr< Mat<REAL_t>> A(new Mat<REAL_t>(3, 5) );
    A->w = (A->w.array() + 1.2).matrix();

    // build random matrix of double type with standard deviation 2:
    std::shared_ptr< Mat<REAL_t> > B(new Mat<REAL_t>(A->n, A->d, 2.0));
    std::shared_ptr< Mat<REAL_t> > C(new Mat<REAL_t>(A->d, 4, 2.0));

    A->print();
    B->print();

    Graph<REAL_t> graph;
	std::shared_ptr< Mat<REAL_t> > A_plus_B     = graph.add(A, B);
	std::shared_ptr< Mat<REAL_t> > A_times_B    = graph.eltmul(A, B);
	std::shared_ptr< Mat<REAL_t> > A_plus_B_sig = graph.sigmoid(A_plus_B);
	std::shared_ptr< Mat<REAL_t> > A_dot_C      = graph.mul(A, C);

    std::shared_ptr< Mat<REAL_t> > A_dot_C_tanh( graph.tanh(A_dot_C) );

    A_plus_B    ->print();
    A_times_B   ->print();
    A_plus_B_sig->print();
    A_dot_C     ->print();

    std::shared_ptr< Mat<REAL_t> > A_plucked(graph.row_pluck(A, 2));
    A_plucked->print();
    forward_model(graph, A, C);

    std::shared_ptr< Mat<REAL_t> > prod(graph.mul(A, C));
	std::shared_ptr< Mat<REAL_t> > activ(graph.tanh(prod));

    // add some random singularity and use exponential
    // normalization:
    A_plucked->w(2,0) += 3.0;
    std::shared_ptr< Mat<REAL_t> > A_plucked_normed(softmax(A_plucked));
    A_plucked_normed->print();

    // backpropagate to A and B
    graph.backward();

    return 0;
}