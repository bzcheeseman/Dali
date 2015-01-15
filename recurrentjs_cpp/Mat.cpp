#include "utils.hpp"
// Matrix class (with random initializer):

template<typename T> class Mat {
	public:
		int n;
		int d;
		T * w;
		T * dw;

		Mat (int n, int d) {
			this->n = n;
			this->d = d;
			this->w = (T *) calloc(n * d, sizeof(T));
			this->dw = (T *) calloc(n * d, sizeof(T));
		}
		Mat (int n, int d, T* w) {
			this->n = n;
			this->d = d;
			this->w = w;
			this->dw = (T *) calloc(n * d, sizeof(T));
		}
		void print () {utils::print_matrix(this->n, this->d, this->w);}
		~Mat() {
			free(this->w);
			free(this->dw);
		}
		// random matrices:
		Mat (int n, int d, T std) {
			this->n = n;
			this->d = d;
			this->w = (T *) malloc(n * d * sizeof(T));
			utils::fill_random(n * d, std, this->w);
			this->dw = (T *) calloc(n * d, sizeof(T));
		}
		static Mat RandMat(int n, int d, T std) {
			return Mat(n, d, std);
		}

		friend std::ostream& operator<<(std::ostream& strm, const Mat& a) {
			return strm << "<#Mat n=" << a.n << ", d=" << a.d << " >";
		}
		operator std::string() const {
			return "<#Mat n=" << this->n << ", d=" << this->d << " >";
		}
};



template<typename T> class Backward {
	Mat<T> * matrix1;
	Mat<T> * matrix2;
	Mat<T> * out;
	uint type;
	public:

		Backward (Mat<T> * matrix1, Mat<T> * out, uint type) {
			this->matrix1 = matrix1;
			this->out = out;
			this->type = type;
		}
		Backward (Mat<T> * matrix1, Mat<T> * matrix2, Mat<T> * out, uint type) {
			this->matrix1 = matrix1;
			this->matrix2 = matrix2;
			this->out = out;
			this->type = type;
		}

		friend std::ostream& operator<<(std::ostream& strm, const Backward<T>& a) {
			return strm << "<#Backward matrix = " << *(a.matrix1) << ", out = " << *(a.out) << ">";
		}
		operator std::string() const {
			return "<#Backward matrix = " << *(this->matrix1) << ", out = " << *(this->out) << ">";
		}
};

template<typename T> class Graph {
	bool needs_backprop;
	std::vector<Backward<T> > backprop;

	public:

		Graph (bool needs_backprop) {
			this->needs_backprop = needs_backprop;
		}
		Graph () {
			this->needs_backprop = true;
		}
		void backward () {
			// std::vector<int> myvector;
			// for (int i=1; i<=5; i++) myvector.push_back(i);
			std::cout << "myvector contains (reverse):";
			for (typename std::vector<Backward<T> >::reverse_iterator it = this->backprop.rbegin(); it != this->backprop.rend(); ++it) {
				std::cout << ' ' << *it;
			}
			std::cout << '\n';
		}
		Mat<T>* eltmul(Mat<T>& matrix1, Mat<T>& matrix2) {
			if (matrix1.n != matrix2.n || matrix1.d != matrix2.d) {
				throw std::invalid_argument("Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
			}
			Mat<T>* out = new Mat<T>(
				matrix1.n,
				matrix1.d,
				utils::element_mult(
					matrix1.n * matrix1.d,
					matrix1.w,
					matrix2.w)
				);

			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(&matrix1, &matrix2, out, utils::ops::eltmul);
			}
		}
		Mat<T>* add(Mat<T>& matrix1, Mat<T>& matrix2) {
			if (matrix1.n != matrix2.n || matrix1.d != matrix2.d) {
				throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
			}
			Mat<T>* out = new Mat<T>(
				matrix1.n,
				matrix1.d,
				utils::element_sum(
					matrix1.n * matrix1.d,
					matrix1.w,
					matrix2.w)
				);

			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(&matrix1, out, utils::ops::add);
				this->backprop.emplace_back(&matrix2, out, utils::ops::add);
			}

			return out;
		}

		Mat<T>* mul(Mat<T>& matrix1, Mat<T>& matrix2) {
			if (matrix1.d != matrix2.n) {
				throw std::invalid_argument("matmul dimensions misaligned.");
			}
			// TODO: use eigen for these matrix operations:
			Mat<T>* out = new Mat<T>(
				matrix1.n,
				matrix1.d,
				utils::matrix_dot_product(
					matrix1.n * matrix1.d,
					matrix1.w,
					matrix2.w)
				);

			if (this->needs_backprop) {
				// allocates a new backward element in the vector using these arguments:
				this->backprop.emplace_back(&matrix1, &matrix2, out, utils::ops::mul);
			}

			return out;
		}
};

typedef double REAL_t;

int main() {
	// build blank matrix of double type:
    Mat<REAL_t> mymatrix(3,5);
    // build random matrix of REAL_t type with standard deviation 2:
    Mat<REAL_t> mymatrix2 = Mat<REAL_t>::RandMat(3, 5, 2.0);
    mymatrix2.print();
    mymatrix.print();

    Graph<REAL_t> mygraph;

    std::unique_ptr< Mat<REAL_t> > thesum( mygraph.add(mymatrix, mymatrix2) );

    thesum->print();

    mygraph.backward();

    return 0;
}