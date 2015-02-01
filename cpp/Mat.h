#ifndef RECURRENT_MAT_H
#define RECURRENT_MAT_H

#include "utils.h"
#include "cnpy.h"

#define EIGEN_USE_MKL_VML
#define EIGEN_USE_BLAS
// doesnt work, but its also not useful for now
// #define EIGEN_USE_LAPACKE
#include <Eigen>

typedef Eigen::MatrixBase<Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> >::ColXpr eigen_index_block;
typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> eigen_index_vector;
typedef std::shared_ptr<eigen_index_vector> shared_eigen_index_vector;
typedef Eigen::MatrixWrapper<
	Eigen::CwiseUnaryOp< Eigen::internal::scalar_add_op<unsigned int>, Eigen::ArrayWrapper<eigen_index_block> const > const > eigen_index_block_scalar;
typedef std::vector<uint> index_std_vector;
// typedef Eigen::MatrixWrapper<Eigen::CwiseUnaryOp<Eigen::internal::scalar_add_op<unsigned int>, Eigen::ArrayWrapper<Eigen::Block<Eigen::Matrix<unsigned int, -1, -1, 0, -1, -1>, -1, 1, true> >>> eigen_scalar_add_block_wrapper;

template<typename T> class Mat {
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
	public:
		int n; int d;
		eigen_mat w;
		eigen_mat dw;
		const std::size_t random_id;
		Mat (int, int);
		Mat (int, int, bool);
		void print();
		~Mat();
		// random matrices:
		Mat (int, int, T);
		Mat (int, int, T, T);
		void npy_save(std::string fname, std::string mode = "w");
		Mat (std::string fname);
		static Mat RandMat(int, int, T);
		static Mat Empty(int, int);
		operator std::string() const;
};

template<typename T>
std::ostream& operator<<(std::ostream&, const Mat<T>&);

// define hash code for matrices:
namespace std {
	template <typename T> struct hash<Mat<T>> {
		std::size_t operator()(const Mat<T>&) const;
	};
}

template <typename T>
static bool operator!=(const Mat<T>&, const Mat<T>&);

template <typename T>
static bool operator==(const Mat<T>&, const Mat<T>&);

template<typename T> class Backward {
	int ix;
	uint* indices;
	int num_indices;
	uint type;
	typedef Mat<T> mat;
	typedef std::shared_ptr<mat> shared_mat;
	void backward_rows_pluck();
	public:
		shared_mat matrix1;
		shared_mat matrix2;
		shared_mat matrix3;
		shared_mat matrix4;
		shared_mat matrix5;

		shared_mat out;
		Backward(shared_mat, shared_mat, uint);
		Backward(shared_mat, shared_mat, int, uint);
		Backward(shared_mat, shared_mat, index_std_vector&, uint);
		Backward(shared_mat, shared_mat, eigen_index_block, uint);
		Backward(shared_mat, shared_mat, shared_mat, uint);
		Backward(shared_mat, shared_mat, shared_mat, shared_mat, uint);
		Backward(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat, uint);
		Backward(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat, shared_mat, uint);
		
		operator std::string() const;
		std::string op_type () const;
		void operator ()();
};

template<typename T>
int argmax(std::shared_ptr<Mat<T>>);

template<typename T>
std::ostream& operator<<(std::ostream&, const Backward<T>&);

template<typename T> std::shared_ptr<Mat<T>> softmax(std::shared_ptr<Mat<T>>);
template<typename T> T cross_entropy(std::shared_ptr<Mat<T>>, int&);
template<typename T, typename M> T cross_entropy(std::shared_ptr<Mat<T>>, const M);

template<typename T, typename M, typename K, typename F> T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, const K&, const F&, const M);
template<typename T, typename M, typename K> T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, const K&, shared_eigen_index_vector, const M);
template<typename T, typename M, typename F> T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, const F&, const M);
template<typename T, typename M> T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const M);

template<typename T, typename K, typename F> T masked_sum(std::shared_ptr<Mat<T>>, uint&, const K&, const F&, T);
template<typename T, typename K> T masked_sum(std::shared_ptr<Mat<T>>, uint&, const K&, shared_eigen_index_vector, T);
template<typename T, typename F> T masked_sum(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, const F&, T);
template<typename T> T masked_sum(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, T);

template<typename T> class Graph {
	bool                     needs_backprop;
	std::vector<Backward<T>>       backprop;
	typedef Mat<T>                      mat;
	typedef std::shared_ptr<mat> shared_mat;
	public:
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
		// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
		// and with caveat that x is actually a column, and should be broadcasted
		shared_mat mul_add_broadcast_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
		shared_mat add_broadcast(shared_mat, shared_mat);
		shared_mat add(shared_mat, shared_mat);
		shared_mat sigmoid(shared_mat);
		shared_mat transpose(shared_mat);
		shared_mat tanh(shared_mat);
		shared_mat relu(shared_mat);
		shared_mat mul(shared_mat, shared_mat);
		shared_mat rows_pluck(shared_mat, index_std_vector&);
		shared_mat rows_pluck(shared_mat, eigen_index_block);
		shared_mat row_pluck(shared_mat, int);
};
#include <unordered_map>
namespace Solver {

	template<typename T> class RMSProp {
		T decay_rate;
		T smooth_eps;
		T clipval;
		typedef Mat<T>                      mat;
		typedef std::shared_ptr<mat> shared_mat;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
		std::unordered_map<mat, eigen_mat> gsums;
		public:
			RMSProp (T decay_rate= 0.999, T smooth_eps =1e-8, T clipval = 5.0);
			RMSProp (std::vector<shared_mat>&, T decay_rate= 0.999, T smooth_eps =1e-8, T clipval = 5.0);
			void step( std::vector<shared_mat>&, T, T);
			void create_gradient_caches(std::vector<shared_mat>&);
	};

	template<typename T> class AdaDelta {
		T rho;
		T smooth_eps;
		T clipval;
		typedef Mat<T>                      mat;
		typedef std::shared_ptr<mat> shared_mat;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
		std::unordered_map<mat, eigen_mat> gsums;
		std::unordered_map<mat, eigen_mat> xsums;
		public:
			AdaDelta (T rho= 0.95, T smooth_eps =1e-8, T clipval = 5.0);
			AdaDelta (std::vector<shared_mat>&, T rho= 0.95, T smooth_eps =1e-8, T clipval = 5.0);
			void step( std::vector<shared_mat>&, T);
			void create_gradient_caches(std::vector<shared_mat>&);
	};

	template<typename T> class AdaGrad {
		T smooth_eps;
		T clipval;
		typedef Mat<T>                      mat;
		typedef std::shared_ptr<mat> shared_mat;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
		std::unordered_map<mat, eigen_mat> gsums;
		public:
			AdaGrad (T smooth_eps =1e-8, T clipval = 5.0);
			AdaGrad (std::vector<shared_mat>&, T smooth_eps =1e-8, T clipval = 5.0);
			void step( std::vector<shared_mat>&, T, T);
			void reset_caches( std::vector<shared_mat>&);
			void create_gradient_caches(std::vector<shared_mat>&);
	};

	template<typename T> class SGD {
		T clipval;
		typedef Mat<T>                      mat;
		typedef std::shared_ptr<mat> shared_mat;
		public:
			SGD (T clipval = 5.0);
			void step( std::vector<shared_mat>&, T, T);
	};
}
#endif