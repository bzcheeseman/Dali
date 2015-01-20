#ifndef RECURRENT_MAT_H
#define RECURRENT_MAT_H


#include "utils.h"
#include <Eigen>

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
	std::vector<int>* indices;
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
		Backward(shared_mat, shared_mat, std::vector<int>&, uint);
		Backward(shared_mat, shared_mat, shared_mat, uint);
		Backward(shared_mat, shared_mat, shared_mat, shared_mat, uint);
		Backward(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat, uint);
		Backward(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat, shared_mat, uint);
		
		operator std::string() const;
		std::string op_type () const;
		void operator ()();
};

template<typename T>
std::ostream& operator<<(std::ostream&, const Backward<T>&);

template<typename T> std::shared_ptr<Mat<T>> softmax(std::shared_ptr<Mat<T>>);
template<typename T> T cross_entropy(std::shared_ptr<Mat<T>>, int&);

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
		shared_mat mul_with_bias(shared_mat, shared_mat, shared_mat);
		// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
		shared_mat mul_add_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
		// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
		// and with caveat that x is actually a column, and should be broadcasted
		shared_mat mul_add_broadcast_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
		shared_mat add_broadcast(shared_mat, shared_mat);
		shared_mat add(shared_mat, shared_mat);
		shared_mat sigmoid(shared_mat);
		shared_mat tanh(shared_mat);
		shared_mat relu(shared_mat);
		shared_mat mul(shared_mat, shared_mat);
		shared_mat rows_pluck(shared_mat, std::vector<int>);
		shared_mat row_pluck(shared_mat, int);
};

#include <unordered_map>
template<typename T> class Solver {
	T decay_rate;
	T smooth_eps;
	T clipval;
	typedef Mat<T>                      mat;
	typedef std::shared_ptr<mat> shared_mat;
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
	std::unordered_map<mat, eigen_mat> step_cache;
	utils::clip_operator<T> clip_values;
	utils::squared_operator<T> square_values;
	public:
		Solver (T decay_rate= 0.999, T smooth_eps =1e-8, T clipval = 5.0);
		Solver (std::vector<shared_mat>&, T decay_rate= 0.999, T smooth_eps =1e-8, T clipval = 5.0);
		void step( std::vector<shared_mat>&, T, T);
		void create_gradient_caches(std::vector<shared_mat>&);
		~Solver ();
};

#endif