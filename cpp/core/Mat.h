#ifndef RECURRENT_MAT_H
#define RECURRENT_MAT_H

#include "utils.h"
#include "cnpy.h"
#include <initializer_list>

// #define EIGEN_USE_MKL_VML
// #define EIGEN_USE_BLAS
// doesnt work, but its also not useful for now
// #define EIGEN_USE_LAPACKE
#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>

typedef Eigen::MatrixBase<Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> >::ColXpr eigen_index_block;
typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> eigen_index_vector;
typedef std::shared_ptr<eigen_index_vector> shared_eigen_index_vector;
typedef Eigen::MatrixWrapper<
	Eigen::CwiseUnaryOp< Eigen::internal::scalar_add_op<unsigned int>, Eigen::ArrayWrapper<eigen_index_block> const > const > eigen_index_block_scalar;
typedef std::vector<uint> index_std_vector;
typedef std::size_t random_t;

template<typename T> class Mat {
 	private:
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
		typedef Eigen::Map<eigen_mat> eigen_mat_view;
		eigen_mat _w;
		eigen_mat _dw;
 	public:
		int n; int d;
		mutable eigen_mat_view w;
		bool sparse;
		std::shared_ptr<std::vector<uint>> sparse_row_keys;
		mutable eigen_mat_view dw;
		std::shared_ptr<std::string> name = NULL;
		const random_t random_id;
		Mat (int, int);
		Mat (int, int, bool);
        Mat (const Mat<T>& m, bool copy_w, bool copy_dw);
		void print();
		~Mat();
		void set_name(std::string&);
		void set_name(char*);
		void set_name(const char*);
		// random matrices:
		Mat (int, int, T);
		Mat (int, int, T, T);
		void npy_save(std::string fname, std::string mode = "w");
		void npy_save(FILE*);
		void npy_load(std::string fname);
		void npy_load(FILE*);
		void npy_load(cnpy::NpyArray&);
		Mat (std::string fname);
		static Mat RandMat(int, int, T);
		static Mat Empty(int, int);
		static Mat shallow_copy(const Mat&);
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

namespace utils {
	template<typename T>
    void save_matrices(std::vector<std::shared_ptr<Mat<T>>>&, std::string);

    template<typename T>
    void load_matrices(std::vector<std::shared_ptr<Mat<T>>>&, std::string);
}

template <typename T>
static bool operator!=(const Mat<T>&, const Mat<T>&);

template <typename T>
static bool operator==(const Mat<T>&, const Mat<T>&);

template<typename T>
int argmax(std::shared_ptr<Mat<T>>);

template<typename T>
int argmax_slice(std::shared_ptr<Mat<T>>, int, int);

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
