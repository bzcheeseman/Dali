#ifndef RECURRENT_MAT_H
#define RECURRENT_MAT_H

#include "core/utils.h"
#include "core/cnpy.h"
#include <initializer_list>

// #define EIGEN_USE_MKL_VML
// #define EIGEN_USE_BLAS
// doesnt work, but its also not useful for now
// #define EIGEN_USE_LAPACKE
#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>


DECLARE_bool(eigen_parallel);

#define SMOOTH_DEFAULT 1e-6

// typedef Eigen::MatrixBase<Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> >::ColXpr eigen_index_block;
// typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> eigen_index_vector;
// typedef Eigen::MatrixWrapper<
//         Eigen::CwiseUnaryOp< Eigen::internal::scalar_add_op<unsigned int>, Eigen::ArrayWrapper<eigen_index_block> const > const > eigen_index_block_scalar;
// typedef std::vector<uint> index_std_vector;

#include "core/Index.h"

typedef std::size_t random_t;

/**
Mat
---

Main matrix class for this library. The Mat
class contains two pieces of memory, `w`
and `dw`. The first is the actual weights or
values associated with this matrix, and the
second is the local contribution to the
objective function (or ∂E/∂Mat). This local
contribution can then be used in
backpropagation.

Mat is used almost everywhere in the library
except in `utils`.

Note: ideally this class would generalize to
higher and lower dimensions. For instance
see `Graph::sum` and `Graph::mean`, or
`Mat::grad` methods to see where a Scalar
type would be useful (today this is a 1x1
matrix -- mathematically sound, but this
is inefficient, and inadequate for good
broadcasting across other operations).

**/
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
                Mat (int n, int d);
                Mat (int n, int d, bool empty);
                /**
                Mat<T>::Mat<T>
                --------------

                A copy constructor that perform shallow and deep
                copies of a Mat.

                Key usage is for Hogwild style training of parameters
                where different computation threads share memory for
                the parameters but each compute their own gradients.
                The gradients are kept in separate `dw` memory buffers
                but `w` buffers are shared amongst threads.

                For usage see `Solver::Adadelta`,
                              `examples/character_prediction.cpp`

                Inputs
                ------

                const Mat<T>& m : matrix to copy or point to
                    bool copy_w : whether matrix parameters should be copied over, or shared
                                  between matrices
                   bool copy_dw : whether matrix gradient parameters should be copied over,
                                  or shared (Note: it is unclear when `dw` should be shared,
                                  proceed with caution).

                Outputs
                -------

                Mat<T> out : deep or shallow copy of m
                **/
        Mat (const Mat<T>& m, bool copy_w, bool copy_dw);
                void print() const;

                /**
                Adds 1 to the gradient (`dw`) of every element in this Matrix as long
                as the matrix is 1x1 (a scalar).

                Useful for computing cross entropy, mean squared error, and other
                loss functions in vanilla ML fashion.

                **/
                void grad();

                ~Mat();
                /*
                Set Name
                --------

                Used for giving names to matrices for debugging or convenience purposes,
                but the names have no bearing on computation or identification in
                lookup tables;

                Inputs
                ------

                std::string& name : name the Mat should take on

                */
                void set_name(std::string& newname);
                /*
                Set Name
                --------
                See `Mat<T>::set_name` above
                */
                void set_name(char* newname);
                /*
                Set Name
                --------
                See `Mat<T>::set_name` above
                */
                void set_name(const char* newname);
                // random matrices:
                /*
                Mat<T>::Mat<T>
                --------------

                Matrix constructor using a zero mean
                normal distribution with a user provided
                standard deviation.

                Inputs
                ------

                int _n : number of rows
                int _d : number of columns
                 T std : standard deviation for normal distribution

                Outputs
                -------

                Mat<T> out : the matrix filled with random numbers from ~ N(0, std^2)

                See `Mat<T>::Mat(int, int, T, T)` for uniform distribution (below).
                */
                Mat (int n, int d, T std);
                /*
                Mat<T>::Mat<T>
                --------------

                Matrix constructor using a uniform
                distribution with user defined min
                and max support.

                Inputs
                ------

                  int _n : number of rows
                  int _d : number of columns
                 T lower : minimum of uniform distribution
                 T upper : maximum of uniform distribution

                Outputs
                -------

                Mat<T> out : the matrix filled with random numbers from ~U(lower, upper)

                See `Mat<T>::Mat(int, int, T)` for normal distribution (above)
                */
                Mat (int n, int d, T lower, T upper);
                void npy_save(std::string fname, std::string mode = "w");
                void npy_save(FILE*);
                void npy_load(std::string fname);
                void npy_load(FILE*);
                void npy_load(cnpy::NpyArray&);
                Mat (std::string fname);
                static Mat RandMat(int n, int d, T std);
                static Mat Empty(int n, int d);
                /**
                Shallow Copy
                ------------

                A copy constructor that perform shallow copies of a Mat.

                Key usage is for Hogwild style training of parameters
                where different computation threads share memory for
                the parameters but each compute their own gradients.
                The gradients are kept in separate `dw` memory buffers
                but `w` buffers are shared amongst threads.

                For usage see `Mat<T>::Mat<T>(const Mat&, bool, bool)`, `examples/character_prediction.cpp`

                Inputs
                ------

                const Mat<T>& m : matrix that will own the underlying memory
                                  for `w`

                Outputs
                -------

                Mat<T> out : shallow copy of m
                **/
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
bool operator!=(const Mat<T>&, const Mat<T>&);

template <typename T>
bool operator==(const Mat<T>&, const Mat<T>&);

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

                public:
                        std::unordered_map<mat, eigen_mat> gsums;
                        RMSProp (T decay_rate= 0.999, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
                        RMSProp (std::vector<shared_mat>&, T decay_rate= 0.999, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
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

                public:
                        std::unordered_map<mat, eigen_mat> gsums;
                        std::unordered_map<mat, eigen_mat> xsums;
                        AdaDelta (T rho= 0.95, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
                        AdaDelta (std::vector<shared_mat>&, T rho= 0.95, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
                        void step( std::vector<shared_mat>&, T);
                        void create_gradient_caches(std::vector<shared_mat>&);
        };

        template<typename T> class AdaGrad {
                T smooth_eps;
                T clipval;
                typedef Mat<T>                      mat;
                typedef std::shared_ptr<mat> shared_mat;
                typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

                public:
                        std::unordered_map<mat, eigen_mat> gsums;
                        AdaGrad (T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
                        AdaGrad (std::vector<shared_mat>&, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
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
                        SGD (std::vector<shared_mat>&, T clipval = 5.0);
                        void step( std::vector<shared_mat>&, T, T = 0.0);
        };
}
#endif
