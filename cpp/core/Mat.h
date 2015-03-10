#ifndef CORE_MAT_H
#define CORE_MAT_H

#include <Eigen/Eigen>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <functional>

#include "core/cnpy.h"
#include "core/Index.h"
#include "core/Tape.h"
#include "core/utils.h"
#include "core/MatOps.h"

#define EPS 1e-9

typedef unsigned int dim_t;

template<typename R>
class MatInternal {
    typedef Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
    typedef Eigen::Map<eigen_mat> eigen_mat_view;
    private:
        static std::atomic<int> next_matrix;
    public:
        eigen_mat w;
        std::vector<dim_t> dims;
        const size_t id;

        MatInternal(int n, int d, bool empty=false);
        MatInternal(const MatInternal<R>& m);

};

template<typename R>
class GradInternal {
    typedef Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
    typedef Eigen::Map<eigen_mat> eigen_mat_view;
    public:
        eigen_mat dw;

        GradInternal(int n, int d, bool empty=true);
        GradInternal(const GradInternal<R>& g);

};

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
template<typename R>
class Mat {
    typedef Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
    typedef Eigen::Map<eigen_mat> eigen_mat_view;
    private:
        std::shared_ptr<MatInternal<R>> m;
        std::shared_ptr<GradInternal<R>> g;

    public:
        std::shared_ptr<std::string> name = nullptr;

        mutable eigen_mat_view w;
        mutable eigen_mat_view dw;
    private:
        void point_view_to_internal_memory();

    public:

        // sometimes we don't need to reset m
        // (for example if it's about to be assigned).
        Mat (dim_t n, dim_t d, bool empty=false);
        // Zero mean normal distribution for weights.
        Mat (dim_t n, dim_t d, R std);
        // uniform distribution for weights.
        Mat (dim_t n, dim_t d, R lower, R upper);
        /*
        A copy constructor that perform shallow and deep
        copies of a Mat.
        By default shallow copy is performed.

        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads.
        */
        Mat (const Mat<R>& m, bool copy_w=false, bool copy_d=false);

        ~Mat();

        void print() const;

        /**
        Adds 1 to the gradient (`dw`) of every element in this Matrix as long
        as the matrix is 1x1 (a scalar).

        Useful for computing cross entropy, mean squared error, and other
        loss functions in vanilla ML fashion.

        **/
        void grad();


        const std::vector<int>& dims() const;
        const int& dims(int idx) const;
        unsigned int number_of_elements() const;

        const int& id() const;

        void set_name(std::string& newname);
        void set_name(char* newname);
        void set_name(const char* newname);

        void npy_save(std::string fname, std::string mode = "w");
        void npy_save(FILE*);
        void npy_load(std::string fname);
        void npy_load(FILE*);
        void npy_load(cnpy::NpyArray&);
        Mat (std::string fname);
        static Mat RandMat(dim_t n, dim_t d, R std);
        static Mat Empty(dim_t n, dim_t d);
        /* A copy constructor that perform shallow copies of a Mat.

        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads. */
        static Mat shallow_copy(const Mat&);
        operator std::string() const;


        // Various operations on matrix.
        // Soon to be replaced by legitimate operators
        // See MatOps for documentation.
        Mat<R> eltmul_broadcast(Mat<R>);
        Mat<R> eltmul(Mat<R>);
        Mat<R> eltmul(R);
        Mat<R> eltmul_broadcast_rowwise(Mat<R>);
        Mat<R> eltmul_rowwise(Mat<R>);
        Mat<R> add_broadcast(Mat<R>);
        Mat<R> add(Mat<R>);
        Mat<R> sub(Mat<R>);
        Mat<R> sub_broadcast(Mat<R>);
        Mat<R> sub_broadcast_reversed(Mat<R>);
        Mat<R> square();
        Mat<R> sum();
        Mat<R> mean();
        Mat<R> log();
        Mat<R> exp();
        Mat<R> sigmoid();
        Mat<R> steep_sigmoid(R aggressiveness = 3.75);
        Mat<R> T();
        Mat<R> tanh();
        Mat<R> relu();
        Mat<R> mul(Mat<R>);
        Mat<R> rows_pluck(Indexing::Index);
        Mat<R> rows_cols_pluck(Indexing::Index, Indexing::Index);
        Mat<R> row_pluck(int);
};

template<typename R>
std::ostream& operator<<(std::ostream&, const Mat<R>&);

// define hash code for matrices:
namespace std {
        template <typename R> struct hash<Mat<R>> {
                std::size_t operator()(const Mat<R>&) const;
        };
}

namespace utils {
    template<typename R>
    void save_matrices(std::vector<std::shared_ptr<Mat<R>>>&, std::string);

    template<typename R>
    void load_matrices(std::vector<std::shared_ptr<Mat<R>>>&, std::string);
}

template <typename R>
bool operator!=(const Mat<R>&, const Mat<R>&);

template <typename R>
bool operator==(const Mat<R>&, const Mat<R>&);

template<typename R>
int argmax(std::shared_ptr<Mat<R>>);

template<typename R>
int argmax_slice(std::shared_ptr<Mat<R>>, int, int);

#endif
