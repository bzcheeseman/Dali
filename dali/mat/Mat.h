#ifndef CORE_MAT_H
#define CORE_MAT_H

#include <atomic>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// For handling json_finite_distribution
#include <json11.hpp>

#include "dali/utils.h"
#include "dali/mat/math/MatOps.h"
#include "dali/mat/Index.h"
#include "dali/mat/Tape.h"


#define EPS 1e-9

typedef unsigned int dim_t;

template<typename R> class MatInternal;
template<typename R> class GradInternal;
template<typename R> class Mat;

template<typename R>
struct weights {
    typedef std::function<void(Mat<R>&)> initializer_t;

    static initializer_t uninitialized();
    static initializer_t zeros();
    static initializer_t uniform(R lower, R upper);
    static initializer_t uniform(R bound);
    static initializer_t gaussian(R mean, R std);
    static initializer_t gaussian(R std);
    static initializer_t eye(R diag = 1.0);

    // Preinitializer is first run on the matrix and then SVD initialization
    // happens. If you are unsure what kind of preinitializer to use, then try
    // weights<R>::uniform(m), were m is the number of columns in your matrix.
    // DISCLAIMER: do not use on big matrices (like embeddings) - faster version
    // is a subject of current research.
    static initializer_t svd(initializer_t preinitializer=gaussian(1.0));
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
    public:
        typedef std::shared_ptr<MatInternal<R>> mat_internal_t;
        typedef std::shared_ptr<GradInternal<R>> grad_internal_t;
    private:
        mat_internal_t  m;
        grad_internal_t g;
    public:

        std::shared_ptr<std::string> name = nullptr;

        // TODO(jonathan): wtf!
        bool sparse = false;
        std::shared_ptr<std::vector<uint>> sparse_row_keys;
        bool constant;

        Mat();

        // Initializes with zeros;
        Mat (dim_t n, dim_t d);
        // sometimes we don't need to reset m
        // (for example if it's about to be assigned).
        Mat (dim_t n, dim_t d, bool fill_zeros);
        Mat (dim_t n, dim_t d,
             typename weights<R>::initializer_t wi);
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

        mat_internal_t  w() const;
        R w(int i) const;
        R dw(int i) const;
        R w(int i, int j) const;
        R dw(int i, int j) const;

        grad_internal_t dw() const;

        const std::vector<dim_t>& dims() const;
        const dim_t dims(int idx) const;

        unsigned int number_of_elements() const;

        bool empty() const;

        const int id() const;

        void set_name(std::string& newname);
        void set_name(char* newname);
        void set_name(const char* newname);

        void npy_save(std::string fname, std::string mode = "w");
        void npy_save(FILE*);
        void npy_load(std::string fname);
        void npy_load(FILE*);
        void npy_load(cnpy::NpyArray&);
        Mat (std::string fname);
        static Mat Empty(dim_t n, dim_t d);
        /* A copy constructor that perform shallow copies of a Mat.

        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads. */
        Mat shallow_copy();
        operator std::string() const;
        void resize(dim_t rows, dim_t cols);

        // Various operations on matrix.
        // Soon to be replaced by legitimate operators
        // See MatOps for documentation.
        Mat<R> eltmul_broadcast(Mat<R>) const;
        Mat<R> eltmul(Mat<R>) const;
        Mat<R> eltmul(R) const;
        Mat<R> eltmul_broadcast_rowwise(Mat<R>) const;
        Mat<R> eltmul_rowwise(Mat<R>) const;
        Mat<R> add_broadcast(Mat<R>) const;
        Mat<R> add(Mat<R>) const;
        Mat<R> sub(Mat<R>) const;
        Mat<R> sub_broadcast(Mat<R>) const;
        Mat<R> sub_broadcast_reversed(Mat<R>) const;
        Mat<R> square() const;
        Mat<R> L2_norm() const;
        Mat<R> sum() const;
        Mat<R> mean() const;
        Mat<R> log() const;
        Mat<R> exp() const;
        Mat<R> abs() const;
        Mat<R> sigmoid() const;
        Mat<R> steep_sigmoid(R aggressiveness = 3.75) const;
        // Warning: transpose makes a copy, uses extra memory
        Mat<R> T() const;
        Mat<R> tanh() const;
        Mat<R> relu() const;
        Mat<R> mul(Mat<R>) const;
        Mat<R> dot(Mat<R>) const;
        Mat<R> pow(R) const;
        Mat<R> pow(int) const;
        Mat<R> sqrt() const;
        Mat<R> elt_inv() const;
        int argmax() const;
        /*
        Restricted range argmax: returns the index of the
        highest value between two indices, lower and upper
        (useful if a range of predictions is inadmissible,
        so we are only considering a subset of predictions)
        */
        int argmax_slice(int lower, int upper) const;

        Mat<R> operator-() const;

        Mat<R> operator+(Mat<R>) const;
        Mat<R> operator+(R) const;
        Mat<R>& operator+=(Mat<R>);
        Mat<R>& operator+=(R);

        Mat<R> operator-(Mat<R>) const;
        Mat<R> operator-(R) const;
        Mat<R>& operator-=(Mat<R>);
        Mat<R>& operator-=(R);

        Mat<R> operator*(Mat<R> other) const;
        Mat<R> operator*(R alpha) const;
        Mat<R>& operator*=(Mat<R>);
        Mat<R>& operator*=(R);

        Mat<R> operator/(Mat<R> other) const;
        Mat<R> operator/(R alpha) const;
        Mat<R>& operator/=(Mat<R>);
        Mat<R>& operator/=(R);

        Mat<R> operator^(R) const;
        Mat<R> operator^(Mat<R>) const;
        Mat<R> operator^(int) const;

        // Plucking rows and columns:
        Mat<R> operator[](int) const;
        Mat<R> operator()(int) const;
        Mat<R> operator[](Indexing::Index) const;
        Mat<R> operator()(Indexing::Index) const;
        Mat<R> operator()(Indexing::Index, Indexing::Index) const;
        // Mat<R> operator()(void*, Indexing::Index) const;
        Mat<R> operator()(void*, int) const;
        static Mat<R> zeros_like(Mat<R> shape);
        static Mat<R> empty_like(Mat<R> shape);
};
extern const std::vector<dim_t> mat_missing_dimensions;

template<typename R>
Mat<R> operator+(int other, Mat<R> mat);
template<typename R>
Mat<R> operator+(float other, Mat<R> mat);
template<typename R>
Mat<R> operator+(double other, Mat<R> mat);

template<typename R>
Mat<R> operator-(int other, Mat<R> mat);
template<typename R>
Mat<R> operator-(float other, Mat<R> mat);
template<typename R>
Mat<R> operator-(double other, Mat<R> mat);

template<typename R>
Mat<R> operator*(int other, Mat<R> mat);
template<typename R>
Mat<R> operator*(float other, Mat<R> mat);
template<typename R>
Mat<R> operator*(double other, Mat<R> mat);

template<typename R>
std::ostream& operator<<(std::ostream&, const Mat<R>&);

// define hash code for matrices:
namespace std {
    template <typename R> struct hash<Mat<R>> {
            std::size_t operator()(const Mat<R>&) const;
    };
}

namespace utils {
    template<typename T>
    std::vector<size_t> argsort_rowwise(Mat<T> &m);

    template<typename R>
    void save_matrices(std::vector<Mat<R>>, std::string);

    template<typename R>
    void load_matrices(std::vector<Mat<R>>, std::string);

    template<typename R>
    json11::Json json_finite_distribution(const Mat<R>&, const std::vector<std::string>& labels);

}

template <typename R>
bool operator!=(const Mat<R>&, const Mat<R>&);

template <typename R>
bool operator==(const Mat<R>&, const Mat<R>&);

#endif
