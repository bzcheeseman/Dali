#include "dali/tensor/op/other.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

using std::vector;

namespace matops {
    template<typename R>
    Mat<R> Other<R>::fill(Mat<R> matrix, R filler) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = filler;
        return out;
    }

    template<typename R>
    Mat<R> Other<R>::consider_constant_if(
            Mat<R> matrix,
            bool should_consider_constant) {
        if (should_consider_constant)
            return consider_constant(matrix);
        return matrix;
    }

    template<typename R>
    Mat<R> Other<R>::consider_constant(Mat<R> matrix) {
        // perform a copy of the matrix that references
        // everything and owns nothing. A true nomad.
        Mat<R> out(matrix, false, false);
        out.constant = true;
        return out;
    }

    template<typename R>
    vector<int> Other<R>::argsort(Mat<R> mat) {
        return MAT(mat).argsort();
    }

    template<typename R>
    vector<size_t> Other<R>::argsort(const vector<Mat<R>>& v) {
        // https://www.linux.com/news/software/developer/81090-c-the-gpu-and-thrust-sorting-numbers-on-the-gpu
        // should switch to gpu when more than 10,000 elements to sort.
        // initialize original index locations
        vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return MAT(v[i1])(0) < MAT(v[i2])(0);});

        return idx;
    }

    template<typename R>
    vector<int> Other<R>::argmax(const Mat<R>& mat, int reduce_dim) {
        return MAT(mat).argmax(reduce_dim);
    }

    template<typename R>
    vector<int> Other<R>::argmin(const Mat<R>& mat, int reduce_dim) {
        return MAT(mat).argmin(reduce_dim);
    }

    template<typename R>
    int Other<R>::argmin(const Mat<R>& mat) {
        return MAT(mat).argmin();
    }

    template<typename R>
    int Other<R>::argmax(const Mat<R>& mat) {
        return MAT(mat).argmax();
    }

    template<typename R>
    int Other<R>::argmax_slice(const Mat<R>& mat, int lower, int upper) {
        assert(lower <= upper);
        return MAT(mat).argmax_slice(lower, upper);
    }

    template<typename R>
    int Other<R>::argmin_slice(const Mat<R>& mat, int lower, int upper) {
        assert(lower <= upper);
        return MAT(mat).argmin_slice(lower, upper);
    }

    template<typename R>
    bool Other<R>::is_nan(Mat<R> a) {
        return MAT(a).is_nan();
    }

    template<typename R>
    bool Other<R>::is_grad_nan(Mat<R> a) {
        return GRAD(a).is_nan();
    }

    template<typename R>
    bool Other<R>::equals(Mat<R> a, Mat<R> b) {
        // wrong dimensions
        if (a.dims() != b.dims())
            return false;
        return MAT(a) == MAT(b);
    }

    template<typename R>
    bool Other<R>::allclose(Mat<R> a, Mat<R> b, R tol) {
        if (a.dims() != b.dims())
            return false;
        return MAT(a).allclose(MAT(b), tol);
    }

    template<typename R>
    bool Other<R>::grad_allclose(Mat<R> a, Mat<R> b, R tol) {
        if (a.dims() != b.dims())
            return false;
        return GRAD(a).allclose(GRAD(b), tol);
    }

    template<typename R>
    void Other<R>::copy(Mat<R>* dest, const Mat<R>& source) {
        MAT(*dest)  = MAT(source).wrapper();
    }

    template<typename R>
    void Other<R>::copy_grad(Mat<R>* dest, const Mat<R>& source) {
        GRAD(*dest) = GRAD(source).wrapper();
    }

    template class Other<float>;
    template class Other<double>;
    template class Other<int>;

}
