#include "dali/tensor/op/other.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

#define DONT_COMPILE

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
    Mat<R> Other<R>::consider_constant(
            Mat<R> matrix
            ) {
        // perform a copy of the matrix that references
        // everything and owns nothing. A true nomad.
        Mat<R> out(matrix, false, false);
        out.constant = true;
        return out;
    }

    template<typename R>
    vector<size_t> Other<R>::argsort_rowwise(Mat<R> m) {
        #ifndef DONT_COMPILE
        // initialize original index locations
        vector<size_t> idx(m.dims(0));
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(), [&m](size_t i1, size_t i2) {
            return MAT(m)(i1) < MAT(m)(i2);
        });
        return idx;
        #else
        return {};
        #endif
    }

    template<typename R>
    vector<size_t> Other<R>::argsort(const vector<Mat<R>>& v) {
        // https://www.linux.com/news/software/developer/81090-c-the-gpu-and-thrust-sorting-numbers-on-the-gpu
        // should switch to gpu when more than 10,000 elements to sort.
        #ifndef DONT_COMPILE
        // initialize original index locations
        vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return MAT(v[i1])(0) < MAT(v[i2])(0);});

        return idx;
        #else
        return {};
        #endif
    }

    template<typename R>
    void Other<R>::resize(const Mat<R>& mat, dim_t n, dim_t d) {
        #ifndef DONT_COMPILE
        mat.w()->dims[0] = n;
        mat.w()->dims[1] = d;
        MAT(mat).conservativeResize(n, d);
        GRAD(mat).conservativeResize(n, d);
        #else
        #endif
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
    int Other<R>::argmax(const Mat<R>& mat) {
        #ifndef DONT_COMPILE
        int i = 0;
        R current_max = -std::numeric_limits<R>::infinity();
        auto ptr = mat.w()->data();
        for (int j = 0; j < mat.number_of_elements(); j++) {
            if (*ptr > current_max) {
                current_max = *ptr;
                i = j;
            }
            ptr++;
        }
        return i;
        #else
        return 0;
        #endif
    }

    template<typename R>
    int Other<R>::argmax_slice(const Mat<R>& mat, int lower, int upper) {
        #ifndef DONT_COMPILE
        int i = 0;
        R current_max = -std::numeric_limits<R>::infinity();
        auto ptr = mat.w()->data();
        for (int j = lower; j < upper; j++) {
            if (*ptr > current_max) {
                current_max = *ptr;
                i = j;
            }
            ptr++;
        }
        return i;
        #else
        return 0;
        #endif
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

    template class Other<float>;
    template class Other<double>;

}
