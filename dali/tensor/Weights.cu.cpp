#include "dali/tensor/Weights.h"

#include "dali/math/TensorOps.h"
#include "dali/math/TensorInternal.h"

template<typename R>
typename weights<R>::initializer_t weights<R>::empty() {
    return [](sync_t){};
};

template<typename R>
typename weights<R>::initializer_t weights<R>::zeros() {
    return [](sync_t matrix){
        matrix = (R)0.0;
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::eye(R diag) {
    return [diag](sync_t matrix) {
        #ifdef DALI_USE_CUDA
            if (matrix.compute_me_on_gpu()) {
                TensorOps::eye(matrix.mutable_gpu_data(), diag);
                return;
            }
        #endif
        TensorOps::eye(matrix.mutable_cpu_data(), diag);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R lower, R upper) {
    return [lower, upper](sync_t matrix) {
        #ifdef DALI_USE_CUDA
            if (matrix.compute_me_on_gpu()) {
                TensorOps::random::uniform(matrix.mutable_gpu_data(), lower, upper);
                return;
            }
        #endif
        TensorOps::random::uniform(matrix.mutable_cpu_data(), lower, upper);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R bound) {
    return uniform(-bound/2.0, bound/2.0);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R mean, R std) {
    return [mean, std](sync_t matrix) {
        #ifdef DALI_USE_CUDA
            if (matrix.compute_me_on_gpu()) {
                TensorOps::random::gaussian(matrix.mutable_gpu_data(), mean, std);
                return;
            }
        #endif
        TensorOps::random::gaussian(matrix.mutable_cpu_data(), mean, std);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::bernoulli(R prob) {
    return [prob](sync_t matrix) {
        #ifdef DALI_USE_CUDA
            if (matrix.compute_me_on_gpu()) {
                TensorOps::random::bernoulli(matrix.mutable_gpu_data(), prob);
                return;
            }
        #endif
        TensorOps::random::bernoulli(matrix.mutable_cpu_data(), prob);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::bernoulli_normalized(R prob) {
    return [prob](sync_t matrix) {
        #ifdef DALI_USE_CUDA
            if (matrix.compute_me_on_gpu()) {
                TensorOps::random::bernoulli_normalized(matrix.mutable_gpu_data(), prob);
                return;
            }
        #endif
        TensorOps::random::bernoulli_normalized(matrix.mutable_cpu_data(), prob);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R std) {
    return gaussian(0.0, std);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::svd(initializer_t preinitializer) {
    return [preinitializer](sync_t matrix) {
        // assert(matrix.dims().size() == 2);
        // preinitializer(matrix);
        // auto svd = GET_MAT(matrix).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        // int n = matrix.dims(0);
        // int d = matrix.dims(1);
        // if (n < d) {
        //     GET_MAT(matrix) = svd.matrixV().block(0, 0, n, d);
        // } else {
        //     GET_MAT(matrix) = svd.matrixU().block(0, 0, n, d);
        // }
    };
}

template struct weights<float>;
template struct weights<double>;
