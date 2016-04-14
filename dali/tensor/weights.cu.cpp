#include "dali/tensor/Weights.h"

#include "dali/array/array.h"
// #include "dali/array/TensorRandom.h"
namespace weights {
    initializer_t empty() {
        return [](const sync_t&){};
    };

    initializer_t zeros() {
        return [](const sync_t& tensor){
            tensor.clear();
        };
    };

    initializer_t ones() {
        return [](const sync_t& matrix){
            tensor = 1;
        };
    };

    initializer_t eye(R diag) {
        return [diag](const sync_t& matrix) {
            ASSERT2(false, "eye: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (matrix.compute_me_on_gpu()) {
            //         TensorOps::eye(matrix.mutable_gpu_data(), diag);
            //         return;
            //     }
            // #endif
            // TensorOps::eye(matrix.mutable_cpu_data(), diag);
        };
    };

    initializer_t uniform(R lower, R upper) {
        return [lower, upper](const sync_t& matrix) {
            ASSERT2(false, "uniform: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (matrix.compute_me_on_gpu()) {
            //         TensorOps::random::uniform(matrix.mutable_gpu_data(), lower, upper);
            //         return;
            //     }
            // #endif
            // TensorOps::random::uniform(matrix.mutable_cpu_data(), lower, upper);
        };
    };

    initializer_t uniform(R bound) {
        return uniform(-bound/2.0, bound/2.0);
    }

    initializer_t gaussian(R mean, R std) {
        return [mean, std](const sync_t& matrix) {
            ASSERT2(false, "gaussian: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (matrix.compute_me_on_gpu()) {
            //         TensorOps::random::gaussian(matrix.mutable_gpu_data(), mean, std);
            //         return;
            //     }
            // #endif
            // TensorOps::random::gaussian(matrix.mutable_cpu_data(), mean, std);
        };
    };

    initializer_t bernoulli(R prob) {
        return [prob](const sync_t& matrix) {
            ASSERT2(false, "bernoulli: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (matrix.compute_me_on_gpu()) {
            //         TensorOps::random::bernoulli(matrix.mutable_gpu_data(), prob);
            //         return;
            //     }
            // #endif
            // TensorOps::random::bernoulli(matrix.mutable_cpu_data(), prob);
        };
    };

    initializer_t bernoulli_normalized(R prob) {
        return [prob](const sync_t& matrix) {
            ASSERT2(false, "bernoulli_normalized: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (matrix.compute_me_on_gpu()) {
            //         TensorOps::random::bernoulli_normalized(matrix.mutable_gpu_data(), prob);
            //         return;
            //     }
            // #endif
            // TensorOps::random::bernoulli_normalized(matrix.mutable_cpu_data(), prob);
        };
    };

    initializer_t gaussian(R std) {
        return gaussian(0.0, std);
    }

    initializer_t svd(initializer_t preinitializer) {
        return [preinitializer](const sync_t& matrix) {
            ASSERT2(false, "SVD INIT: Not implemented yet");
            /* Eigen implementation */
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
}
