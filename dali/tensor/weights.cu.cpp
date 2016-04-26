#include "dali/tensor/weights.h"

#include "dali/array/array.h"
#include "dali/array/op/random.h"
// #include "dali/array/Tensordoubleandom.h"

namespace weights {
    initializer_t empty() {
        return [](Array&){};
    };

    initializer_t zeros() {
        return [](Array& tensor){
            tensor.clear();
        };
    };

    initializer_t ones() {
        return [](Array& tensor){
            tensor = 1;
        };
    };

    initializer_t eye(double diag) {
        return [diag](Array& tensor) {
            ASSERT2(false, "eye: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (tensor.compute_me_on_gpu()) {
            //         TensorOps::eye(tensor.mutable_gpu_data(), diag);
            //         return;
            //     }
            // #endif
            // TensorOps::eye(tensor.mutable_cpu_data(), diag);
        };
    };

    initializer_t uniform(double lower, double upper) {
        return [lower, upper](Array& tensor) {
            ASSERT2(false, "uniform: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (tensor.compute_me_on_gpu()) {
            //         TensorOps::random::uniform(tensor.mutable_gpu_data(), lower, upper);
            //         return;
            //     }
            // #endif
            // TensorOps::random::uniform(tensor.mutable_cpu_data(), lower, upper);
        };
    };

    initializer_t uniform(double bound) {
        return uniform(-bound/2.0, bound/2.0);
    }

    initializer_t gaussian(double mean, double std) {
        return [mean, std](Array& tensor) {
            tensor = random::gaussian(mean, std);

            ASSERT2(false, "gaussian: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (tensor.compute_me_on_gpu()) {
            //         TensorOps::random::gaussian(tensor.mutable_gpu_data(), mean, std);
            //         return;
            //     }
            // #endif
            // TensorOps::random::gaussian(tensor.mutable_cpu_data(), mean, std);
        };
    };

    initializer_t bernoulli(double prob) {
        return [prob](Array& tensor) {
            ASSERT2(false, "bernoulli: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (tensor.compute_me_on_gpu()) {
            //         TensorOps::random::bernoulli(tensor.mutable_gpu_data(), prob);
            //         return;
            //     }
            // #endif
            // TensorOps::random::bernoulli(tensor.mutable_cpu_data(), prob);
        };
    };

    initializer_t bernoulli_normalized(double prob) {
        return [prob](Array& tensor) {
            ASSERT2(false, "bernoulli_normalized: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (tensor.compute_me_on_gpu()) {
            //         TensorOps::random::bernoulli_normalized(tensor.mutable_gpu_data(), prob);
            //         return;
            //     }
            // #endif
            // TensorOps::random::bernoulli_normalized(tensor.mutable_cpu_data(), prob);
        };
    };

    initializer_t gaussian(double std) {
        return gaussian(0.0, std);
    }

    initializer_t svd(initializer_t preinitializer) {
        return [preinitializer](Array& tensor) {
            ASSERT2(false, "SVD INIT: Not implemented yet");
            /* Eigen implementation */
            // assert(tensor.dims().size() == 2);
            // preinitializer(tensor);
            // auto svd = GET_MAT(tensor).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
            // int n = tensor.dims(0);
            // int d = tensor.dims(1);
            // if (n < d) {
            //     GET_MAT(tensor) = svd.tensorV().block(0, 0, n, d);
            // } else {
            //     GET_MAT(tensor) = svd.tensorU().block(0, 0, n, d);
            // }
        };
    }
}
