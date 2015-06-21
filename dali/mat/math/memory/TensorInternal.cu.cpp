#include "dali/mat/math/memory/TensorInternal.h"
#include "dali/mat/math/memory/TensorOps.h"
#include "dali/mat/math/memory/LazyTensor.h"

template<typename R, int dimension>
R TensorInternal<R, dimension>::sum() const {
    #ifdef DALI_USE_CUDA
        if (compute_me_on_gpu()) {
            return TensorOps::sum(this->gpu_data(), this->number_of_elements() );
        } else {
            return TensorOps::sum(this->cpu_data(), this->number_of_elements() );
        }
    #else
        return TensorOps::sum(this->cpu_data(), this->number_of_elements());
    #endif
}

template<typename R, int dimension>
bool TensorInternal<R,dimension>::operator==(const TensorInternal<R,dimension>& other) const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu({std::cref(*this), std::cref(other)})) {
            return TensorOps::equals(this->gpu_data(), other.gpu_data(), this->number_of_elements());
        }
    #endif
    return TensorOps::equals(this->cpu_data(), other.cpu_data(), this->number_of_elements());
}

template<typename R, int dimension>
bool TensorInternal<R,dimension>::allclose(const TensorInternal<R,dimension>& other, R tol) const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu({std::cref(*this), std::cref(other)})) {
            return TensorOps::allclose(this->gpu_data(), other.gpu_data(), this->number_of_elements(), tol);
        }
    #endif
    return TensorOps::allclose(this->cpu_data(), other.cpu_data(), this->number_of_elements(), tol);
}

template<typename R, int dimension>
typename TensorInternal<R, dimension>::lazy_t TensorInternal<R,dimension>::wrapper() {
    return lazy_t(std::cref(*this));
}

template<typename R, int dimension>
TensorInternal<R,dimension>::operator lazy_t() {
    return wrapper();
}

template<typename R, int dimension>
bool TensorInternal<R, dimension>::compute_me_on_gpu() const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu({std::cref(*this)})) {
            return true;
        }
    #endif
    return false;
}

template class TensorInternal<float,2>;
template class TensorInternal<double,2>;

template<typename R, int dimension>
bool should_compute_on_gpu(
        std::initializer_list<std::reference_wrapper<const TensorInternal<R,dimension>>> sts) {

#ifdef DALI_USE_CUDA
    if (sts.size() == 1) {
        const auto& mat = (*sts.begin()).get();
        return (mat.prefers_gpu() && (mat.gpu_fresh || !mat.cpu_fresh && !mat.gpu_fresh));
    }
    bool everybody_cpu = true;
    bool everybody_gpu = true;
    for (auto& st : sts) {
        everybody_gpu = everybody_gpu && st.get().prefers_gpu();
        everybody_cpu = everybody_cpu && st.get().prefers_cpu();
    }
    if (everybody_cpu) {
        return false;
    } else if (everybody_gpu) {
        return true;
    } else {
        return SynchronizedMemory<R,dimension>::tie_breaker_device == DEVICE_GPU;
    }
#else
    return false;
#endif
}

template<typename R, int dimension>
bool should_compute_on_gpu(const std::vector<std::reference_wrapper<const TensorInternal<R,dimension>>>& sts) {

#ifdef DALI_USE_CUDA
    if (sts.size() == 1) {
        const auto& mat = (*sts.begin()).get();
        return (mat.prefers_gpu() && (mat.gpu_fresh || !mat.cpu_fresh && !mat.gpu_fresh));
    }
    bool everybody_cpu = true;
    bool everybody_gpu = true;
    for (auto& st : sts) {
        everybody_gpu = everybody_gpu && st.get().prefers_gpu();
        everybody_cpu = everybody_cpu && st.get().prefers_cpu();
    }
    if (everybody_cpu) {
        return false;
    } else if (everybody_gpu) {
        return true;
    } else {
        return SynchronizedMemory<R,dimension>::tie_breaker_device == DEVICE_GPU;
    }
#else
    return false;
#endif
}

template bool should_compute_on_gpu(
        std::initializer_list<
            std::reference_wrapper<
                const TensorInternal<float,2>
            >
        >);
template bool should_compute_on_gpu(
        std::initializer_list<
            std::reference_wrapper<
                const TensorInternal<double,2>
            >
        >);

template bool should_compute_on_gpu(
        const std::vector<
            std::reference_wrapper<
                const TensorInternal<float,2>
            >
        >&);
template bool should_compute_on_gpu(
        const std::vector<
            std::reference_wrapper<
                const TensorInternal<double,2>
            >
        >&);
