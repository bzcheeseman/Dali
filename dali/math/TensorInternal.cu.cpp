#include "dali/math/TensorInternal.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

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
            return TensorOps::comparison::equals(this->gpu_data(), other.gpu_data(), this->number_of_elements());
        }
    #endif
    return TensorOps::comparison::equals(this->cpu_data(), other.cpu_data(), this->number_of_elements());
}

template<typename R, int dimension>
bool TensorInternal<R,dimension>::allclose(const TensorInternal<R,dimension>& other, R tol) const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu({std::cref(*this), std::cref(other)})) {
            return TensorOps::comparison::allclose(this->gpu_data(), other.gpu_data(), this->number_of_elements(), tol);
        }
    #endif
    return TensorOps::comparison::allclose(this->cpu_data(), other.cpu_data(), this->number_of_elements(), tol);
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


template<typename R, int dimension>
R& TensorInternal<R,dimension>::operator()(int i, int j) {
    return this->mutable_cpu_data()[i][j];
}

template<typename R, int dimension>
R TensorInternal<R,dimension>::operator()(int i, int j) const {
    return this->cpu_data()[i][j];
}

template<typename R, int dimension>
R& TensorInternal<R,dimension>::operator()(int i) {
    return *(this->mutable_cpu_data().dptr_ + i);
}

template<typename R, int dimension>
R TensorInternal<R,dimension>::operator()(int i) const {
    return *(this->cpu_data().dptr_ + i);
}

template<typename R, int dimension>
const R* TensorInternal<R,dimension>::data() const {
    return this->cpu_data().dptr_;
}

template<typename R, int dimension>
R* TensorInternal<R,dimension>::data() {
    return this->mutable_cpu_data().dptr_;
}

template<typename R, int dimension>
void TensorInternal<R,dimension>::print() const {
    const auto& data = this->cpu_data();
    for (int i = 0; i < data.shape_[0] ; ++i) {
        std::cout << (i == 0 ? "[" : " ");
        for (int j = 0; j < data.shape_[1]; ++j) {
            std::cout << std::fixed
                      << std::setw( 7 ) // keep 7 digits
                      << std::setprecision( 3 ) // use 3 decimals
                      << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                      << data[i][j] << " ";
        }
        std::cout << (i == data.shape_[0] - 1 ? "]" : "\n");
    }
    std::cout << std::endl;
}

template<typename R, int dimension>
void TensorInternal<R,dimension>::clear() {
    *this = (R) 0.0;
}

template<typename R, int dimension>
TensorInternal<R,dimension> TensorInternal<R, dimension>::zeros(mshadow::Shape<dimension> shape) {
    auto tensor = TensorInternal<R,dimension>(shape);
    tensor.clear();
    return tensor;
}

// template class TensorInternal<float, 1>;
// template class TensorInternal<double,1>;
template class TensorInternal<float, 2>;
template class TensorInternal<double,2>;
// template class TensorInternal<float, 3>;
// template class TensorInternal<double,3>;
// template class TensorInternal<float, 4>;
// template class TensorInternal<double,4>;
// template class TensorInternal<float, 5>;
// template class TensorInternal<double,5>;
// template class TensorInternal<float, 6>;
// template class TensorInternal<double,6>;
// template class TensorInternal<float, 7>;
// template class TensorInternal<double,7>;
// template class TensorInternal<float, 8>;
// template class TensorInternal<double,8>;
// template class TensorInternal<float, 9>;
// template class TensorInternal<double,9>;






/**** SHOULD COMPUTE GPU-land **/

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
