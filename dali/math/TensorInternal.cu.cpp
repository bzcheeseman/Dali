#include "dali/math/TensorInternal.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"
#include "dali/utils/core_utils.h"

template<typename R, int dimension>
R TensorInternal<R, dimension>::sum() const {
    #ifdef DALI_USE_CUDA
        if (compute_me_on_gpu()) {
            return TensorOps::sum(this->gpu_data(), this->number_of_elements() );
        }
    #endif

    return TensorOps::sum(this->cpu_data(), this->number_of_elements() );
}

template<typename R, int dimension>
std::vector<int> TensorInternal<R, dimension>::argmin(int reduce_dim) const {
    // reduce colwise
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmin(this->gpu_data(), reduce_dim);
    }
    #endif

    return TensorOps::arg::argmin(this->cpu_data(), reduce_dim);
}

template<typename R, int dimension>
std::vector<int> TensorInternal<R, dimension>::argmax(int reduce_dim) const {
    // reduce colwise
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmax(this->gpu_data(), reduce_dim);
    }
    #endif

    return TensorOps::arg::argmax(this->cpu_data(), reduce_dim);
}

template<typename R, int dimension>
int TensorInternal<R, dimension>::argmin() const {
    // reduce colwise
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmin( TensorOps::to_thrust(this->gpu_data()), this->number_of_elements() )[0];
    }
    #endif

    return TensorOps::arg::argmin(this->cpu_data().dptr_, this->number_of_elements() )[0];
}

template<typename R, int dimension>
int TensorInternal<R, dimension>::argmax() const {
    // reduce colwise
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmax( TensorOps::to_thrust(this->gpu_data()), this->number_of_elements() )[0];
    }
    #endif

    return TensorOps::arg::argmax(this->cpu_data().dptr_, this->number_of_elements() )[0];
}

template<typename R, int dimension>
int TensorInternal<R, dimension>::argmax_slice(int lower, int upper) const {
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmax( TensorOps::to_thrust(this->gpu_data()) + lower, upper - lower )[0];
    }
    #endif

    return TensorOps::arg::argmax(this->cpu_data().dptr_ + lower, upper - lower)[0];
}
template<typename R, int dimension>
int TensorInternal<R, dimension>::argmin_slice(int lower, int upper) const {
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmin( TensorOps::to_thrust(this->gpu_data()) + lower, upper - lower )[0];
    }
    #endif

    return TensorOps::arg::argmin(this->cpu_data().dptr_ + lower, upper - lower)[0];
}

template<typename R, int dimension>
R TensorInternal<R, dimension>::L2_norm() const {
    #ifdef DALI_USE_CUDA
        if (compute_me_on_gpu()) {
            return TensorOps::L2_norm(this->gpu_data(), this->number_of_elements() );
        }
    #endif

    return TensorOps::L2_norm(this->cpu_data(), this->number_of_elements() );
}

template<typename R, int dimension>
bool TensorInternal<R,dimension>::operator==(const TensorInternal<R,dimension>& other) const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu({this, &other})) {
            return TensorOps::comparison::equals(this->gpu_data(), other.gpu_data(), this->number_of_elements());
        }
    #endif
    return TensorOps::comparison::equals(this->cpu_data(), other.cpu_data(), this->number_of_elements());
}

template<typename R, int dimension>
bool TensorInternal<R,dimension>::allclose(const TensorInternal<R,dimension>& other, R tol) const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu({this, &other})) {
            return TensorOps::comparison::allclose(this->gpu_data(), other.gpu_data(), this->number_of_elements(), tol);
        }
    #endif
    return TensorOps::comparison::allclose(this->cpu_data(), other.cpu_data(), this->number_of_elements(), tol);
}

template<typename R, int dimension>
typename TensorInternal<R, dimension>::lazy_t TensorInternal<R,dimension>::wrapper() const {
    return lazy_t(*this);
}

template<typename R, int dimension>
TensorInternal<R,dimension>::operator lazy_t() const {
    return wrapper();
}

template<typename R, int dimension>
bool TensorInternal<R, dimension>::compute_me_on_gpu() const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu({this})) {
            return true;
        }
    #endif
    return false;
}


template<typename R, int dimension>
R& TensorInternal<R,dimension>::operator()(int i, int j) {
    int offset = this->cpu_data().stride_  * i + j;
    return *(this->mutable_cpu_data().dptr_ + offset);
}

template<typename R, int dimension>
R TensorInternal<R,dimension>::operator()(int i, int j) const {
    // this is wrong for dimension > 2 or == 1
    int offset = this->cpu_data().stride_  * i + j;
    return *(this->cpu_data().dptr_ + offset);
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
                      << (*this)(i, j) << " ";
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

template class TensorInternal<float, 1>;
template class TensorInternal<double,1>;
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
