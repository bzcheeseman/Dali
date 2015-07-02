#include "dali/math/SynchronizedMemory.h"
#include "dali/math/LazyTensor.h"
#include "dali/math/TensorOps.h"

using mshadow::AllocSpace;
using mshadow::FreeSpace;
using mshadow::Shape2;
using mshadow::Tensor;
using mshadow::Copy;

template<int dimension>
std::ostream &operator <<(std::ostream &os, const mshadow::Shape<dimension> &shape) {
    if (dimension == 0) {
        return os << "<shape ()>";
    } else {
        os << "<shape (";
        for (int i = 0; i < dimension;i++) {
            os << shape[i];
            if (i != dimension - 1) os << ", ";
        }
        os << ")>";
        return os;
    }
}

template std::ostream& operator<< <0>(std::ostream& strm, const mshadow::Shape<0>& a);
template std::ostream& operator<< <1>(std::ostream& strm, const mshadow::Shape<1>& a);
template std::ostream& operator<< <2>(std::ostream& strm, const mshadow::Shape<2>& a);
template std::ostream& operator<< <3>(std::ostream& strm, const mshadow::Shape<3>& a);
template std::ostream& operator<< <4>(std::ostream& strm, const mshadow::Shape<4>& a);
template std::ostream& operator<< <5>(std::ostream& strm, const mshadow::Shape<5>& a);
template std::ostream& operator<< <6>(std::ostream& strm, const mshadow::Shape<6>& a);
template std::ostream& operator<< <7>(std::ostream& strm, const mshadow::Shape<7>& a);
template std::ostream& operator<< <8>(std::ostream& strm, const mshadow::Shape<8>& a);
template std::ostream& operator<< <9>(std::ostream& strm, const mshadow::Shape<9>& a);

void dali_init() {
    mshadow::InitTensorEngine<mshadow::cpu>();
    #ifdef DALI_USE_CUDA
        mshadow::InitTensorEngine<mshadow::gpu>();
    #endif
}


/**** SHOULD COMPUTE GPU-land **/

template<typename R>
bool should_compute_on_gpu(const std::vector<const SynchronizedMemory<R>*>& sts) {

#ifdef DALI_USE_CUDA
    if (sts.size() == 1) {
        auto mover = (sts.front());
        return (mover->prefers_gpu() && (mover->gpu_fresh || !mover->cpu_fresh && !mover->gpu_fresh));
    }
    bool everybody_cpu = true;
    bool everybody_gpu = true;
    for (auto st : sts) {
        everybody_gpu = everybody_gpu && st->prefers_gpu();
        everybody_cpu = everybody_cpu && st->prefers_cpu();
    }
    if (everybody_cpu) {
        return false;
    } else if (everybody_gpu) {
        return true;
    } else {
        return SynchronizedMemory<R>::tie_breaker_device == DEVICE_GPU;
    }
#else
    return false;
#endif
}

template bool should_compute_on_gpu(const std::vector<const SynchronizedMemory<float>*>& sts);
template bool should_compute_on_gpu(const std::vector<const SynchronizedMemory<double>*>& sts);

/******************* SYNCHRONIZED MEMORY ************************************************/

template<typename R>
bool SynchronizedMemory<R>::prefers_cpu() const {
    return preferred_device == DEVICE_CPU;
}

template<typename R>
bool SynchronizedMemory<R>::prefers_gpu()  const {
    return preferred_device == DEVICE_GPU;
}

#ifdef DALI_USE_CUDA
    template<typename R>
    Device SynchronizedMemory<R>::tie_breaker_device = DEVICE_GPU;
#endif

template<typename R>
SynchronizedMemory<R>::SynchronizedMemory(int _total_memory,
                                          int _inner_dimension,
                                          Device _preferred_device) :
#ifdef DALI_USE_CUDA
        gpu_fresh(false),
        allocated_gpu(false),
        gpu_ptr(NULL),
#endif
        cpu_fresh(false),
        allocated_cpu(false),
        cpu_ptr(NULL),
        total_memory(_total_memory),
        inner_dimension(_inner_dimension),
        preferred_device(_preferred_device) {
    assert(total_memory % inner_dimension == 0);
}

template<typename R>
SynchronizedMemory<R>::SynchronizedMemory(const SynchronizedMemory& other) :
        SynchronizedMemory(other.total_memory, other.inner_dimension, other.preferred_device) {
    if (other.cpu_fresh) {
        const auto& data_source = other.dummy_cpu();
        copy_data_from(data_source);
    }
#ifdef DALI_USE_CUDA
    else if (other.gpu_fresh) {
        const auto& data_source = other.dummy_gpu();
        copy_data_from(data_source);
    }
#endif
    else {
        // data was not initialized on the source
        // so we also choose not to initialize.
        return;
    }
}

template<typename R>
void SynchronizedMemory<R>::free_cpu() const {
    if (allocated_cpu) {
        auto dummy = dummy_cpu();
        FreeSpace(&dummy);
        cpu_ptr = dummy.dptr_;
    }
    allocated_cpu = false;
}

#ifdef DALI_USE_CUDA
template<typename R>
void SynchronizedMemory<R>::free_gpu() const {
    if (allocated_gpu) {
        auto dummy = dummy_gpu();
        FreeSpace(&dummy);
        gpu_ptr = dummy.dptr_;
    }
    allocated_gpu = false;
}
#endif

template<typename R>
SynchronizedMemory<R>::~SynchronizedMemory() {
    free_cpu();
#ifdef DALI_USE_CUDA
    free_gpu();
#endif
}

#ifdef DALI_USE_CUDA
template<typename R>
mshadow::Tensor<mshadow::gpu, 2, R> SynchronizedMemory<R>::dummy_gpu() const {
    return mshadow::Tensor<mshadow::gpu, 2, R>(gpu_ptr, mshadow::Shape2(
            total_memory / inner_dimension, inner_dimension));
}
#endif

template<typename R>
mshadow::Tensor<mshadow::cpu, 2, R> SynchronizedMemory<R>::dummy_cpu() const {
     return mshadow::Tensor<mshadow::cpu, 2, R>(cpu_ptr, mshadow::Shape2(
            total_memory / inner_dimension, inner_dimension));
}

#ifdef DALI_USE_CUDA
    template<typename R>
    void SynchronizedMemory<R>::to_gpu() const {
        if (!this->gpu_fresh) {
            if (!allocated_gpu) {
                auto dummy = dummy_gpu();
                AllocSpace(&dummy, false);
                allocated_gpu = true;
                gpu_ptr = dummy.dptr_;
            }
            if (this->cpu_fresh) {
                auto mem_gpu = dummy_gpu();
                auto mem_cpu = dummy_cpu();
                Copy(mem_gpu, mem_cpu);
            }
            this->gpu_fresh = true;
        }
    }
#endif

template<typename R>
void SynchronizedMemory<R>::to_cpu() const {
    if (!this->cpu_fresh) {
        if (!allocated_cpu) {
            auto dummy = dummy_cpu();
            AllocSpace(&dummy, false);
            allocated_cpu = true;
            cpu_ptr = dummy.dptr_;
        }
#ifdef DALI_USE_CUDA
        if (this->gpu_fresh) {
            auto mem_gpu = dummy_gpu();
            auto mem_cpu = dummy_cpu();
            Copy(mem_cpu, mem_gpu);
        }
#endif
        this->cpu_fresh = true;
    }
}

template <typename R>
R* SynchronizedMemory<R>::cpu_data() const {
    to_cpu();
    return cpu_ptr;
}
template <typename R>
R* SynchronizedMemory<R>::mutable_cpu_data() {
    to_cpu();
    #ifdef DALI_USE_CUDA
        gpu_fresh = false;
    #endif;
    return cpu_ptr;
}


#ifdef DALI_USE_CUDA
template <typename R>
R* SynchronizedMemory<R>::gpu_data() const {
    to_gpu();
    return gpu_ptr;
}
template <typename R>
R* SynchronizedMemory<R>::mutable_gpu_data() {
    to_gpu();
    cpu_fresh = false;
    return gpu_ptr;
}

#endif

template<typename R>
template<typename SourceType>
void SynchronizedMemory<R>::copy_data_from(SourceType& data_source) {
    if (this->prefers_cpu()) {
        auto my_data = dummy_cpu();
        AllocSpace(&my_data, false);
        cpu_ptr = my_data.dptr_;
        allocated_cpu = true;

        Copy(my_data, data_source);
        this->cpu_fresh = true;
    } else {
#ifdef DALI_USE_CUDA
        auto my_data = dummy_gpu();
        AllocSpace(&my_data, false);
        gpu_ptr = my_data.dptr_;
        allocated_gpu = true;

        Copy(my_data, data_source);
        this->gpu_fresh = true;
#endif
    }
}

template class SynchronizedMemory<float>;
template class SynchronizedMemory<double>;
