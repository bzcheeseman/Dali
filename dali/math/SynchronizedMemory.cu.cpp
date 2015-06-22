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

bool MemoryMover::prefers_cpu() const {
    return preferred_device == DEVICE_CPU;
}

bool MemoryMover::prefers_gpu() const {
    return preferred_device == DEVICE_GPU;
}

#ifdef DALI_USE_CUDA
    MemoryMover::MemoryMover(bool _cpu_fresh, bool _gpu_fresh, PreferredDevice _preferred_device) :
            cpu_fresh(_cpu_fresh), gpu_fresh(_gpu_fresh), preferred_device(_preferred_device) {
    }
#else
    MemoryMover::MemoryMover(bool _cpu_fresh, PreferredDevice _preferred_device) :
            cpu_fresh(_cpu_fresh), preferred_device(_preferred_device) {
    }
#endif

#ifdef DALI_USE_CUDA
    PreferredDevice MemoryMover::tie_breaker_device = DEVICE_GPU;
#endif


/**** SHOULD COMPUTE GPU-land **/

bool should_compute_on_gpu(const std::vector<const MemoryMover*>& sts) {

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
        return MemoryMover::tie_breaker_device == DEVICE_GPU;
    }
#else
    return false;
#endif
}

/******************* SYNCHRONIZED MEMORY ************************************************/


template<typename R, int dimension>
SynchronizedMemory<R, dimension>::SynchronizedMemory(mshadow::Shape<dimension> dim, PreferredDevice _preferred_device) :
#ifdef DALI_USE_CUDA
    MemoryMover(false, false, _preferred_device),
#else
    MemoryMover(false, _preferred_device),
#endif
#ifdef DALI_USE_CUDA
    mem_gpu(dim),
    allocated_gpu_(false),
#endif
    allocated_cpu_(false),
    mem_cpu(dim) {
}

template<typename R, int dimension>
SynchronizedMemory<R,dimension>::SynchronizedMemory(const SynchronizedMemory& other) :
#ifdef DALI_USE_CUDA
    MemoryMover(false, false, other.preferred_device),
#else
    MemoryMover(false, other.preferred_device),
#endif
#ifdef DALI_USE_CUDA
        allocated_gpu_(false),
        mem_gpu(other.mem_gpu.shape_),
#endif
        allocated_cpu_(false),
        mem_cpu(other.mem_cpu.shape_) {
    if (other.cpu_fresh) {
        const auto& data_source = other.cpu_data();
        copy_data_from(data_source);
#ifdef DALI_USE_CUDA
    } else if (other.gpu_fresh) {
        const auto& data_source = other.gpu_data();
        copy_data_from(data_source);
#endif
    } else {
        // data was not initialized on the source
        // so we also choose not to initialize.
        return;
    }
}

template<typename R, int dimension>
unsigned int SynchronizedMemory<R,dimension>::number_of_elements() const {
    unsigned int dim = 1;
    for (int i = 0; i < dimension;i++) {
        dim *= mem_cpu.shape_.shape_[i];
    }
    return dim;
}

template<typename R, int dimension>
mshadow::Shape<dimension> SynchronizedMemory<R,dimension>::shape() const {
    return mem_cpu.shape_;
}


template<typename R, int dimension>
SynchronizedMemory<R,dimension>::~SynchronizedMemory() {
    if (allocated_cpu_)
        FreeSpace(&mem_cpu);
#ifdef DALI_USE_CUDA
    if (allocated_gpu_) {
        FreeSpace(&mem_gpu);
    }
#endif
}

template<typename R, int dimension>
const typename SynchronizedMemory<R,dimension>::cpu_tensor_t & SynchronizedMemory<R,dimension>::cpu_data() const {
    to_cpu();
    return mem_cpu;
}

template<typename R, int dimension>
typename SynchronizedMemory<R,dimension>::cpu_tensor_t & SynchronizedMemory<R,dimension>::mutable_cpu_data() {
    to_cpu();
#ifdef DALI_USE_CUDA
    this->gpu_fresh = false;
#endif
    return mem_cpu;
}

#ifdef DALI_USE_CUDA
    template<typename R, int dimension>
    const Tensor<mshadow::gpu, dimension, R>& SynchronizedMemory<R,dimension>::gpu_data() const {
        to_gpu();
        return mem_gpu;
    }

    template<typename R, int dimension>
    Tensor<mshadow::gpu, dimension, R>& SynchronizedMemory<R,dimension>::mutable_gpu_data() {
        to_gpu();
        this->cpu_fresh = false;
        return mem_gpu;
    }
#endif

#ifdef DALI_USE_CUDA
    template<typename R, int dimension>
    void SynchronizedMemory<R,dimension>::to_gpu() const {
        if (!this->gpu_fresh) {
            if (!allocated_gpu_) {
                std::cout << utils::blue << "    allocating gpu" << utils::reset_color << std::endl;
                AllocSpace(&mem_gpu);
                allocated_gpu_ = true;
            }
            if (this->cpu_fresh) {
                std::cout << utils::blue <<  "    copying from cpu" << utils::reset_color << std::endl;
                Copy(mem_gpu, mem_cpu);
            }
            this->gpu_fresh = true;
        } else {
            std::cout << utils::blue << "    gpu fresh" << utils::reset_color << std::endl;
        }
    }
#endif

template<typename R, int dimension>
void SynchronizedMemory<R,dimension>::to_cpu() const {
    if (!this->cpu_fresh) {
        if (!allocated_cpu_) {
            AllocSpace(&mem_cpu);
            allocated_cpu_ = true;
        }
#ifdef DALI_USE_CUDA
        if (this->gpu_fresh) {
            Copy(mem_cpu, mem_gpu);
        }
#endif
        this->cpu_fresh = true;
    }
}

template<typename R, int dimension>
bool SynchronizedMemory<R, dimension>::allocated_cpu() const {
    return allocated_cpu_;
}
#ifdef DALI_USE_CUDA
template<typename R, int dimension>
bool SynchronizedMemory<R, dimension>::allocated_gpu() const {
    return allocated_gpu_;
}
#endif

template<typename R, int dimension>
template<typename SourceType>
void SynchronizedMemory<R,dimension>::copy_data_from(SourceType& data_source) {
    if (this->prefers_cpu()) {
        AllocSpace(&mem_cpu);
        allocated_cpu_ = true;
        Copy(mem_cpu, data_source);
        this->cpu_fresh = true;
    } else {
#ifdef DALI_USE_CUDA
        AllocSpace(&mem_gpu);
        allocated_gpu_ = true;
        Copy(mem_gpu, data_source);
        this->gpu_fresh = true;
#endif
    }
}

template class SynchronizedMemory<float, 1>;
template class SynchronizedMemory<double,1>;
template class SynchronizedMemory<float, 2>;
template class SynchronizedMemory<double,2>;
// template class SynchronizedMemory<float, 3>;
// template class SynchronizedMemory<double,3>;
// template class SynchronizedMemory<float, 4>;
// template class SynchronizedMemory<double,4>;
// template class SynchronizedMemory<float, 5>;
// template class SynchronizedMemory<double,5>;
// template class SynchronizedMemory<float, 6>;
// template class SynchronizedMemory<double,6>;
// template class SynchronizedMemory<float, 7>;
// template class SynchronizedMemory<double,7>;
// template class SynchronizedMemory<float, 8>;
// template class SynchronizedMemory<double,8>;
// template class SynchronizedMemory<float, 9>;
// template class SynchronizedMemory<double,9>;


