#ifdef DONT_COMPILE
#include "dali/array/SynchronizedMemory.h"

#include "dali/array/memory_bank/MemoryBank.h"

#ifdef DALI_USE_CUDA
    Device default_preferred_device = DEVICE_GPU;
#else
    Device default_preferred_device = DEVICE_CPU;
#endif

/**** SHOULD COMPUTE GPU-land **/

template<typename R>
void SynchronizedMemory<R>::clear_cpu() {
    memory_bank<R>::clear_cpu();
}

#ifdef DALI_USE_CUDA
    template<typename R>
    void SynchronizedMemory<R>::clear_gpu() {
        memory_bank<R>::clear_gpu();
    }
#endif

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
template bool should_compute_on_gpu(const std::vector<const SynchronizedMemory<int>*>& sts);

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
                                          Device _preferred_device,
                                          bool _clear_on_allocation) :
#ifdef DALI_USE_CUDA
        gpu_fresh(false),
        allocated_gpu(false),
        gpu_ptr(NULL),
#endif
        clear_on_allocation(_clear_on_allocation),
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
        SynchronizedMemory(other.total_memory, other.inner_dimension, other.preferred_device, other.clear_on_allocation) {
    if (other.cpu_fresh && this->prefers_cpu()) {
        allocate_cpu();
        memory_operations<R>::copy_memory_cpu_to_cpu(this->cpu_ptr, other.cpu_ptr, total_memory, inner_dimension);
        this->cpu_fresh = true;
    }
#ifdef DALI_USE_CUDA
    else if (other.cpu_fresh && this->prefers_gpu()) {
        allocate_gpu();
        memory_operations<R>::copy_memory_cpu_to_gpu(this->gpu_ptr, other.cpu_ptr, total_memory, inner_dimension);
        this->gpu_fresh = true;
    } else if (other.gpu_fresh && this->prefers_cpu()) {
        allocate_cpu();
        memory_operations<R>::copy_memory_gpu_to_cpu(this->cpu_ptr, other.gpu_ptr, total_memory, inner_dimension);
        this->cpu_fresh = true;
    } else if (other.gpu_fresh && this->prefers_gpu()) {
        allocate_gpu();
        memory_operations<R>::copy_memory_gpu_to_gpu(this->gpu_ptr, other.gpu_ptr, total_memory, inner_dimension);
        this->gpu_fresh = true;
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
        memory_bank<R>::deposit_cpu(total_memory, inner_dimension, cpu_ptr);
        cpu_ptr = NULL;
    }
    allocated_cpu = false;
}

#ifdef DALI_USE_CUDA
template<typename R>
void SynchronizedMemory<R>::free_gpu() const {
    if (allocated_gpu) {
        memory_bank<R>::deposit_gpu(total_memory, inner_dimension, gpu_ptr);
        gpu_ptr = NULL;
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


template<typename R>
void SynchronizedMemory<R>::clear() {
    clear_on_allocation = true;
    #ifdef DALI_USE_CUDA
    if (preferred_device == DEVICE_GPU) {
        allocate_gpu();
        memory_operations<R>::clear_gpu_memory(this->gpu_ptr, total_memory, inner_dimension);
        this->cpu_fresh = false;
        this->gpu_fresh = true;
        return;
    }
    #endif
    if (preferred_device == DEVICE_CPU) {
        allocate_cpu();
        memory_operations<R>::clear_cpu_memory(this->cpu_ptr, total_memory, inner_dimension);
        this->cpu_fresh = true;
        #ifdef DALI_USE_CUDA
            this->gpu_fresh = false;
        #endif
    }
}

template<typename R>
void SynchronizedMemory<R>::lazy_clear() {
    clear_on_allocation = true;
    cpu_fresh = false;

    #ifdef DALI_USE_CUDA
        gpu_fresh = false;
        if (!allocated_cpu && !allocated_cpu) {
            return;
        }
    #else
        if (!allocated_cpu) {
            return;
        }
    #endif
    clear();
}

#ifdef DALI_USE_CUDA
    template<typename R>
    bool SynchronizedMemory<R>::allocate_gpu() const {
        if (allocated_gpu) {
            return false;
        }
        gpu_ptr = memory_bank<R>::allocate_gpu( total_memory , inner_dimension );
        allocated_gpu = true;
        return true;
    }

    template<typename R>
    void SynchronizedMemory<R>::to_gpu() const {
        if (!this->gpu_fresh) {
            auto just_allocated_gpu = allocate_gpu();
            // now that memory was freshly allocated
            // on gpu we either copy the CPU data over
            // or clear the buffer if the `clear_on_allocation`
            // flag is true:
            if (this->cpu_fresh) {
                memory_operations<R>::copy_memory_cpu_to_gpu(this->gpu_ptr, this->cpu_ptr, total_memory, inner_dimension);

            } else if (just_allocated_gpu && clear_on_allocation) {
                memory_operations<R>::clear_gpu_memory(this->gpu_ptr, total_memory, inner_dimension);
            }
            this->gpu_fresh = true;
        }
    }
#endif

template<typename R>
bool SynchronizedMemory<R>::allocate_cpu() const {
    if (allocated_cpu) {
        return false;
    }
    cpu_ptr = memory_bank<R>::allocate_cpu( total_memory , inner_dimension );
    allocated_cpu = true;
    return true;
}

template<typename R>
void SynchronizedMemory<R>::to_cpu() const {
    if (!this->cpu_fresh) {
        auto just_allocated_cpu = allocate_cpu();
#ifdef DALI_USE_CUDA
        if (this->gpu_fresh) {
            memory_operations<R>::copy_memory_gpu_to_cpu(this->cpu_ptr, this->gpu_ptr, total_memory, inner_dimension);
        } else if (just_allocated_cpu && clear_on_allocation) {
            memory_operations<R>::clear_cpu_memory(this->cpu_ptr, total_memory, inner_dimension);
        }
#else
        if (just_allocated_cpu && clear_on_allocation) {
            memory_operations<R>::clear_cpu_memory(this->cpu_ptr, total_memory, inner_dimension);
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
    #endif
    return cpu_ptr;
}
template <typename R>
R* SynchronizedMemory<R>::overwrite_cpu_data() {
    #ifdef DALI_USE_CUDA
        gpu_fresh = false;
    #endif
    allocate_cpu();
    cpu_fresh = true;
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
template <typename R>
R* SynchronizedMemory<R>::overwrite_gpu_data() {
    cpu_fresh = false;
    allocate_gpu();
    gpu_fresh = true;
    return gpu_ptr;
}

#endif

template class SynchronizedMemory<float>;
template class SynchronizedMemory<int>;
template class SynchronizedMemory<double>;


#endif
