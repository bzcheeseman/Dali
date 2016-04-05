#include "dali/array/memory/synchronized_memory.h"

#include <cassert>

#include "dali/array/memory/memory_bank.h"

namespace memory {
    ////////////////////////////////////////////////////////////////////////////////
    //                           DEVICE MEMORY                                    //
    ////////////////////////////////////////////////////////////////////////////////

    SynchronizedMemory::DeviceMemory::DeviceMemory() : ptr(NULL), allocated(false), fresh(false) {}

    ////////////////////////////////////////////////////////////////////////////////
    //                     SYNCHRONIZED MEMORY                                    //
    ////////////////////////////////////////////////////////////////////////////////

    Device SynchronizedMemory::idx_to_device(int idx) const {
        if (idx == memory::MAX_GPU_DEVICES) {
            return Device::cpu();
        }
#ifdef DALI_USE_CUDA
        else if(0 <= idx && idx < memory::MAX_GPU_DEVICES) {
            return Device::gpu(idx);
        }
#endif
        else {
            ASSERT2(false, "Wrong idx passed to idx_to_device");
            return Device::device_of_doom();
        }
    }


    SynchronizedMemory::DeviceMemory& SynchronizedMemory::get_device_memory(Device device) const {
        if (device.is_cpu()) {
            return device_memories[memory::MAX_GPU_DEVICES];
        }
#ifdef DALI_USE_CUDA
        else if(device.is_gpu()) {
            return device_memories[device.number];
        }
#endif
        else {
            ASSERT2(false, "Unsupported device passed to SynchronizedMemory.");
            return device_memories[0];
        }
    }

    void SynchronizedMemory::mark_fresh(const Device& device) {
        get_device_memory(device).fresh = true;
    }

    void SynchronizedMemory::mark_all_not_fresh() {
        for (int i=0; i < DEVICE_MEMORIES_SIZE; ++i) {
            device_memories[i].fresh = false;
        }
    }

    void SynchronizedMemory::free_device_memory(const Device& device, SynchronizedMemory::DeviceMemory& dev_memory) const {
        if (dev_memory.allocated) {
            memory::bank::deposit(DevicePtr(device, dev_memory.ptr), total_memory, inner_dimension);
            dev_memory.ptr = (void*)NULL;
            dev_memory.allocated = false;
        }
    }

    SynchronizedMemory::SynchronizedMemory(int _total_memory,
                                           int _inner_dimension,
                                           Device _preferred_device,
                                           bool _clear_on_allocation) :
            total_memory(_total_memory),
            inner_dimension(_inner_dimension),
            preferred_device(_preferred_device),
            clear_on_allocation(_clear_on_allocation) {
        assert(total_memory % inner_dimension == 0);
    }

    SynchronizedMemory::SynchronizedMemory(const SynchronizedMemory& other) :
            SynchronizedMemory(other.total_memory, other.inner_dimension, other.preferred_device, other.clear_on_allocation) {


        allocate(preferred_device);
        auto dest = DevicePtr(preferred_device, get_device_memory(preferred_device).ptr);

        // find a source device, such that the copying process will have the least overhead.
        int source_device_idx = -1;
        bool device_type_matching = false;
        bool device_number_match = false;

        for (int i = 0; i < DEVICE_MEMORIES_SIZE; ++i) {
            if (other.device_memories[i].fresh) {
                auto current_device                 = idx_to_device(i);
                bool current_device_type_matching   = current_device.type == preferred_device.type;
                bool current_device_number_matching = current_device.number == preferred_device.number;

                bool update = false;
                if (source_device_idx == -1) {
                    update = true;
                } else if (!device_type_matching && current_device_type_matching) {
                    update = true;
                } else if (!device_number_match && current_device_number_matching) {
                    update = true;
                }

                if (update) {
                    source_device_idx        = i;
                    device_type_matching = current_device_type_matching;
                    device_number_match  = current_device_number_matching;
                }
            }
        }

        // TODO(szymon): if we do this with two different gpus death.

        if (source_device_idx == -1) return;

        auto source = DevicePtr(idx_to_device(source_device_idx), other.device_memories[source_device_idx].ptr);

        memory::copy(dest, source, total_memory, inner_dimension);
        get_device_memory(preferred_device).fresh = true;
    }

    SynchronizedMemory::~SynchronizedMemory() {
        for (int i=0; i < DEVICE_MEMORIES_SIZE; ++i) {
            free_device_memory(idx_to_device(i), device_memories[i]);
        }
    }


    void SynchronizedMemory::clear() {
        clear_on_allocation = true;

        this->allocate(preferred_device);
        auto ptr = get_device_memory(preferred_device).ptr;
        memory::clear(DevicePtr(preferred_device, ptr), total_memory, inner_dimension);

        mark_all_not_fresh();
        mark_fresh(preferred_device);
    }


    void SynchronizedMemory::lazy_clear() {
        clear_on_allocation = true;

        mark_all_not_fresh();

        // clear immediately if some memory was already allocated.
        for (int i=0; i < DEVICE_MEMORIES_SIZE; ++i) {
            if (device_memories[i].allocated) {
                clear();
                return;
            }
        }
    }

    bool SynchronizedMemory::allocate(const Device& device) const {
        auto& dev_memory = get_device_memory(device);
        if (dev_memory.allocated) {
            return false;
        }

        auto dev_ptr = memory::bank::allocate(device, total_memory, inner_dimension);
        dev_memory.ptr = dev_ptr.ptr;
        dev_memory.allocated = true;
        return true;
    }


    void SynchronizedMemory::free(const Device& device) const {
        auto& dev_memory = get_device_memory(device);
        free_device_memory(device, dev_memory);
    }

    void SynchronizedMemory::move_to(const Device& device) const {
        auto& dev_memory = get_device_memory(device);

        // if memory is fresh, we are done
        if (dev_memory.fresh) return;

        // make sure memory is allocated.
        bool just_allocated = allocate(device);

        // if another piece of memory is fresh copy from it.
        for (int i=0; i < DEVICE_MEMORIES_SIZE; ++i) {
            if (device_memories[i].fresh) {
                auto source = DevicePtr(idx_to_device(i), device_memories[i].ptr);
                auto dest   = DevicePtr(device, dev_memory.ptr);
                memory::copy(dest, source, total_memory, inner_dimension);
                dev_memory.fresh = true;
                return;
            }
        }

        // if just allocated, and we need to clear the memory.
        // this only happens if we did not copy memory from elsewhere.
        if (just_allocated && clear_on_allocation) {
            memory::clear(DevicePtr(device, dev_memory.ptr), total_memory, inner_dimension);
        }
        dev_memory.fresh = true;
    }

    void* SynchronizedMemory::data(const Device& device, AM access_mode) {
        if (access_mode == AM_READONLY) {
            return readonly_data(device);
        } else if (access_mode == AM_MUTABLE) {
            return mutable_data(device);
        } else if (access_mode == AM_OVERWRITE) {
            return overwrite_data(device);
        } else {
            ASSERT2(false, "Unsupported access mode passed to SynchronizedMemory::data.");
        }
    }


    void* SynchronizedMemory::readonly_data(const Device& device) const {
        move_to(device);
        return get_device_memory(device).ptr;
    }

    void* SynchronizedMemory::mutable_data(const Device& device) {
        move_to(device);
        mark_all_not_fresh();
        mark_fresh(device);
        return get_device_memory(device).ptr;
    }

    void* SynchronizedMemory::overwrite_data(const Device& device) {
        allocate(device);
        mark_all_not_fresh();
        mark_fresh(device);

        return get_device_memory(device).ptr;
    }
}
