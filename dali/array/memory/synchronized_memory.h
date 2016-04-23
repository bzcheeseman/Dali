#ifndef DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H
#define DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H

#include <atomic>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>

#include "dali/config.h"
#include "dali/array/memory/device.h"
#include "dali/array/dtype.h"
#include "dali/utils/core_utils.h"
#include "dali/runtime_config.h"


/*
Synchronized Memory
-------------------

SynchronizedMemory wraps N tensors for all GPUs and CPU
and remembers how fresh each version is.
This class minimizes transfers between GPUs and CPU by keeping
track of which device the master copy is on, and if an inter-
device copy is necessary.
*/


namespace memory {
    // access mode
    enum AM {
        AM_READONLY,
        AM_MUTABLE,
        AM_OVERWRITE
    };

    class SynchronizedMemory {
        public:
            struct DeviceMemory {
                DeviceMemory();
                void* ptr;
                bool allocated;
                bool fresh;
            };
        private:
            // whether memory must be filled with zeros on allocation
            // defaults to false
            bool clear_on_allocation;

            mutable DeviceMemory cpu_memory;
#ifdef DALI_USE_CUDA
            mutable DeviceMemory gpu_memories[MAX_GPU_DEVICES];
#endif
            // helper functions
            DeviceMemory& get_device_memory(Device device) const;
            typedef std::function<void(const Device&,DeviceMemory&)> device_iterator_t;
            void iterate_device_memories(device_iterator_t f) const;
            void mark_fresh(const Device& device);
            void mark_all_not_fresh();
            void free_device_memory(const Device& device, DeviceMemory& dev_memory) const;
        public:

            Device preferred_device;
            // total amount of memory expressed in number of bytes
            const int total_memory;
            // hint for inner dimension. Must divide total_memory.
            const int inner_dimension;

            SynchronizedMemory(int total_size,
                               int inner_dimension=1,
                               Device preferred_device=default_preferred_device,
                               bool clear_on_allocation=false);

           // inherits preferred device and copies memory to it.
           SynchronizedMemory(const SynchronizedMemory& other);
           ~SynchronizedMemory();

           SynchronizedMemory& operator=(const SynchronizedMemory&) = delete;

           // returns true if memory is fresh on device
           bool is_fresh(const Device& device) const;

           // returns true if memory is allocated on device
           bool is_allocated(const Device& device) const;

           // returns first device in the list that is fresh
           // and if no devices are fresh return device of
           // DOOM.
           memory::Device find_some_fresh_device() const;

           // returns true if any of the devices has fresh memory.
           bool is_any_fresh() const;

           // returns true if any of the devices has an allocation
           bool is_any_allocated() const;

           // fill memory with zeros if it's been allocated, else
           // ask for memory to be cleared on allocation
           void lazy_clear();
           // immediately clear the memory and allocate if necessary
           void clear();

           bool allocate(const Device& device) const;
           void free(const Device& device) const;

           // Ensure a fresh copy of the memory is on the device
           void move_to(const Device& device) const;
#ifdef DALI_USE_CUDA
           // Ensure a fresh copy of the memory is on gpu[number]
           void to_gpu(const int& gpu_number) const;
#endif
           // Ensure a fresh copy of the memory is on the cpu
           void to_cpu() const;

           // This function selects on of the 3 functions below based on
           // access_mode value (so it is sometimes const even if not
           // explicitly marked as such)
           void* data(const Device& device, AM access_mode=AM_READONLY);
           // depending on how memory is accessed, its freshess changes:
           // calling `data` without mutable asks for a fresh copy
           // of the memory, but promises to not modify it
           void* readonly_data(const Device& device) const;
           // mutable_cpu_data on the other hand declares that the memory
           // will be modified, and thus will need to be resynchronized
           // if a different device needs it (cpu vs. gpu freshness)
           void* mutable_data(const Device& device);
           // just like mutable_data, except the memory is not guaranteed to
           // be fresh (e.g. you should ignore its contents, as they could be
           // stale).
           void* overwrite_data(const Device& device);

           // prints various memory statistics. For debug use only.
           void debug_info(std::basic_ostream<char>& stream = std::cout, bool print_contents=false, DType dtype=DTYPE_FLOAT) const;
    };

    namespace debug {
        extern SynchronizedMemory::DeviceMemory fake_device_memories[MAX_FAKE_DEVICES];
    }
}  // namespace memory

#endif
