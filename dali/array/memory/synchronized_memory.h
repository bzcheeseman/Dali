#ifndef DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H
#define DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H

#include <atomic>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <variant.hpp>
#include <vector>

#include "dali/config.h"
#include "dali/array/memory/device.h"
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


    class SynchronizedMemory {
        private:
            struct DeviceMemory {
                DeviceMemory();
                void* ptr;
                bool allocated;
                bool fresh;
            };


            // whether memory must be filled with zeros on allocation
            // defaults to false
            bool clear_on_allocation;


            // by convention device_memories[0..(MAX_GPU_DEVICES-1) are gpu
            // devices and device_memories[MAX_GPU_DEVICES] is the cpu device.
            static const int DEVICE_MEMORIES_SIZE = MAX_GPU_DEVICES + 1;

            mutable DeviceMemory device_memories[DEVICE_MEMORIES_SIZE];

            DeviceMemory& get_device_memory(Device device) const;
            Device idx_to_device(int idx) const;
            void mark_fresh(const Device& device);
            void mark_all_not_fresh();
            void free_device_memory(const Device& device, DeviceMemory& dev_memory) const;
        public:
            Device preferred_device;
            // total amount of memory expressed in number or Dtypes
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

            // fill memory with zeros if it's been allocated, else
            // ask for memory to be cleared on allocation
            void lazy_clear();
            // immediately clear the memory and allocate if necessary
            void clear();

            bool allocate(const Device& device) const;
            void free(const Device& device) const;

            // Ensure a fresh copy of the memory is on the device
            void move_to(const Device& device) const;



            // depending on how memory is accessed, its freshess changes:
            // calling `data` without mutable asks for a fresh copy
            // of the memory, but promises to not modify it
            void* data(const Device& device) const;
            // mutable_cpu_data on the other hand declares that the memory
            // will be modified, and thus will need to be resynchronized
            // if a different device needs it (cpu vs. gpu freshness)
            void* mutable_data(const Device& device);
            // just like mutable_data, except the memory is not guaranteed to
            // be fresh (e.g. you should ignore its contents, as they could be
            // stale).
            void* overwrite_data(const Device& device);
    };
}  // namespace memory

#endif
