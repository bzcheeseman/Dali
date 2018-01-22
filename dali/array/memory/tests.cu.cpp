#include <chrono>
#include <vector>
#include <thread>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/config.h"

#include "dali/array/memory/device.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/array/memory/memory_bank.h"
#include "dali/array/memory/synchronized_memory.h"
#include "dali/utils/print_utils.h"

using std::vector;
using std::chrono::milliseconds;


using memory::Device;
using memory::DevicePtr;
using memory::SynchronizedMemory;

TEST(MemoryTests, alloc_cpu) {
    for (int alloc_size = 100; alloc_size <= 10000000; alloc_size *= 7) {
        auto mem = memory::allocate(Device::cpu(), alloc_size, 10);
        memory::free(mem, alloc_size, 10);
    }
}

#ifdef DALI_USE_CUDA
    TEST(MemoryTests, alloc_gpu) {
        for (int alloc_size = 100; alloc_size <= 10000000; alloc_size *= 7) {
            auto mem = memory::allocate(Device::gpu(0), alloc_size, 10);
            memory::free(mem, alloc_size, 10);
        }
    }
#endif

TEST(MemoryTests, test_memory_bank_cpu) {
    auto device = Device::cpu();
    auto mem  = memory::bank::allocate(device, 100, 4);
    auto first_ptr = mem.ptr;
    memory::bank::deposit(mem, 100, 4);

    // reallocating memory of the same size, expecting to reuse
    auto mem2 = memory::bank::allocate(device, 100, 4);
    EXPECT_EQ(mem2.ptr, first_ptr);
    memory::bank::deposit(mem2, 100, 4);

    // reallocating memory of the same size, expecting not to reuse
    auto mem3 = memory::bank::allocate(device, 120, 4);
    EXPECT_NE(mem3.ptr, first_ptr);
    memory::bank::deposit(mem3, 120, 4);

    memory::bank::clear(device);
}


#ifdef DALI_USE_CUDA
    TEST(MemoryTests, test_memory_bank_gpu) {
        auto device = Device::gpu(0);
        auto mem  = memory::bank::allocate(device, 100, 4);
        auto first_ptr = mem.ptr;
        memory::bank::deposit(mem, 100, 4);

        // reallocating memory of the same size, expecting to reuse
        auto mem2 = memory::bank::allocate(device, 100, 4);
        EXPECT_EQ(mem2.ptr, first_ptr);
        memory::bank::deposit(mem2, 100, 4);

        // reallocating memory of the same size, expecting not to reuse
        auto mem3 = memory::bank::allocate(device, 120, 4);
        EXPECT_NE(mem3.ptr, first_ptr);
        memory::bank::deposit(mem3, 120, 4);

        memory::bank::clear(device);
    }

    TEST(MemoryTests, copy_test) {
        auto mem_cpu1 = memory::allocate(Device::cpu(),  4, 4);
        auto mem_cpu2 = memory::allocate(Device::cpu(),  4, 4);
        auto mem_gpu1 = memory::allocate(Device::gpu(0), 4, 4);
        auto mem_gpu2 = memory::allocate(Device::gpu(0), 4, 4);
        int* mem_cpu1_ptr = (int*)mem_cpu1.ptr;
        int* mem_cpu2_ptr = (int*)mem_cpu2.ptr;

        *mem_cpu1_ptr = 42;
        memory::copy(mem_gpu1, mem_cpu1, 4, 4);  // CPU -> GPU
        memory::copy(mem_gpu2, mem_gpu1, 4, 4);  // GPU -> GPU
        memory::copy(mem_cpu2, mem_gpu2, 4, 4);  // GPU -> CPU
        EXPECT_EQ(*mem_cpu2_ptr, 42);

        *mem_cpu1_ptr = 69;
        memory::copy(mem_cpu2, mem_cpu1, 4, 4);  // CPU->CPU
        EXPECT_EQ(69, *mem_cpu1_ptr);
        EXPECT_EQ(69, *mem_cpu2_ptr);

        memory::free(mem_cpu1, 4, 4);
        memory::free(mem_cpu2, 4, 4);
        memory::free(mem_gpu1, 4, 4);
        memory::free(mem_gpu2, 4, 4);
    }
#endif

    TEST(MemoryTests, synchronized_memory_copy) {
        SynchronizedMemory s(12, 1, Device::cpu(), false);
        auto data = (uint8_t*)s.overwrite_data(Device::cpu());
        for (int i=0; i < 12; ++i) {
            data[i] = i;
        }
        SynchronizedMemory copied(s);
        auto copied_data = (uint8_t*)copied.data(Device::cpu());
        for (int i=0; i < 12; ++i) {
            ASSERT_EQ(copied_data[i], i);
        }
    }

    TEST(MemoryTests, fake_devices) {
        memory::debug::enable_fake_devices = false;
        EXPECT_THROW(Device::fake(1), std::runtime_error);
        memory::debug::enable_fake_devices = true;
        SynchronizedMemory s(12, 1, Device::fake(1), true);
        s.is_fresh(Device::fake(1));
        memory::debug::fake_device_memories[1].fresh = true;
        EXPECT_EQ(s.is_fresh(Device::fake(1)), true);
        memory::debug::fake_device_memories[1].fresh = false;
        EXPECT_EQ(s.is_fresh(Device::fake(1)), false);
        memory::debug::enable_fake_devices = false;
    }

#ifdef DALI_USE_CUDA
    TEST(MemoryTests, synchronized_memory_clear_on_alloc) {
        SynchronizedMemory s(12, 1, Device::cpu(), true);
        auto data = (uint8_t*)s.data(Device::cpu());
        for (int i=0; i < 12; ++i) {
            ASSERT_EQ(data[i], 0);
        }
    }

    TEST(MemoryTests, synchronized_memory_cpu_gpu_sync) {
        // create CPU memory
        SynchronizedMemory s(12, 1, Device::cpu(), false);
        // get pointer to cpu data
        auto data = (uint8_t*)s.overwrite_data(Device::cpu());
        // and assign to it 0..11
        for (int i = 0; i < 12; ++i) {
            data[i] = i;
        }

        // we will now try to copy this data over to the gpu
        auto data_gpu = s.data(Device::gpu(0));

        // we allocate a new chunk of CPU memory
        // to receive the memory that round-triped through the gpu
        auto data_gpu_as_cpu = memory::allocate(Device::cpu(), 12, 1);
        // execute the copy "manually"
        memory::copy(data_gpu_as_cpu, DevicePtr(Device::gpu(0), data_gpu), 12, 1);

        // get a pointer to underlying memory buffer
        auto data_gpu_as_cpu_ptr = (uint8_t*)data_gpu_as_cpu.ptr;

        // check that it is equal
        for (int i=0; i < 12; ++i) {
            ASSERT_EQ(data_gpu_as_cpu_ptr[i], i);
        }
    }
    namespace {
        void __global__ add_value(int* data, int value, int n) {
            int stride = blockDim.x * gridDim.x;
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            for (int i = idx; i < n; i += stride) {
                data[i] += value;
            }
        }
    }

    TEST(MemoryTests, synchronized_memory_gpu_cpu_sync) {
        // In this test we make modifications to GPU
        // memory and check that the copied values on the CPU
        // line-up with what was done there.

        // create GPU memory
        int n = 12;
        int value = 13;
        SynchronizedMemory s(
            n * sizeof(int),
            1,
            Device::gpu(0),
            /*clear_on_allocation=*/true
        );
        int nthreads = 256;
        int grid_size = (n + nthreads - 1) / nthreads;
        // make some modifications to the data
        add_value<<<grid_size, nthreads>>>(static_cast<int*>(s.mutable_data(Device::gpu(0))), value, n);
        // memory on device should now all be equal to 13
        auto cpu_data = (int*)s.data(Device::cpu());
        // check that it is equal
        for (int i=0; i < n; ++i) {
            ASSERT_EQ(cpu_data[i], value);
        }
    }

#endif
