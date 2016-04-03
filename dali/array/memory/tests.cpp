#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/config.h"
#include "dali/array/memory/device.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/array/memory/memory_bank.h"

using std::vector;
using std::chrono::milliseconds;


using memory::Device;

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
