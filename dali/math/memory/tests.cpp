#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/config.h"
#include "dali/math/memory/memory_ops.h"
#include "dali/math/memory/memory_bank.h"

using std::vector;
using std::chrono::milliseconds;


using memory_ops::DEVICE_CPU;
#ifdef DALI_USE_CUDA
using memory_ops::DEVICE_GPU;
#endif

TEST(MemoryTests, alloc_cpu) {
    for (int alloc_size = 100; alloc_size <= 10000000; alloc_size *= 7) {
        auto mem = memory_ops::allocate(DEVICE_CPU, alloc_size, 10);
        memory_ops::free(mem, alloc_size, 10);
    }
}

#ifdef DALI_USE_CUDA
    TEST(MemoryTests, alloc_gpu) {
        for (int alloc_size = 100; alloc_size <= 10000000; alloc_size *= 7) {
            auto mem = memory_ops::allocate(DEVICE_GPU, alloc_size, 10);
            memory_ops::free(mem, alloc_size, 10);
        }
    }
#endif

TEST(MemoryTests, test_memory_bank_cpu) {
    auto device = DEVICE_CPU;
    auto mem  = memory_bank::allocate(device, 100, 4);
    auto first_ptr = mem.ptr;
    memory_bank::deposit(mem, 100, 4);

    // reallocating memory of the same size, expecting to reuse
    auto mem2 = memory_bank::allocate(device, 100, 4);
    EXPECT_EQ(mem2.ptr, first_ptr);
    memory_bank::deposit(mem2, 100, 4);

    // reallocating memory of the same size, expecting not to reuse
    auto mem3 = memory_bank::allocate(device, 120, 4);
    EXPECT_NE(mem3.ptr, first_ptr);
    memory_bank::deposit(mem3, 120, 4);

    memory_bank::clear(device);
}


#ifdef DALI_USE_CUDA
    TEST(MemoryTests, test_memory_bank_gpu) {
        auto device = DEVICE_GPU;
        auto mem  = memory_bank::allocate(device, 100, 4);
        auto first_ptr = mem.ptr;
        memory_bank::deposit(mem, 100, 4);

        // reallocating memory of the same size, expecting to reuse
        auto mem2 = memory_bank::allocate(device, 100, 4);
        EXPECT_EQ(mem2.ptr, first_ptr);
        memory_bank::deposit(mem2, 100, 4);

        // reallocating memory of the same size, expecting not to reuse
        auto mem3 = memory_bank::allocate(device, 120, 4);
        EXPECT_NE(mem3.ptr, first_ptr);
        memory_bank::deposit(mem3, 120, 4);

        memory_bank::clear(device);
    }

    TEST(MemoryTests, copy_test) {
        auto mem_cpu1 = memory_ops::allocate(DEVICE_CPU, 4, 4);
        auto mem_cpu2 = memory_ops::allocate(DEVICE_CPU, 4, 4);
        auto mem_gpu1 = memory_ops::allocate(DEVICE_GPU, 4, 4);
        auto mem_gpu2 = memory_ops::allocate(DEVICE_GPU, 4, 4);
        int* mem_cpu1_ptr = (int*)mem_cpu1.ptr;
        int* mem_cpu2_ptr = (int*)mem_cpu2.ptr;

        *mem_cpu1_ptr = 42;
        memory_ops::copy(mem_gpu1, mem_cpu1, 4, 4);  // CPU -> GPU
        memory_ops::copy(mem_gpu2, mem_gpu1, 4, 4);  // GPU -> GPU
        memory_ops::copy(mem_cpu2, mem_gpu2, 4, 4);  // GPU -> CPU
        EXPECT_EQ(*mem_cpu2_ptr, 42);

        *mem_cpu1_ptr = 69;
        memory_ops::copy(mem_cpu2, mem_cpu1, 4, 4);  // CPU->CPU
        EXPECT_EQ(69, *mem_cpu1_ptr);
        EXPECT_EQ(69, *mem_cpu2_ptr);

        memory_ops::free(mem_cpu1, 4, 4);
        memory_ops::free(mem_cpu2, 4, 4);
        memory_ops::free(mem_gpu1, 4, 4);
        memory_ops::free(mem_gpu2, 4, 4);
    }
#endif
