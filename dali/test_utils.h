#ifndef DALI_TEST_UTILS_H
#define DALI_TEST_UTILS_H

#ifdef GPERFTOOLS_FOUND
    #include <gperftools/heap-checker.h>
#endif

#include <gtest/gtest.h>
#include <memory>

#ifdef GPERFTOOLS_FOUND
    class MemorySafeTest : public ::testing::Test {
        private:
            std::shared_ptr<HeapLeakChecker> heap_checker;
        protected:
            virtual void SetUp() {
                if (HeapLeakChecker::IsActive())
                    heap_checker = std::make_shared<HeapLeakChecker>("memory_leak_checker");
            }

            virtual void TearDown() {
                if (HeapLeakChecker::IsActive())
                    ASSERT_TRUE(heap_checker->NoLeaks()) << "Memory Leak";
            }

    };
#else
    class MemorySafeTest : public ::testing::Test {
    };
#endif
#endif
