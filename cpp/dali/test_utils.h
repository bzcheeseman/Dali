#ifndef DALI_TEST_UTILS_H
#define DALI_TEST_UTILS_H

#include <gperftools/heap-checker.h>
#include <gtest/gtest.h>
#include <memory>

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
#endif
