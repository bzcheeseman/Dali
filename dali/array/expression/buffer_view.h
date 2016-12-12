#ifndef DALI_ARRAY_EXPRESSION_BUFFER_VIEW_H
#define DALI_ARRAY_EXPRESSION_BUFFER_VIEW_H

#include <memory>
#include <vector>

#include "dali/array/expression/expression.h"
#include "dali/array/memory/synchronized_memory.h"

struct BufferView : public Expression {
    std::shared_ptr<memory::SynchronizedMemory> memory_;
    // condition variable ready_;

    static std::shared_ptr<memory::SynchronizedMemory> create_memory(
            const std::vector<int>& shape,
            DType dtype,
            memory::Device preferred_device);

    BufferView(std::shared_ptr<memory::SynchronizedMemory> memory,
               const std::vector<int>& shape,
               DType dtype,
               int offset,
               const std::vector<int>& strides);

    BufferView(const std::vector<int>& shape,
               DType dtype,
               memory::Device preferred_device,
               int offset=0,
               const std::vector<int>& strides={});

    virtual memory::Device preferred_device() const ;
};

#endif  // DALI_ARRAY_EXPRESSION_BUFFER_VIEW_H
