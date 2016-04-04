#ifndef DALI_ARRAY_ARRAY_H
#define DALI_ARRAY_ARRAY_H

#include <memory>
#include <vector>

#include "dali/array/memory/device.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/array/memory/synchronized_memory.h"

struct ArrayState {
    std::vector<int> shape;
    std::shared_ptr<memory::SynchronizedMemory> memory;
    int offset; // expressing in number of numbers (not bytes)
    DType dtype;
    ArrayState(const std::vector<int>& _shape, std::shared_ptr<memory::SynchronizedMemory> _memory, int _offset, DType _device);
};

class Array {
    private:
        std::shared_ptr<ArrayState> state;
    public:
      typedef uint index_t;
      Array();

      Array(const std::vector<int>& shape, DType dtype_=DTYPE_FLOAT);
      Array(std::initializer_list<int> shape, DType dtype_=DTYPE_FLOAT);
      Array(const std::vector<int>& shape, std::shared_ptr<memory::SynchronizedMemory>, int offset, DType dtype_=DTYPE_FLOAT);

      Array(const Array& other, bool copy_memory=false);

      int dimension() const;
      const std::vector<int>& shape() const;
      std::vector<int> subshape() const;

      Array operator[](index_t idx) const;
      void* operator()(index_t idx) const;



      void print(std::basic_ostream<char>& stream = std::cout, int indent=0) const;
};

#endif
