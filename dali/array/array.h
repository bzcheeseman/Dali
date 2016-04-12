#ifndef DALI_ARRAY_ARRAY_H
#define DALI_ARRAY_ARRAY_H

#include <memory>
#include <vector>

#include "dali/array/memory/device.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/array/memory/synchronized_memory.h"
#include "dali/array/lazy_op/expression.h"
#include "dali/runtime_config.h"

class Array;

struct AssignableArray {
    typedef std::function<void(Array&)> assign_t;
    assign_t assign_to;
    AssignableArray(assign_t&& _assign_to);
};

struct ArrayState {
    std::vector<int> shape;
    std::shared_ptr<memory::SynchronizedMemory> memory;
    int offset; // expressing in number of numbers (not bytes)
    DType dtype;
    ArrayState(const std::vector<int>& _shape, std::shared_ptr<memory::SynchronizedMemory> _memory, int _offset, DType _device);
};

class Array : public Exp<Array> {
  private:
    std::shared_ptr<ArrayState> state;
    template<typename T>
    T scalar_value() const;
    template<typename T>
    T& scalar_value();
    template<typename T>
    Array& assign_constant(const T& other);
  public:
    typedef uint index_t;
    Array();

    /* Various ways of constructing array */
    Array(const std::vector<int>& shape, DType dtype_=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array(std::initializer_list<int> shape, DType dtype_=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array(const std::vector<int>& shape, std::shared_ptr<memory::SynchronizedMemory>, int offset, DType dtype_=DTYPE_FLOAT);

    Array(const Array& other, bool copy_memory=false);
    Array(const AssignableArray& assignable);

    template<typename T>
    Array(const T& castable_to_assignable)
            : Array((AssignableArray)castable_to_assignable) {
    }


    static Array zeros(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array zeros_like(const Array& other);

    bool is_stateless() const;
    void initialize(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array& reset();

    /* Accesing internal state */
    const std::vector<int>& shape() const;
    std::shared_ptr<memory::SynchronizedMemory> memory() const;
    int offset() const;
    DType dtype() const;

    /* Shape-related convinence */
    int dimension() const;
    int number_of_elements() const;
    std::vector<int> subshape() const;

    /* Creating a view into memory */
    Array operator[](index_t idx) const;
    Array operator()(index_t idx) const;
    Array ravel() const;
    Array reshape(const std::vector<int>& shape) const;

    /* Interpreting scalars as numbers */
    template<typename T>
    operator T&();

    template<typename T>
    operator const T&() const;

    template<typename T>
    Array& operator=(const T& other) {
        auto assignable = (AssignableArray)other;
        return (*this = assignable);
    }

    Array& operator=(const float& other);
    Array& operator=(const double& other);
    Array& operator=(const int& other);
    Array& operator=(const AssignableArray& assignable);

    /* Debugging */
    void print(std::basic_ostream<char>& stream = std::cout, int indent=0) const;
};

#endif
