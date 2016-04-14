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

#include "dali/utils/print_utils.h" // delete me

class Array;

struct AssignableArray {
    typedef std::function<void(Array&)> assign_t;
    assign_t assign_to;

    explicit AssignableArray(assign_t&& _assign_to);
    AssignableArray(const float& constant);
    AssignableArray(const double& constant);
    AssignableArray(const int& constant);
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
    Array(const RValueExp<T>& expr) :
            Array(expr.as_assignable()) {
    }


    static Array zeros(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array zeros_like(const Array& other);

    // true if just creted with empty constructor or reset
    // (has no assossiated memory)
    bool is_stateless() const;
    // true if Array's contents conver entirety of underlying
    // memory (as opposed to offset memory, strided memory etc.).
    bool spans_entire_memory() const;
    void initialize(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array& reset();

    /* Accesing internal state */
    const std::vector<int>& shape() const;
    std::shared_ptr<memory::SynchronizedMemory> memory() const;
    int offset() const;
    DType dtype() const;

    /* memory moving logic */
    memory::Device preferred_device() const;
    void to_device(memory::Device device) const;

    /* Shape-related convinence */
    int ndim() const;
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
    operator  T() const;

    template<typename T>
    Array& operator=(const RValueExp<T>& other) {
        return (*this = other.as_assignable());
    }

    Array& operator=(const AssignableArray& assignable);

    /* Debugging */
    void print(std::basic_ostream<char>& stream = std::cout, int indent=0) const;
    void debug_memory(bool print_contents=true);

    /* Operations */
    void clear();
};

#endif
