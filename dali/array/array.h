#ifndef DALI_ARRAY_ARRAY_H
#define DALI_ARRAY_ARRAY_H

#include <memory>
#include <vector>

#include "dali/array/dtype.h"
#include "dali/array/function/operator.h"
#include "dali/array/function/expression.h"
#include "dali/array/function/operator.h"
#include "dali/array/memory/device.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/array/memory/synchronized_memory.h"
#include "dali/array/slice.h"
#include "dali/array/shape.h"
#include "dali/runtime_config.h"

class Array;

struct AssignableArray {
    typedef std::function<void(Array&, const OPERATOR_T&)> assign_t;
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
    std::vector<int> strides;
    DType dtype;
    ArrayState(const std::vector<int>& _shape,
               std::shared_ptr<memory::SynchronizedMemory> _memory,
               int _offset,
               const std::vector<int>& _strides,
               DType _dtype);
};

class ArraySlice;

class Array : public Exp<Array> {
  private:
    std::shared_ptr<ArrayState> state;
    template<typename T>
    T scalar_value() const;
    template<typename T>
    T& scalar_value();
    template<typename T>
    Array& assign_constant(const T& other);
    void broadcast_axis_internal(int axis);
  public:
    typedef uint index_t;
    Array();

    /* Various ways of constructing array */
    Array(const std::vector<int>& shape, DType dtype_=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array(std::initializer_list<int> shape, DType dtype_=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array(const std::vector<int>& shape,
          std::shared_ptr<memory::SynchronizedMemory>,
          int offset,
          const std::vector<int>& strides,
          DType dtype_=DTYPE_FLOAT);
    Array(const Array& other, bool copy_memory=false);
    Array(const AssignableArray& assignable);

    template<typename T>
    Array(const LazyExp<T>& expr) :
            Array(expr.as_assignable()) {
    }


    static Array arange(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array zeros(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array zeros_like(const Array& other);
    static Array ones(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array ones_like(const Array& other);

    // true if just creted with empty constructor or reset
    // (has no assossiated memory)
    bool is_stateless() const;
    // true if Array's contents conver entirety of underlying
    // memory (as opposed to offset memory, strided memory etc.).
    bool spans_entire_memory() const;
    bool contiguous_memory() const;

    void initialize(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    void initialize_with_bshape(const std::vector<int>& bshape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array& reset();

    /* Accesing internal state */
    const std::vector<int>& shape() const;
    std::shared_ptr<memory::SynchronizedMemory> memory() const;
    int offset() const;
    const std::vector<int>& strides() const;
    DType dtype() const;

    std::vector<int> normalized_strides() const;
    // just like regular shape by broadcased dimensions are negated.
    // for example if array has shape {2, 1, 3, 1} and dimension 1 is
    // broadcasted then it retuns {2, -1, 3, 1}.
    std::vector<int> bshape() const;

    /* memory moving logic */
    memory::Device preferred_device() const;
    void to_device(memory::Device device) const;

    /* Shape-related convinence */
    int ndim() const;
    int number_of_elements() const;
    std::vector<int> subshape() const;

    /* Creating a view into memory */
    Array operator[](int idx) const;
    ArraySlice operator[](const Slice& s) const;
    ArraySlice operator[](const Broadcast& b) const;
    Array operator()(index_t idx) const;
    Array transpose() const;
    Array transpose(const std::vector<int>& axes) const;
    Array ravel() const;
    Array reshape(const std::vector<int>& shape) const;
    /*
     * reshape_broadcasted can only be run on The
     * the broadcastable dimensions of size 1.
     *
     * Note: An exception to this rule is when the
     * array was previously 'reshape_broadcasted' to
     * the same size, maa no-op):
     * e.g. starting with {-1, 3, -1}, the following sequence
     * of functions will NOT result in an error:
     *    - reshape_broadcasted({2, 3, 1})
     *    - reshape_broadcasted({2, 3, 1})
     *    - reshape_broadcasted({2, 3, 5})
     *    - reshape_broadcasted({2, 3, 5})
     * but if we now call:
     *    - reshape_broadcasted({5, 3, 5})
     * or even:
     *    - reshape_broadcasted({1, 3, 5})
     * then we will see and error.
     */
    Array reshape_broadcasted(const std::vector<int>& new_shape) const;

    // TODO(szymon): look up what it's called in tensorflow/numpy and rename.
    Array pluck_axis(int axis, const Slice& slice) const;
    Array pluck_axis(int axis, int idx) const;
    Array squeeze(int axis) const;
    Array expand_dims(int new_axis) const;
    Array broadcast_axis(int axis) const;
    Array insert_broadcast_axis(int new_axis) const;

    AssignableArray sum() const;
    AssignableArray mean() const;

    /* Interpreting scalars as numbers */
    operator float&();
    operator double&();
    operator int&();

    operator float() const;
    operator double() const;
    operator int() const;

    template<typename T>
    Array& operator=(const LazyExp<T>& other) {
        return (*this = other.as_assignable());
    }
    template<typename T>
    Array& operator+=(const LazyExp<T>& other) {
        return (*this += other.as_assignable());
    }
    template<typename T>
    Array& operator-=(const LazyExp<T>& other) {
        return (*this -= other.as_assignable());
    }
    template<typename T>
    Array& operator/=(const LazyExp<T>& other) {
        return (*this /= other.as_assignable());
    }
    template<typename T>
    Array& operator*=(const LazyExp<T>& other) {
        return (*this *= other.as_assignable());
    }

    Array& operator=(const AssignableArray& assignable);
    Array& operator+=(const AssignableArray& assignable);
    Array& operator-=(const AssignableArray& assignable);
    Array& operator*=(const AssignableArray& assignable);
    Array& operator/=(const AssignableArray& assignable);

    /* Debugging */
    void print(std::basic_ostream<char>& stream = std::cout, int indent=0) const;
    void debug_memory(bool print_contents=true) const;

    /* Operations */
    void clear();
};


struct ArraySlice {
  private:
    enum ArraySliceAction {
        SLICE_RANGE,
        SLICE_IDX,
        BROADCAST
    };

    int consumed_dims;
    std::vector<Slice>            slice;
    std::vector<ArraySliceAction> action;
    Array input;

  public:
    ArraySlice(const Array& input_);
    ArraySlice(const ArraySlice& other);
    ArraySlice operator[](const Slice& s);
    ArraySlice operator[](const Broadcast& b);
    ArraySlice operator[](int idx);
    operator Array();
};

#endif
