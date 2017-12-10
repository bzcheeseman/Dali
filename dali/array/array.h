#ifndef DALI_ARRAY_ARRAY_H
#define DALI_ARRAY_ARRAY_H

#include <memory>
#include <vector>

#include "dali/array/dtype.h"
#include "dali/array/expression/operator.h"
#include "dali/array/memory/device.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/array/memory/synchronized_memory.h"
#include "dali/array/expression/expression.h"
#include "dali/array/shape.h"
#include "dali/array/slice.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"

class Array  {
  private:
    struct ArrayState {
        mutable std::shared_ptr<Expression> expression_;
        ArrayState(std::shared_ptr<Expression> expression);
        // std::mutex mutex;
    };
    std::shared_ptr<ArrayState> state_;

    template<typename T>
    T scalar_value() const;


  public:
    std::shared_ptr<Expression> expression() const;
    void set_expression(std::shared_ptr<Expression>) const;
    Array();
    std::string expression_name() const;
    std::string full_expression_name() const;

    /* Various ways of constructing array */
    Array(std::shared_ptr<Expression>);
    Array(const std::vector<int>& shape, DType dtype_=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array(std::initializer_list<int> shape, DType dtype_=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array(const std::vector<int>& shape,
          std::shared_ptr<memory::SynchronizedMemory>,
          const int& offset,
          const std::vector<int>& strides,
          DType dtype_=DTYPE_FLOAT);
    Array(const Array& other, const bool& copy_memory=false);

    static Array arange(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array arange(const double& start, const double& stop, const double& step, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array zeros(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array zeros_like(const Array& other);
    static Array empty_like(const Array& other);
    static Array ones(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array ones_like(const Array& other);

    // Warning: after this function call Array takes ownership of the buffer.
    static Array adopt_buffer(void* buffer,
                              const std::vector<int>& shape,
                              DType dtype,
                              memory::Device buffer_location=memory::Device::cpu(),
                              const std::vector<int>& strides = {});
    // Warning: after this function call Array owns no buffer
    void disown_buffer(memory::Device buffer_location=memory::Device::cpu());

    void eval(bool wait=true) const;


    // IO methods
    static Array load(const std::string& fname);
    static Array load(FILE * fp);
    static void save(const std::string& fname, const Array& arr, const std::ios_base::openmode& mode=std::ios_base::out);
    static void save(std::basic_ostream<char>& stream, const Array& arr);
    static bool equals(const Array& left, const Array& right);
    static bool state_equals(const Array& left, const Array& right);
    // compare two arrays and considered close if abs(left - right) <= atolerance
    static bool allclose(const Array& left, const Array& right, const double& atolerance);

    // true if just created with empty constructor or reset
    // (has no assossiated memory)
    bool any_isnan() const;
    bool any_isinf() const;
    bool is_stateless() const;
    bool is_scalar() const;
    bool is_vector() const;
    bool is_matrix() const;
    // true if Array's contents cover entirety of underlying
    // memory (as opposed to offset memory, strided memory etc.).

    Array ascontiguousarray() const;

    void initialize(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    void initialize_with_bshape(const std::vector<int>& bshape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    Array& reset();

    /* Accesing internal state */
    const std::vector<int>& shape() const;
    std::shared_ptr<memory::SynchronizedMemory> memory() const;
    int offset() const;
    const std::vector<int>& strides() const;
    std::vector<int> normalized_strides() const;
    DType dtype() const;

    Array astype(DType dtype_) const;

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
    bool contiguous_memory() const;


    /* Creating a view into memory */
    Array operator[](const int& idx) const;
    Array operator[](const Array& indices) const;
    SlicingInProgress<Array> operator[](const Slice& s) const;
    SlicingInProgress<Array> operator[](const Broadcast& b) const;
    Array gather_from_rows(const Array& indices) const;
    // Get scalar at this offset:
    Array operator()(int idx) const;
    // returns true if array is possibly a result of calling .transpose()
    // on another array.
    bool is_transpose() const;
    // special checks for internal expression type:
    bool is_buffer() const;
    bool is_assignment() const;
    bool is_control_flow() const;
    // returns a BufferView Array if the node is a buffer view,
    // a control flow op, or an assignment, else returns a stateless Array
    // (e.g. falsy)
    Array buffer_arg() const;
    // create a view of the transposed memory
    Array transpose() const;
    Array transpose(const std::vector<int>& axes) const;
    Array swapaxes(int axis1, int axis2) const;
    // a less flexible version of the dimension switching
    // TODO(jonathan): add swapaxes + should allow insertion of
    // broadcasts in dimshuffle (aka [1, 'x', 0], where 'x' is broadcasted)
    Array dimshuffle(const std::vector<int>& pattern) const;
    Array ravel() const;
    // only ravel if underlying memory is contiguous
    // ensures that no unexpected memory aliasing occurs
    Array copyless_ravel() const;
    Array reshape(const std::vector<int>& shape) const;
    // only reshapes if underlying memory is contiguous
    // ensures that no unexpected memory aliasing occurs
    Array copyless_reshape(const std::vector<int>& shape) const;

    Array collapse_axis_with_axis_minus_one(int axis) const;

    Array right_fit_ndim(int dimensionality) const;
    Array copyless_right_fit_ndim(int dimensionality) const;
    /*
     * reshape_broadcasted can only be run on The
     * the broadcastable dimensions of size 1.
     *
     * Note: An exception to this rule is when the
     * array was previously 'reshape_broadcasted' to
     * the same size, maa no-op):
     * e.g. starting with {-1, 3, -1}, the following sequence
     * of functions will NOToperator<<= result in an error:
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
    Array pluck_axis(const int& axis, const int& idx) const;
    Array squeeze(int axis) const;
    Array expand_dims(int new_axis) const;
    Array broadcast_axis(int axis) const;
    Array insert_broadcast_axis(int new_axis) const;
    Array broadcast_scalar_to_ndim(const int& ndim) const;

    // reduce over all axes
    Array sum() const;
    Array mean() const;
    Array min() const;
    Array max() const;
    Array L2_norm() const;

    // reduce over one axis
    Array sum(const int& axis) const;
    Array mean(const int& axis) const;
    Array min(const int& axis) const;
    Array max(const int& axis) const;
    Array L2_norm(const int& axis) const;

    Array argmin(const int& axis) const;
    Array argmin() const;
    Array argmax(const int& axis) const;
    Array argmax() const;

    Array argsort(const int& axis) const;
    Array argsort() const;

    operator float() const;
    operator double() const;
    operator int() const;

    void copy_from(const Array& other);
    Array& operator=(const int& other);
    Array& operator=(const float& other);
    Array& operator=(const double& other);

    template<typename T>
    Array& operator=(const std::vector<T>& values) {
        ASSERT2(values.size() == shape()[0],
                utils::MS() << "mismatch when assigning to Array from vector, expected dim size "
                            << shape()[0] << " got " << values.size() << ".");
        for (int i = 0; i < values.size(); ++i) {
            Array subarray = (*this)[i];
            subarray = values[i];
        }
        return *this;
    }

    /* Debugging */
    void print(std::basic_ostream<char>& stream = std::cout, const int& indent=0, const bool& add_newlines=true, const bool& print_comma=false) const;
    void debug_memory(const bool& print_contents=true) const;

    /* Expressions */
    void clear();

    /* shortcuts for array ops */
    Array dot(const Array& other) const;
};
bool operator==(const Array& left, const Array& right);

DType type_promotion(const Array& a, const Array& b);

#endif
