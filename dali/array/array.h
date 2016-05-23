#ifndef DALI_ARRAY_ARRAY_H
#define DALI_ARRAY_ARRAY_H

#include <memory>
#include <vector>

#include "dali/array/dtype.h"
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
    template<typename ExprT>
    AssignableArray(const LazyExp<ExprT>& expr);
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

class Array : public Exp<Array> {
  private:
    std::shared_ptr<ArrayState> state;
    template<typename T>
    T scalar_value() const;
    void broadcast_axis_internal(int axis);
    int normalize_axis(int axis) const;
  public:
    typedef uint            index_t;

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
    template<typename ExprT>
    Array(const LazyExp<ExprT>& expr);

    static Array arange(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array zeros(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array zeros_like(const Array& other);
    static Array ones(const std::vector<int>& shape, DType dtype=DTYPE_FLOAT, memory::Device preferred_device=memory::default_preferred_device);
    static Array ones_like(const Array& other);

    // IO methods
    static Array load(const std::string& fname);
    static Array load(FILE * fp);
    static void save(const std::string& fname, const Array& arr, const std::ios_base::openmode& mode=std::ios_base::out);
    static void save(std::basic_ostream<char>& stream, const Array& arr);
    static bool equals(const Array& left, const Array& right);

    // true if just creted with empty constructor or reset
    // (has no assossiated memory)
    bool is_nan() const;
    bool is_stateless() const;
    bool is_scalar() const;
    bool is_vector() const;
    bool is_matrix() const;
    Array vectorlike_to_vector() const;
    // true if Array's contents conver entirety of underlying
    // memory (as opposed to offset memory, strided memory etc.).
    bool spans_entire_memory() const;
    bool contiguous_memory() const;
    Array ascontiguousarray() const;

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
    SlicingInProgress<Array> operator[](const Slice& s) const;
    SlicingInProgress<Array> operator[](const Broadcast& b) const;
    Array operator()(index_t idx) const;
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
    Array pluck_axis(int axis, int idx) const;
    Array squeeze(int axis) const;
    Array expand_dims(int new_axis) const;
    Array broadcast_axis(int axis) const;
    Array insert_broadcast_axis(int new_axis) const;
    Array broadcast_scalar_to_ndim(int ndim) const;

    // reduce over all axes
    AssignableArray sum() const;
    AssignableArray mean() const;
    AssignableArray min() const;
    AssignableArray max() const;
    AssignableArray L2_norm() const;

    // reduce over one axis
    AssignableArray sum(const int& axis) const;
    AssignableArray mean(const int& axis) const;
    AssignableArray min(const int& axis) const;
    AssignableArray max(const int& axis) const;
    AssignableArray L2_norm(const int& axis) const;

    operator float() const;
    operator double() const;
    operator int() const;

    Array& operator=(const AssignableArray& assignable);

    #define DALI_DECLARE_ARRAY_INTERACTION_INPLACE(SYMBOL)\
        Array& operator SYMBOL (const AssignableArray& right);\
        Array& operator SYMBOL (const Array& right);\

    #define DALI_DECLARE_SCALAR_INTERACTION_INPLACE(SYMBOL)\
        Array& operator SYMBOL (const double& right);\
        Array& operator SYMBOL (const float& right);\
        Array& operator SYMBOL (const int& right);\

    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(+=);
    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(-=);
    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(*=);
    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(/=);
    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(<<=);

    DALI_DECLARE_SCALAR_INTERACTION_INPLACE(-=);
    DALI_DECLARE_SCALAR_INTERACTION_INPLACE(+=);
    DALI_DECLARE_SCALAR_INTERACTION_INPLACE(*=);
    DALI_DECLARE_SCALAR_INTERACTION_INPLACE(/=);

    /* Debugging */
    void print(std::basic_ostream<char>& stream = std::cout, int indent=0, bool add_newlines=true) const;
    void debug_memory(bool print_contents=true) const;

    /* Operations */
    void clear();

    template<typename ExprT>
    Array& operator=(const LazyExp<ExprT>& expr);

    /* shortcuts for array ops */
    AssignableArray dot(const Array& other) const;
};

#endif

#ifndef DALI_ARRAY_HIDE_LAZY
    // Terminology
    //    * `lazy people` - all the files that you can find by searching for
    //                      phrase DALI_ARRAY_HIDE_LAZY (but excluding array.h)
    //                      those files are key for lazy expressions.
    //    * `Bjarne`      - c++ language design philosophy.
    //    * `may not be`  - definitely isn't
    //    * `Header Extension` -
    //                      the aberration that happens below this comment
    //                      the author hopes that by giving it this friendly
    //                      name the risk of going insane will be minimized.
    //
    // Currently bunch of lazy people depend on array. Even though they never
    // use any of the functions specified below, array.h is included and used
    // for data/strides/shape etc. access. The problem appears when g++ sees
    // operator= in array.h it for some reason tries to expand what's inside
    // the function (even though nobody uses it, but apparently we have
    // inevitably hit template expansion phase of compilation or something like
    // that where compiler gets really excited about contents of templated
    // functions).
    //
    // CURRENT SOLUTION:
    // This may not be the best solution to the issue we are facing.
    // The current solution to this problem is to specify DALI_ARRAY_HIDE_LAZY
    // which hides those function definitions from all the files related to lazy
    // functions. The next time somebody includes array.h outside of those
    // files, this templated functions will become available.
    // The primary disadvantage of this solution is the fact that we are using
    // a new paradigm of optional Header Extension, which may require extra
    // cognitive effort.
    //
    // ALTERNATIVE SOLUTION I:
    // Split Array class into BaseArray used by all the lazy functionality
    // related files and Array which adds the lazy conversions, then all the
    // lazy people must use BaseArray. The problems with this solution include:
    //     * giving in to Bjarne
    //     * the fact that sometimes we need to convert between BaseArray and
    //       Array.
    //     * we might need BaseAssignableArray (?!??)
    // Honestly, this might be a better solution, but the only way to understand
    // how much worse/better it is, is to implement it and try to gradually
    // remove all the associated inconveniences.
    //
    // ALTERNATIVE SOLUTION II:
    // add a bunch of lazy::eval in places where it is inferred automatically
    // thanks to the function below. At the time of writing this is order of
    // tens of occurrences.

    #ifndef DALI_ARRAY_ARRAY_H_EXTENSION
    #define DALI_ARRAY_ARRAY_H_EXTENSION

    #include "dali/array/function/lazy_eval.h"

    template<typename ExprT>
    Array& Array::operator=(const LazyExp<ExprT>& expr) {
        return *this = lazy::eval(expr.self());
    }

    /* Array constructor from a Lazy Expression:
     * here we evaluate the expression, construct an assignable array
     * from it, and pass this assignable array to create a new Array
     * as a destination for the computation in the expression.
     * Since the destination has not been created yet, we evaluate
     * the expression using `lazy::eval_no_autoreduce` instead of
     * `lazy::eval`, since we know that no reductions will be needed
     * when assigning to the same shape as the one dictated by the
     * expression.
     */
    template<typename ExprT>
    Array::Array(const LazyExp<ExprT>& expr) :
            Array(lazy::eval_no_autoreduce(expr.self())) {
    }

    template<typename ExprT>
    AssignableArray::AssignableArray(const LazyExp<ExprT>& expr) :
            AssignableArray(lazy::eval(expr.self())) {
    }

    #endif
#endif
