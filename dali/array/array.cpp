#include "array.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <type_traits>

#include "dali/utils/cnpy.h"
#include "dali/array/debug.h"
#include "dali/array/expression/buffer_view.h"
#include "dali/array/expression/control_flow.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/computation.h"
#include "dali/utils/make_message.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/dot.h"


using std::vector;
using memory::SynchronizedMemory;


void alert_stateless_call(const bool& stateful, const char* fieldname) {
    ASSERT2(stateful, utils::make_message(fieldname, " must not be called on "
        "Array initialized with empty constructor.\n"
        "(To Dali developers: this error may have occurred because an mshadow"
        " expression that keeps an lvalue reference to its parent expression:"
        " `Expr& expr` was used. To fix this ensure the parent expression"
        " is kept by value instead)."));
}



////////////////////////////////////////////////////////////////////////////////
//                                 ARRAY                                      //
////////////////////////////////////////////////////////////////////////////////


Array::ArrayState::ArrayState(std::shared_ptr<Expression> expression):
    expression_(expression) {
}

std::shared_ptr<Expression> Array::expression() const {
    return state_->expression_;
}

void Array::set_expression(std::shared_ptr<Expression> new_expression) const {
    state_->expression_ = new_expression;
}

template<typename T>
T Array::scalar_value() const {
    static_assert(std::is_arithmetic<T>::value,
            "Scalar value only available for arithmetic types (integer or real).");
    ASSERT2(shape().size() == 0, utils::make_message("Attempting to cast array of "
        "shape ", shape(), " to a scalar, which is only allowed for a "
        "zero-dimensional array."));

    void* data = memory()->data(
        memory::Device::cpu());
    if (dtype() == DTYPE_FLOAT) {
        return *((float*)(data) + offset());
    } else if (dtype() == DTYPE_DOUBLE) {
        return *((double*)(data) + offset());
    } else if (dtype() == DTYPE_INT32) {
        return *((int*)(data) + offset());
    }
    return 0;
}


Array::Array() : state_(std::make_shared<ArrayState>(nullptr)) {
}

Array::Array(std::shared_ptr<Expression> expression) :
        state_(std::make_shared<ArrayState>(expression)) {
}

Array::Array(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) :
        Array(std::make_shared<BufferView>(shape, dtype, preferred_device)) {
}

Array::Array(std::initializer_list<int> shape_, DType dtype, memory::Device preferred_device) :
        Array(vector<int>(shape_), dtype, preferred_device) {
}

Array::Array(const std::vector<int>& shape,
             std::shared_ptr<SynchronizedMemory> memory,
             const int& offset,
             const std::vector<int>& strides,
             DType dtype) {
    set_expression(std::make_shared<BufferView>(
        memory, shape, dtype, offset, strides));
}

Array::Array(const Array& other, const bool& copy_memory) {
    if (copy_memory) {
        // TODO(jonathan, szymon):
        // surely we can do better.
        // if memory is broadcasted we do not want to copy
        // entire underlying memory!
        // TODO(szymon): bring back!
        *this = op::identity(other);
    } else {
        state_ = other.state_;
    }
}


Array Array::zeros(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    Array ret(shape, dtype, preferred_device);
    ret.memory()->lazy_clear();
    return ret;
}

Array Array::empty_like(const Array& other) {
    if (other.is_stateless()) {
        return Array();
    } else {
        return Array(BufferView::construct_with_bshape(
                other.bshape(), other.dtype(), other.preferred_device()));
    }
}

Array Array::zeros_like(const Array& other) {
    if (other.is_stateless()) {
        return Array();
    } else {
        Array ret = empty_like(other);
        ret.memory()->lazy_clear();
        return ret;
    }
}

Array Array::arange(const double& start, const double& stop, const double& step, DType dtype, memory::Device preferred_device) {
    int length = ((stop - start) + step - 1) / (step);
    ASSERT2(length > 0, utils::make_message("Array length must be non-zero (got "
        "start=", start, ", stop=", stop, ", step=", step, ")."));
    Array ret({length}, dtype, preferred_device);
    // ret = initializer::arange(start, step);
    return ret;
}

Array Array::arange(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    Array ret(shape, dtype, preferred_device);
    // ret = initializer::arange(0.0, 1.0);
    return ret;
}

Array Array::ones(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    Array ret(shape, dtype, preferred_device);
    // ret = initializer::ones();
    return ret;
}

Array Array::ones_like(const Array& other) {
    if (other.is_stateless()) {
        return Array();
    } else {
        Array ret = empty_like(other);
        // ret = initializer::ones();
        return ret;
    }
}

Array Array::adopt_buffer(void* buffer,
                          const std::vector<int>& shape,
                          DType dtype,
                          memory::Device buffer_location,
                          const std::vector<int>& strides) {
    ASSERT2(strides.size() == 0 || strides.size() == shape.size(), utils::make_message(
        "shape and strides must have the same size (unless strides is empty), got "
        "strides = ", strides, ", shape = ", shape));
    Array ret(shape, dtype, buffer_location);
    ret.memory()->adopt_buffer(buffer_location, buffer);
    ret.expression()->strides_ = strides;
    return ret;
}

void Array::disown_buffer(memory::Device buffer_location) {
    if (!is_stateless()) {
        // TODO(szymon): ensure expression is not evaluated.
        memory()->disown_buffer(buffer_location);
    }
}

Array Array::buffer_arg() const {
    if (is_buffer()) {
        return *this;
    }
    if (is_assignment()) {
        return std::dynamic_pointer_cast<Assignment>(expression())->left_;
    }
    if (is_control_flow()) {
        return std::dynamic_pointer_cast<ControlFlow>(expression())->left_;
    }
    // returning false
    // TODO(jonathan): make this return something better
    //                 than a stateless array
    return Array();
}


Array autoreduce_assign(Array left, Array right) {
    throw std::runtime_error("autoreduce_assign not implemented yet.");
}

Array assign(Array left, OPERATOR_T operator_t, Array right);

Array to_assignment(Array node) {
    return assign(Array::zeros(node.shape(), node.dtype()),
                  OPERATOR_T_EQL,
                  Array(node.expression()));
}

Array assign(Array left, OPERATOR_T operator_t, Array right) {
    if (operator_t == OPERATOR_T_EQL) {
        return Array(std::make_shared<Assignment>(left, operator_t, right));
    } else if (operator_t == OPERATOR_T_LSE) {
        return autoreduce_assign(left, right);
    } else {
        // a temp is added so that non overwriting operators
        // can be run independently from the right side's evaluation.
        return Array(std::make_shared<Assignment>(left, operator_t, to_assignment(right)));
    }
}

std::vector<Array> right_args(Array node) {
    auto buffer = std::dynamic_pointer_cast<Assignment>(node.expression());
    return buffer->right_.expression()->arguments();
}


// TODO(jonathan): add this from Python
Array all_assignments_or_buffers(Array node) {
    if (node.is_buffer()) {
        return node;
    }
    if (!node.is_assignment()) {
        node = to_assignment(node);
    }
    for (auto& arg : right_args(node)) {
        arg.set_expression(all_assignments_or_buffers(arg).expression());
    }
    return node;
}

struct Optimization {
    std::function<bool(const Array&)> condition_;
    std::function<Array(const Array&)> transformation_;
    bool matches(const Array& array) const {
        return condition_(array);
    }
    Array transform(const Array& array) const {
        return transformation_(array);
    }
    Optimization(std::function<bool(const Array&)> condition,
                 std::function<Array(const Array&)> transformation) :
        condition_(condition), transformation_(transformation) {}
};

std::vector<Optimization> OPTIMIZATIONS;

Array simplify_destination(Array root) {
    // leaf node:
    if (root.is_buffer()) {
        return root;
    }
    // recurse on children:
    std::vector<Array> children;
    if (root.is_assignment()) {
        children.emplace_back(std::dynamic_pointer_cast<Assignment>(root.expression())->right_);
    } else {
        children = root.expression()->arguments();
    }

    // recurse on arguments of node:
    for (auto& arg : children) {
        arg.set_expression(simplify_destination(arg).expression());
    }
    for (const auto& optimization : OPTIMIZATIONS) {
        if (optimization.matches(root)) {
            root = optimization.transform(root);
        }
    }
    return root;
}

Array Array::canonical() const {
    // assignment pass
    auto node = all_assignments_or_buffers(*this);
    // simplification pass (jit, merge, etc...)
    return simplify_destination(node);
}

void Array::eval(bool wait) const {
    if (!is_buffer()) {
        auto node = canonical();
        auto computable = convert_to_ops(node);
        // run (DAG evaluation)
        for (auto& step : computable) {
            step->run();
        }
        // update internal expression to reflect
        // that op was evaluated...
        set_expression(node.buffer_arg().expression());
    }
}

/* NPY detect Dtype
 * ================
 * Use the numpy dtype chars (i -> integer type, f -> float type)
 * and the size of a word (e.g. sizeof(float) = 4), determine the
 * right Dali dtype.
 */
DType npy_detect_dtype(const char& dtype, const int& word_size) {
    if (dtype == 'f') {
        if (word_size == sizeof(double)) {
            return DTYPE_DOUBLE;
        } else if (word_size == sizeof(float)) {
            return DTYPE_FLOAT;
        } else {
            ASSERT2(word_size == sizeof(double) || word_size == sizeof(float),
                utils::make_message("attempted to load a npy array of floats with "
                "dtype different from float or doubles (word size = ", word_size, ")."));
        }
    } else if (dtype == 'i') {
        if (word_size == sizeof(int32_t)) {
            return DTYPE_INT32;
        } else {
            ASSERT2(word_size == sizeof(double) || word_size == sizeof(float),
                utils::make_message("can only load numpy arrays with dtype float "
                "or double (got word_size = ", word_size, ")."));
        }
    } else {
        ASSERT2(dtype == 'i' || dtype == 'f', utils::make_message("attempted to "
            "load a npy array with dtype different from float or int (got dtype "
            "= ", dtype, ")."));
    }
    return DTYPE_FLOAT;
}

Array load_npy_from_npyarray(const cnpy::NpyArray& arr) {
    auto dtype = npy_detect_dtype(arr.dtype, arr.word_size);
    std::vector<int> shape(arr.shape.size());
    for (int i = 0; i < arr.shape.size(); i++) {
        shape[i] = arr.shape[i];
    }

    Array loaded;
    if (arr.fortran_order) {
        // in fortran the strides are reversed
        // with respect to c-style memory layout
        // hence we can obtain a similar memory
        // layout by transposing an array
        // that has the dimensions of fortran array
        // reversed:
        // e.g. to load Fortran(2, 3, 4) we:
        // 1) create x = Array(4, 3, 2)
        // 2) transpose x_T = x.transpose();
        // 3) load memory into x_T using buffer
        // => x_T is now a fortran-ordered view
        // onto the memory in Fortran(2, 3, 4)
        std::reverse(shape.begin(), shape.end());
        loaded = Array(shape, dtype);
        loaded = loaded.transpose();
    } else {
        // c-style memory layout
        loaded = Array(shape, dtype);
    }
    loaded.memory()->adopt_buffer(memory::Device::cpu(), arr.data);
    return loaded;
}

Array Array::load(FILE * fp) {
    auto arr = cnpy::load_the_npy_file(fp);
    return load_npy_from_npyarray(arr);
}

Array Array::load(const std::string& fname) {
    auto arr = cnpy::npy_load(fname);
    return load_npy_from_npyarray(arr);
}

void Array::save(const std::string& fname, const Array& arr, const std::ios_base::openmode& mode) {
    std::ofstream outfile(fname, std::ofstream::binary | mode);
    Array::save(outfile, arr);
    outfile.close();
}

void Array::save(std::basic_ostream<char>& stream, const Array& arr) {
    auto contig_array = arr.ascontiguousarray();
    const auto& dimensions = contig_array.shape();
    const void* data = contig_array.memory()->readonly_data(memory::Device::cpu());
    std::vector<unsigned int> dimensions_unsigned(dimensions.size());
    for (int i = 0; i < dimensions.size(); i++) {
        dimensions_unsigned[i] = dimensions[i];
    }
    std::vector<char> header;
    switch(arr.dtype()) {
        case DTYPE_FLOAT:
            header = cnpy::create_npy_header((float*)data,
                                             dimensions_unsigned.data(),
                                             dimensions_unsigned.size());
            break;
        case DTYPE_DOUBLE:
            header = cnpy::create_npy_header((double*)data,
                                             dimensions_unsigned.data(),
                                             dimensions_unsigned.size());
            break;
        case DTYPE_INT32:
            header = cnpy::create_npy_header((int*)data,
                                             dimensions_unsigned.data(),
                                             dimensions_unsigned.size());
            break;
        default:
            ASSERT2(false, "save called on an Array with incorrect DType.");
            break;
    }
    stream.write(header.data(), header.size());
    stream.write((char*)data, contig_array.memory()->total_memory);
}

bool Array::equals(const Array& left, const Array& right) {
    if (state_equals(left, right)) {
        return true;
    }
    if (left.shape() != right.shape()) {
        return false;
    }
    if (left.is_stateless() != right.is_stateless()) {
        return false;
    }
    throw std::runtime_error("not implemented yet");
    // bool all_equals = ((float)(Array)op::all_equals(left, right)) > 0 ? true : false;
    // return all_equals;
}


bool Array::state_equals(const Array& left, const Array& right) {
    if (left.is_stateless() && right.is_stateless())
        return true;
    if (left.is_stateless() != right.is_stateless()) {
        return false;
    }
    return left.state_ == right.state_;
}

bool Array::allclose(const Array& left, const Array& right, const double& atolerance) {
    if (left.is_stateless() && right.is_stateless()) {
        return true;
    }
    if (left.is_stateless() != right.is_stateless()) {
        return false;
    }
    if (left.shape() != right.shape()) {
        return false;
    }
    // bool is_all_close = ((float)(Array)op::all_close(left, right, atolerance)) > 0 ? true : false;
    // return is_all_close;
}

bool Array::any_isnan() const {
    // bool any_isnan_ = ((float)(Array)op::any_isnan(*this)) > 0 ? true : false;
    // return any_isnan_;
}

bool Array::any_isinf() const {
    // bool any_isinf_ = ((float)(Array)op::any_isinf(*this)) > 0 ? true : false;
    // return any_isinf_;
}

bool Array::is_stateless() const {
    return state_ == nullptr;
}

bool Array::is_scalar() const {
    return expression()->is_scalar();
}

bool Array::is_vector() const {
    return expression()->is_vector();
}

bool Array::is_matrix() const {
    return expression()->is_matrix();
}


Array Array::ascontiguousarray() const {
    Array ret;
    if (contiguous_memory()) {
        ret = *this;
    } else {
        debug::array_as_contiguous.notify(*this);
        // ret = op::identity(*this);
    }
    ret.expression()->strides_.clear();
    return ret;
}

Array& Array::reset() {
    set_expression(nullptr);
    return *this;
}

const vector<int>& Array::shape() const {
    alert_stateless_call(!is_stateless(), "shape");
    return expression()->shape_;
}

bool Array::is_buffer() const {
    auto buffer = std::dynamic_pointer_cast<BufferView>(expression());
    return buffer != nullptr;
}

bool Array::is_control_flow() const {
    auto buffer = std::dynamic_pointer_cast<ControlFlow>(expression());
    return buffer != nullptr;
}

bool Array::is_assignment() const {
    auto buffer = std::dynamic_pointer_cast<Assignment>(expression());
    return buffer != nullptr;
}

std::shared_ptr<memory::SynchronizedMemory> Array::memory() const {
    if (is_stateless()) {
        return nullptr;
    } else {
        eval(/*wait=*/true);
        auto buffer = std::dynamic_pointer_cast<BufferView>(expression());
        ASSERT2(buffer, "eval failed to create BufferView.");
        return buffer->memory_;
    }
}

int Array::offset() const {
    alert_stateless_call(!is_stateless(), "offset");
    return expression()->offset_;
}

const std::vector<int>& Array::strides() const {
    alert_stateless_call(!is_stateless(), "strides");
    return expression()->strides_;
}

std::vector<int> Array::normalized_strides() const {
    alert_stateless_call(!is_stateless(), "normalized_strides");
    return expression()->normalized_strides();
}

DType Array::dtype() const {
    alert_stateless_call(!is_stateless(), "dtype");
    return expression()->dtype_;
}

Array Array::astype(DType dtype_) const {
    throw std::runtime_error("not yet implemented.");
    // return op::astype(*this, dtype_);
}

memory::Device Array::preferred_device() const {
    std::cout << "Array::preferred_device" << std::endl;
    alert_stateless_call(!is_stateless(), "preferred_device");
    return expression()->preferred_device();
}

void Array::to_device(memory::Device device) const {
    // TODO(jonathan and only jonathan):
    //     "We will not have that!"
    //                   -- Jurgen Schmidthuber
    //
    // Add asynchronous memory movement.
    memory()->move_to(device);
    memory()->preferred_device = device;
}

int Array::ndim() const {
    return (is_stateless()) ? 0 : expression()->ndim();

}

std::vector<int> Array::bshape() const {
    alert_stateless_call(!is_stateless(), "bshape");
    return expression()->bshape();

}

int Array::number_of_elements() const {
    return (is_stateless()) ? 0 : hypercube_volume(expression()->shape_);
}

vector<int> Array::subshape() const {
    if (is_stateless()) return vector<int>();
    if (expression()->shape_.size() == 0) return vector<int>();
    return vector<int>(expression()->shape_.begin() + 1, expression()->shape_.end());
}

bool Array::contiguous_memory() const {
    return expression()->contiguous_memory();
}

Array Array::operator[](const int& idx) const {
    return pluck_axis(0, idx);
}

Array Array::gather_from_rows(const Array& indices) const {
    // return ArraySubtensor(*this, indices);
}

Array Array::operator[](const Array& indices) const {
    // return ArrayGather(*this, indices);
}

SlicingInProgress<Array> Array::operator[](const Slice& s) const {
    auto ret = SlicingInProgress<Array>(*this);
    return ret[s];
}

SlicingInProgress<Array> Array::operator[](const Broadcast& b) const {
    auto ret = SlicingInProgress<Array>(*this);
    return ret[b];
}

Array Array::operator()(int idx) const {
    alert_stateless_call(!is_stateless(), "operator()");
    return Array((*expression())(idx));
}

bool Array::is_transpose() const {
    alert_stateless_call(!is_stateless(), "is_transpose");
    return expression()->is_transpose();
}

Array Array::transpose() const {
    alert_stateless_call(!is_stateless(), "transpose");
    return Array(expression()->transpose());
}

Array Array::transpose(const std::vector<int>& axes) const {
    alert_stateless_call(!is_stateless(), "transpose");
    return Array(expression()->transpose(axes));
}

Array Array::swapaxes(int axis1, int axis2) const {
    alert_stateless_call(!is_stateless(), "swapaxes");
    return Array(expression()->swapaxes(axis1, axis2));
}

Array Array::dimshuffle(const std::vector<int>& pattern) const {
    alert_stateless_call(!is_stateless(), "dimshuffle");
    return Array(expression()->dimshuffle(pattern));
}

Array Array::ravel() const {
    alert_stateless_call(!is_stateless(), "ravel");
    return Array(expression()->ravel());
}

Array Array::copyless_ravel() const {
    alert_stateless_call(!is_stateless(), "copyless_ravel");
    return Array(expression()->copyless_ravel());
}

Array Array::reshape(const std::vector<int>& shape) const {
    alert_stateless_call(!is_stateless(), "reshape");
    return Array(expression()->reshape(shape));
}

Array Array::collapse_axis_with_axis_minus_one(int axis) const {
    alert_stateless_call(!is_stateless(), "collapse_axis_with_axis_minus_one");
    return Array(expression()->collapse_axis_with_axis_minus_one(axis));
}

Array Array::copyless_reshape(const std::vector<int>& shape) const {
    alert_stateless_call(!is_stateless(), "copyless_reshape");
    return Array(expression()->copyless_reshape(shape));
}


Array Array::right_fit_ndim(int dimensionality) const {
    alert_stateless_call(!is_stateless(), "right_fit_ndim");
    return Array(expression()->right_fit_ndim(dimensionality));
}

Array Array::copyless_right_fit_ndim(int dimensionality) const {
    alert_stateless_call(!is_stateless(), "copyless_right_fit_ndim");
    return Array(expression()->copyless_right_fit_ndim(dimensionality));
}

Array Array::reshape_broadcasted(const std::vector<int>& new_shape) const {
    alert_stateless_call(!is_stateless(), "reshape_broadcasted");
    return Array(expression()->reshape_broadcasted(new_shape));
}

Array Array::pluck_axis(int axis, const Slice& slice) const {

    alert_stateless_call(!is_stateless(), "pluck_axis");
    return Array(expression()->pluck_axis(axis, slice));
}

Array Array::pluck_axis(const int& axis, const int& idx) const {
    alert_stateless_call(!is_stateless(), "pluck_axis");
    return Array(expression()->pluck_axis(axis, idx));
}

Array Array::squeeze(int axis) const {
    alert_stateless_call(!is_stateless(), "squeeze");
    return Array(expression()->squeeze(axis));
}

Array Array::expand_dims(int new_axis) const {
    alert_stateless_call(!is_stateless(), "expand_dims");
    return Array(expression()->expand_dims(new_axis));
}

Array Array::broadcast_axis(int axis) const {
    alert_stateless_call(!is_stateless(), "broadcast_axis");
    return Array(expression()->broadcast_axis(axis));
}

Array Array::insert_broadcast_axis(int new_axis) const {
    alert_stateless_call(!is_stateless(), "insert_broadcast_axis");
    return Array(expression()->insert_broadcast_axis(new_axis));
}

Array Array::broadcast_scalar_to_ndim(const int& ndim) const {
    alert_stateless_call(!is_stateless(), "broadcast_scalar_to_ndim");
    return Array(expression()->broadcast_scalar_to_ndim(ndim));
}



#define DALI_ARRAY_DEFINE_ALL_REDUCER(FUNCTION_NAME, OPNAME)\
    Array Array::FUNCTION_NAME() const {\
        /*return op::OPNAME(*this);*/\
    }\

#define DALI_ARRAY_DEFINE_AXIS_REDUCER(FUNCTION_NAME, OPNAME)\
    Array Array::FUNCTION_NAME(const int& axis) const {\
        /*return op::OPNAME(*this, {axis});*/\
    }\

#define DALI_ARRAY_DEFINE_REDUCER(FUNCTION_NAME, OPNAME)\
    DALI_ARRAY_DEFINE_ALL_REDUCER(FUNCTION_NAME, OPNAME);\
    DALI_ARRAY_DEFINE_AXIS_REDUCER(FUNCTION_NAME, OPNAME);\

DALI_ARRAY_DEFINE_REDUCER(sum, sum);
DALI_ARRAY_DEFINE_REDUCER(L2_norm, L2_norm);
DALI_ARRAY_DEFINE_REDUCER(mean, mean);
DALI_ARRAY_DEFINE_REDUCER(max, max);
DALI_ARRAY_DEFINE_REDUCER(min, min);
DALI_ARRAY_DEFINE_REDUCER(argsort, argsort);
DALI_ARRAY_DEFINE_REDUCER(argmin, argmin);
DALI_ARRAY_DEFINE_REDUCER(argmax, argmax);

Array::operator float() const {
    return scalar_value<float>();
}
Array::operator double() const {
    return scalar_value<double>();
}
Array::operator int() const {
    return scalar_value<int>();
}

void Array::copy_from(const Array& other) {
    *this = op::identity(other);
}

Array& Array::operator=(const int& other) {
    return *this = op::identity(other);
}

Array& Array::operator=(const float& other) {
    return *this = op::identity(other);
}

Array& Array::operator=(const double& other) {
    return *this = op::identity(other);
}

void Array::print(std::basic_ostream<char>& stream, const int& indent, const bool& add_newlines, const bool& print_comma) const {
    std::string end_line_spacing("");
    if (add_newlines) end_line_spacing += "\n";
    int indent_increment = add_newlines ? 4 : 0;
    if (ndim() == 0) {
        if (dtype() == DTYPE_FLOAT) {
            stream << (float)(*this);
        } else if (dtype() == DTYPE_DOUBLE) {
            stream << (double)(*this);
        } else if (dtype() == DTYPE_INT32) {
            stream << (int)(*this);
        } else {
            ASSERT2(false, "Wrong dtype for Array.");
        }
        stream << end_line_spacing;
    } else if (ndim() == 1) {
        stream << std::string(indent, ' ');
        stream << "[";
        for(int i = 0; i < expression()->shape_[0]; i += 1) {
            stream << std::fixed
                      << std::setw( 7 ) /* keep 7 digits*/
                      << std::setprecision( 3 ) /* use 3 decimals*/
                      << std::setfill( ' ' );
            Array scalar = (*this)[i];
            scalar.print(stream, 0, false);
            if (i != expression()->shape_[0] - 1) stream << ", ";
        }
        stream << "]";
        if (print_comma) stream << ",";
        stream << end_line_spacing;
    } else {
        stream << std::string(indent, ' ') << "[" << end_line_spacing;
        for (int i = 0; i < expression()->shape_[0]; ++i) {
            Array subtensor = (*this)[i];
            subtensor.print(
                stream,
                indent + indent_increment,
                add_newlines,
                /*print_comma=*/i != expression()->shape_[0] - 1
            );
        }
        stream << std::string(indent, ' ') << "]";
        if (print_comma) stream << ",";
        stream << end_line_spacing;
    }
}

void Array::debug_memory(const bool& print_contents) const {
    memory()->debug_info(std::cout, print_contents, dtype());
}

void Array::clear() {
    auto buffer = std::dynamic_pointer_cast<BufferView>(expression());
    if (buffer) {
        if (buffer->spans_entire_memory()) {
            memory()->lazy_clear();
        } else {
            ASSERT2(false, "not implemented");
            // TODO(szymon): fill with zeros
        }
    } else {
        set_expression(zeros_like(*this).expression());
    }
}

Array Array::dot(const Array& other) const {
    return op::dot(*this, other);
}

bool operator==(const Array& left, const Array& right) {
    return Array::state_equals(left, right);
}


DType type_promotion(const Array& a, const Array& b) {
    // TODO(jonathan,szymon) speed up this function
    bool a_scalar = a.is_scalar();
    bool b_scalar = b.is_scalar();

    if ((a_scalar ^ b_scalar) == 0) {
        // if they are both scalars or both arrays
        if (a.dtype() == DTYPE_DOUBLE || b.dtype() == DTYPE_DOUBLE) {
            return DTYPE_DOUBLE;
        } else if (a.dtype() == DTYPE_FLOAT || b.dtype() == DTYPE_FLOAT) {
            return DTYPE_FLOAT;
        } else {
            return DTYPE_INT32;
        }
    } else if (a_scalar) {
        // if a is scalar and b is array.
        return b.dtype();
    } else {
        // if a is array and b is scalar.
        return a.dtype();
    }
}
