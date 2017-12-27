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
#include "dali/array/expression/optimization.h"
#include "dali/utils/make_message.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/op/dot.h"
#include "dali/array/op/reducers.h"
#include "dali/array/jit/scalar_view.h"


void alert_stateless_call(const bool& stateful, const char* fieldname) {
    ASSERT2(stateful, utils::make_message(fieldname, " must not be called on "
        "Array initialized with empty constructor."));
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

std::shared_ptr<Array::ArrayState> Array::state() const {
    return state_;
}

void Array::set_expression(std::shared_ptr<Expression> new_expression) const {
    if (state_ == nullptr) {
        state_ = std::make_shared<ArrayState>(new_expression);
    } else {
        state_->expression_ = new_expression;
    }
}

void Array::set_state(std::shared_ptr<ArrayState> new_state) const {
    state_ = new_state;
}

std::string Array::expression_name() const {
    if (is_stateless()) {
        return "Stateless Array";
    }
    auto expr = expression();
    if (expr == nullptr) {
        return "Expressionless Array";
    }
    return expr->name();
}

std::string Array::full_expression_name() const {
    if (is_stateless()) {
        return expression_name();
    }
    return expression()->full_name();
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


Array::Array() : state_(nullptr) {
}

Array::Array(std::shared_ptr<Expression> new_expression) :
        state_(std::make_shared<ArrayState>(new_expression)) {
}

Array::Array(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) :
        Array(std::make_shared<BufferView>(shape, dtype, preferred_device)) {
}

Array::Array(std::initializer_list<int> shape_, DType dtype, memory::Device preferred_device) :
        Array(std::vector<int>(shape_), dtype, preferred_device) {
}

Array::Array(const std::vector<int>& shape,
             std::shared_ptr<memory::SynchronizedMemory> memory,
             const int& offset,
             const std::vector<int>& strides,
             DType dtype) {
    set_expression(std::make_shared<BufferView>(
        memory, shape, dtype, offset, strides));
}

Array::Array(const Array& other, bool copy_memory) {
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

Array::Array(const int& value) : Array(op::jit::wrap_scalar(value)) {}
Array::Array(const double& value) : Array(op::jit::wrap_scalar(value)) {}
Array::Array(const float& value) : Array(op::jit::wrap_scalar(value)) {}

Array Array::zeros(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    Array ret(shape, dtype, preferred_device);
    ret.memory()->lazy_clear();
    return ret;
}

Array Array::empty_like(const Array& other) {
    if (other.is_stateless()) {
        return Array();
    } else {
        // find broadcasted axes:
        std::vector<int> broadcasted_axes;
        const std::vector<int>& other_strides = other.strides();
        for (size_t i = 0; i < other_strides.size(); i++) {
            if (other_strides[i] == 0) {
                broadcasted_axes.emplace_back(i);
            }
        }
        return Array(BufferView::create_with_shape(
                other.shape(), other.dtype(), other.preferred_device(),
                broadcasted_axes));
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

Array Array::ones(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    Array ret(shape, dtype, preferred_device);
    ret = 1.0;
    return ret;
}

Array Array::ones_like(const Array& other) {
    if (other.is_stateless()) {
        return Array();
    } else {
        Array ret = empty_like(other);
        ret = 1.0;
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
        return op::static_as_assignment(*this)->left_.buffer_arg();
    }
    if (is_control_flow()) {
        return op::static_as_control_flow(*this)->left_.buffer_arg();
    }
    // returning false
    // TODO(jonathan): make this return something better
    //                 than a stateless array
    return Array();
}

void Array::eval(bool wait) const {
    if (!is_buffer()) {
        set_expression(canonical(*this).expression());
        auto computable = convert_to_ops(*this);
        // run (DAG evaluation)
        for (auto& step : computable) {
            step->run_and_cleanup();
        }
        ASSERT2(is_buffer(), utils::make_message(
            "After computation expression was not converted "
            "back to a BufferView (expression = ", full_expression_name(), ")."));
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
    bool all_equals = ((float)(Array)op::all_equals(left, right)) > 0 ? true : false;
    return all_equals;
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
    bool is_all_close = ((float)(Array)op::all_close(left, right, atolerance)) > 0 ? true : false;
    return is_all_close;
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
        ret = op::identity(*this);
    }
    ret.expression()->strides_.clear();
    return ret;
}

Array& Array::reset() {
    state_ = nullptr;
    return *this;
}

const std::vector<int>& Array::shape() const {
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
        return op::static_as_buffer_view(*this)->memory_;
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
    return op::astype(*this, dtype_);
}

memory::Device Array::preferred_device() const {
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

int Array::number_of_elements() const {
    return (is_stateless()) ? 0 : hypercube_volume(expression()->shape_);
}

std::vector<int> Array::subshape() const {
    if (is_stateless()) return std::vector<int>();
    if (expression()->shape_.size() == 0) return std::vector<int>();
    return std::vector<int>(expression()->shape_.begin() + 1, expression()->shape_.end());
}

bool Array::contiguous_memory() const {
    return expression()->contiguous_memory();
}

Array Array::operator[](const int& idx) const {
    return pluck_axis(0, idx);
}

Array Array::gather_from_rows(const Array& indices) const {
    throw std::runtime_error("gather_from_rows(const Array& indices) not implemented yet");
}

Array Array::operator[](const Array& indices) const {
    throw std::runtime_error("operator[](const Array& indices) not implemented yet");
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
    if (is_buffer()) {
        return Array((*expression())(idx));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            dest_buffer(idx),
            std::vector<Array>({*this})
        ));
    }
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

Array Array::broadcast_to_shape(const std::vector<int>& shape) const {
    alert_stateless_call(!is_stateless(), "broadcast_to_shape");

    if (is_buffer()) {
        return Array(expression()->broadcast_to_shape(shape));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            dest_buffer.expression()->broadcast_to_shape(shape),
            std::vector<Array>({*this})
        ));
    }
}

Array Array::collapse_axis_with_axis_minus_one(int axis) const {
    alert_stateless_call(!is_stateless(), "collapse_axis_with_axis_minus_one");
    return Array(expression()->collapse_axis_with_axis_minus_one(axis));
}

bool Array::is_axis_collapsible_with_axis_minus_one(int axis) const {
    alert_stateless_call(!is_stateless(), "is_axis_collapsible_with_axis_minus_one");
    return expression()->is_axis_collapsible_with_axis_minus_one(axis);
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

Array Array::pluck_axis(int axis, const Slice& slice) const {
    alert_stateless_call(!is_stateless(), "pluck_axis");
    if (is_buffer()) {
        return Array(expression()->pluck_axis(axis, slice));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            Array(dest_buffer.expression()->pluck_axis(axis, slice)),
            std::vector<Array>({*this})
        ));
    }
}

Array Array::pluck_axis(const int& axis, const int& idx) const {
    alert_stateless_call(!is_stateless(), "pluck_axis");
    if (is_buffer()) {
        return Array(expression()->pluck_axis(axis, idx));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            Array(dest_buffer.expression()->pluck_axis(axis, idx)),
            std::vector<Array>({*this})
        ));
    }
}

Array Array::squeeze(int axis) const {
    alert_stateless_call(!is_stateless(), "squeeze");
    if (is_buffer()) {
        return Array(expression()->squeeze(axis));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            Array(dest_buffer.expression()->squeeze(axis)),
            std::vector<Array>({*this})
        ));
    }
}

Array Array::expand_dims(int new_axis) const {
    alert_stateless_call(!is_stateless(), "expand_dims");
    if (is_buffer()) {
        return Array(expression()->expand_dims(new_axis));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            Array(dest_buffer.expression()->expand_dims(new_axis)),
            std::vector<Array>({*this})
        ));
    }
}

Array Array::broadcast_axis(int axis) const {
    alert_stateless_call(!is_stateless(), "broadcast_axis");
    if (is_buffer()) {
        return Array(expression()->broadcast_axis(axis));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            Array(dest_buffer.expression()->broadcast_axis(axis)),
            std::vector<Array>({*this})
        ));
    }
}

Array Array::insert_broadcast_axis(int new_axis) const {
    alert_stateless_call(!is_stateless(), "insert_broadcast_axis");
    if (is_buffer()) {
        return Array(expression()->insert_broadcast_axis(new_axis));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            Array(dest_buffer.expression()->insert_broadcast_axis(new_axis)),
            std::vector<Array>({*this})
        ));
    }
}

Array Array::broadcast_scalar_to_ndim(const int& ndim) const {
    alert_stateless_call(!is_stateless(), "broadcast_scalar_to_ndim");
    if (is_buffer()) {
        return Array(expression()->broadcast_scalar_to_ndim(ndim));
    } else {
        if (!is_assignment() && !is_control_flow()) {
            set_expression(op::to_assignment(*this).expression());
        }
        auto dest_buffer = buffer_arg();
        return Array(std::make_shared<ControlFlow>(
            Array(dest_buffer.expression()->broadcast_scalar_to_ndim(ndim)),
            std::vector<Array>({*this})
        ));
    }
}



#define DALI_ARRAY_DEFINE_ALL_REDUCER(FUNCTION_NAME, OPNAME)\
    Array Array::FUNCTION_NAME() const {\
        return op::OPNAME(*this);\
    }\

#define DALI_ARRAY_DEFINE_AXIS_REDUCER(FUNCTION_NAME, OPNAME)\
    Array Array::FUNCTION_NAME(const int& axis) const {\
        return op::OPNAME(*this, {axis});\
    }\

#define DALI_ARRAY_DEFINE_REDUCER(FUNCTION_NAME, OPNAME)\
    DALI_ARRAY_DEFINE_ALL_REDUCER(FUNCTION_NAME, OPNAME);\
    DALI_ARRAY_DEFINE_AXIS_REDUCER(FUNCTION_NAME, OPNAME);\

DALI_ARRAY_DEFINE_REDUCER(sum, sum);
DALI_ARRAY_DEFINE_REDUCER(L2_norm, L2_norm);
DALI_ARRAY_DEFINE_REDUCER(mean, mean);
DALI_ARRAY_DEFINE_REDUCER(max, max);
DALI_ARRAY_DEFINE_REDUCER(min, min);
// DALI_ARRAY_DEFINE_REDUCER(argsort, argsort);
// DALI_ARRAY_DEFINE_REDUCER(argmin, argmin);
// DALI_ARRAY_DEFINE_REDUCER(argmax, argmax);

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
    auto assignment = op::assign(Array(expression()), OPERATOR_T_EQL, op::identity(other));
    set_expression(assignment.expression());
    return *this;
}

Array& Array::operator=(const float& other) {
    auto assignment = op::assign(Array(expression()), OPERATOR_T_EQL, op::identity(other));
    set_expression(assignment.expression());
    return *this;
}

Array& Array::operator=(const double& other) {
    auto assignment = op::assign(Array(expression()), OPERATOR_T_EQL, op::identity(other));
    set_expression(assignment.expression());
    return *this;
}

void Array::print(std::basic_ostream<char>& stream, const int& indent, const bool& add_newlines, const bool& print_comma) const {
    eval();
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


#define DALI_DEFINE_ARRAY_INTERACTION_INPLACE(SYMBOL, OPERATOR_NAME)\
    Array& operator SYMBOL (Array& left, const Array& right) {\
        auto assignment = op::assign(Array(left.expression()), OPERATOR_NAME, right);\
        left.set_expression(assignment.expression());\
        return left;\
    }\
    void operator SYMBOL (Array&& left, const Array& right) {\
        auto assignment = op::assign(Array(left.expression()), OPERATOR_NAME, right);\
        left.set_expression(assignment.expression());\
    }\

DALI_DEFINE_ARRAY_INTERACTION_INPLACE(+=, OPERATOR_T_ADD);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(-=, OPERATOR_T_SUB);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(*=, OPERATOR_T_MUL);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(/=, OPERATOR_T_DIV);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(<<=, OPERATOR_T_LSE);

#define DALI_DEFINE_ARRAY_SCALAR_INTERACTION(SYMBOL, OPERATION_NAME, DTYPE)\
    Array operator SYMBOL (const Array& left, DTYPE right) {\
        return OPERATION_NAME(left, right);\
    }\
    Array operator SYMBOL (DTYPE left, const Array& right) {\
        return OPERATION_NAME(left, right);\
    }

#define DALI_DEFINE_ARRAY_INTERACTION(SYMBOL, OPERATION_NAME)\
    Array operator SYMBOL (const Array& left, const Array& right) {\
        return OPERATION_NAME(left, right);\
    }\
    DALI_DEFINE_ARRAY_SCALAR_INTERACTION(SYMBOL, OPERATION_NAME, double);\
    DALI_DEFINE_ARRAY_SCALAR_INTERACTION(SYMBOL, OPERATION_NAME, float);\
    DALI_DEFINE_ARRAY_SCALAR_INTERACTION(SYMBOL, OPERATION_NAME, int);\

DALI_DEFINE_ARRAY_INTERACTION(+, op::add);
DALI_DEFINE_ARRAY_INTERACTION(-, op::subtract);
DALI_DEFINE_ARRAY_INTERACTION(*, op::eltmul);
DALI_DEFINE_ARRAY_INTERACTION(/, op::eltdiv);

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

memory::Device device_promotion(const Array& a, const Array& b) {
    auto apref = a.preferred_device();
    auto bpref = b.preferred_device();
    if (apref == bpref) {return apref;}
    return memory::default_preferred_device;
}
