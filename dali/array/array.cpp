#include "array.h"

#include <iostream>
#include <ostream>
#include <type_traits>

#include "dali/array/op/other.h"
#include "dali/array/op/reducers.h"
#include "dali/utils/print_utils.h"
#include "dali/array/op/initializer.h"
#include "dali/array/function/operator.h"

using std::vector;
using memory::SynchronizedMemory;

////////////////////////////////////////////////////////////////////////////////
//               MISCELANEOUS UTILITIES (NOT EXPOSED)                         //
////////////////////////////////////////////////////////////////////////////////

int hypercube_volume(const vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

vector<int> trivial_strides(const vector<int>& shape) {
    vector<int> res(shape.size());
    int residual_shape = 1;
    for (int i = shape.size() - 1; i >= 0 ; --i) {
        res[i] = residual_shape;
        residual_shape *= shape[i];
    }
    return res;
}

// if strides are trivial (such that they would arrise from shape normally)
// we remove them.
void compact_strides(vector<int>& strides, const vector<int>& shape) {
    if (strides.size() == 0)
        return;
    ASSERT2(strides.size() == shape.size(),
            "Invalid strides passed to compact_strides.");
    if (trivial_strides(shape) == strides) {
        strides.clear();
    }
}


////////////////////////////////////////////////////////////////////////////////
//                        ASSIGNABLE ARRAY                                    //
////////////////////////////////////////////////////////////////////////////////

AssignableArray::AssignableArray(assign_t&& _assign_to) : assign_to(_assign_to) {}

AssignableArray::AssignableArray(const float& constant) :
        AssignableArray(initializer::fill(constant)) {
}

AssignableArray::AssignableArray(const double& constant) :
        AssignableArray(initializer::fill(constant)) {
}

AssignableArray::AssignableArray(const int& constant) :
        AssignableArray(initializer::fill(constant)) {
}

////////////////////////////////////////////////////////////////////////////////
//                              ARRAY STATE                                   //
////////////////////////////////////////////////////////////////////////////////


ArrayState::ArrayState(const std::vector<int>& _shape,
                       std::shared_ptr<SynchronizedMemory> _memory,
                       int _offset,
                       const std::vector<int>& _strides,
                       DType _dtype) :
        shape(_shape),
        memory(_memory),
        offset(_offset),
        strides(_strides),
        dtype(_dtype) {
}


////////////////////////////////////////////////////////////////////////////////
//                                 ARRAY                                      //
////////////////////////////////////////////////////////////////////////////////

template<typename T>
T Array::scalar_value() const {
    static_assert(std::is_arithmetic<T>::value,
            "Scalar value only available for arithmetic types (integer or real).");
    ASSERT2(
        shape().size() == 0,
        "Scalar value can only be requested for scalar (ndim zero) Array."
    );
    void* data = memory()->data(memory::Device::cpu());
    if (dtype() == DTYPE_FLOAT) {
        return *((float*)(data) + offset());
    } else if (dtype() == DTYPE_DOUBLE) {
        return *((double*)(data) + offset());
    } else if (dtype() == DTYPE_INT32) {
        return *((int*)(data) + offset());
    }
}

template<typename T>
T& Array::scalar_value() {
    static_assert(std::is_arithmetic<T>::value,
            "Scalar value only available for arithmetic types (integer or real).");
    ASSERT2(
        shape().size() == 0,
        "Scalar value can only be requested for scalar (ndim zero) Array."
    );
    ASSERT2(template_to_dtype<T>() == dtype(), "Scalar assign attempted with wrong type.");
    void* data = memory()->mutable_data(memory::Device::cpu());

    return *(((T*)(data)) + offset());
}


Array::Array() {}

Array::Array(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    initialize(shape, dtype, preferred_device);
}

Array::Array(std::initializer_list<int> shape_, DType dtype, memory::Device preferred_device) :
        Array(vector<int>(shape_), dtype, preferred_device) {
}

Array::Array(const std::vector<int>& shape,
             std::shared_ptr<SynchronizedMemory> memory,
             int offset,
             const std::vector<int>& strides,
             DType dtype) {
    state = std::make_shared<ArrayState>(shape, memory, offset, strides, dtype);
}

Array::Array(const Array& other, bool copy_memory) {
    if (copy_memory) {
        // TODO(jonathan, szymon):
        // surely we can do better.
        // if memory is broadcasted we do not want to copy
        // entire underlying memory!
        state = std::make_shared<ArrayState>(*(other.state));
        state->memory = std::make_shared<SynchronizedMemory>(*(other.state->memory));
    } else {
        state = other.state;
    }
}

Array::Array(const AssignableArray& assignable) {
    assignable.assign_to(*this, OPERATOR_T_EQL);
}

Array Array::zeros(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    Array ret(shape, dtype, preferred_device);
    ret.memory()->lazy_clear();
    return ret;
}

Array Array::zeros_like(const Array& other) {
    if (other.is_stateless()) {
        return Array();
    } else {
        return zeros(other.shape(), other.dtype(), other.memory()->preferred_device);
    }
}


Array Array::arange(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    Array ret(shape, dtype, preferred_device);
    ret = initializer::arange();
    return ret;
}

Array Array::ones(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    Array ret(shape, dtype, preferred_device);
    ret = initializer::ones();
    return ret;
}

Array Array::ones_like(const Array& other) {
    if (other.is_stateless()) {
        return Array();
    } else {
        return zeros(other.shape(), other.dtype(), other.memory()->preferred_device);
    }
}

bool Array::is_stateless() const {
    return state == nullptr;
}

bool Array::spans_entire_memory() const {
    ASSERT2(!is_stateless(), "spans_entire_memory must not be called with stateless Array.");
    return offset() == 0 &&
           number_of_elements() * size_of_dtype(dtype()) == memory()->total_memory;
}

bool Array::contiguous_memory() const {
    ASSERT2(!is_stateless(), "contiguous_memory must not be called with stateless Array.");

    return strides().empty() == true;
}

void Array::initialize(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    int number_of_elements = hypercube_volume(shape);

    auto memory = std::make_shared<SynchronizedMemory>(
            number_of_elements * size_of_dtype(dtype),
            (shape.size() > 0) ? shape[shape.size()-1] : 1,
            preferred_device
        );

    state = std::make_shared<ArrayState>(shape, memory, 0, vector<int>(), dtype);
}

Array& Array::reset() {
    state = nullptr;
    return *this;
}


const vector<int>& Array::shape() const {
    ASSERT2(state != nullptr, "shape must not be called on Array initialized with empty constructor");
    return state->shape;
}


std::shared_ptr<memory::SynchronizedMemory> Array::memory() const {
    if (state == nullptr) {
        return nullptr;
    } else {
        return state->memory;
    }
}

int Array::offset() const {
    ASSERT2(state != nullptr, "offset must not be called on Array initialled with empty constructor");
    return state->offset;
}

const std::vector<int>& Array::strides() const {
    ASSERT2(state != nullptr, "strides must not be called on Array initialled with empty constructor");
    return state->strides;
}

std::vector<int> Array::normalized_strides() const {
    return (strides().size() > 0) ? strides() : trivial_strides(shape());
}

DType Array::dtype() const {
    ASSERT2(state != nullptr, "dtype must not be called on Array initialled with empty constructor");
    return state->dtype;
}

memory::Device Array::preferred_device() const {
    ASSERT2(!is_stateless(), "preferred_device must not be called on Array initialled with empty constructor");
    return state->memory->preferred_device;
}

void Array::to_device(memory::Device device) const {
    memory()->move_to(device);
}


int Array::ndim() const {
    return (state == nullptr) ? 0 : state->shape.size();

}

int Array::number_of_elements() const {
    return (state == nullptr) ? 0 : hypercube_volume(state->shape);
}


vector<int> Array::subshape() const {
    if (state == nullptr) return vector<int>();
    if (state->shape.size() == 0) return vector<int>();
    return vector<int>(state->shape.begin() + 1, state->shape.end());
}


Array Array::operator[](int idx) const {
    return pluck_axis(0, idx);
}

ArraySlice Array::operator[](Slice s) const {
    auto ret = ArraySlice(*this);
    return ret[s];
}

Array Array::operator()(index_t idx) const {
    ASSERT2(contiguous_memory(),
            "at the moment slicing is only supported for contiguous_memory");
    ASSERT2(0 <= idx && idx <= number_of_elements(),
            utils::MS() << "Index " << idx << " must be in [0," << number_of_elements() << "].");
    return Array(vector<int>(),
                 memory(),
                 offset() + idx,
                 vector<int>(),
                 dtype());
}

Array Array::ravel() const {
    ASSERT2(contiguous_memory(),
            "at the moment ravel is only supported for contiguous_memory");
    return Array({number_of_elements()},
                 memory(),
                 offset(),
                 vector<int>(),
                 dtype());
}

Array Array::reshape(const vector<int>& new_shape) const {
    ASSERT2(contiguous_memory(),
            "at the moment reshape is only supported for contiguous_memory");
    ASSERT2(hypercube_volume(new_shape) == number_of_elements(),
            utils::MS() << "New shape (" << new_shape << ") must have the same nubmer of elements as previous shape (" << shape() << ")");
    return Array(new_shape,
                 memory(),
                 offset(),
                 vector<int>(),
                 dtype());
}

Array Array::collapse_axis(int axis) const {
    ASSERT2(axis < shape().size(),
            utils::MS() << "collapse_axis dimension (" << axis << ") must be less the dimensionality of compacted tensor (" << shape().size() << ")");
    ASSERT2(shape()[axis] == 1,
            utils::MS() << "collapse_axis(" << axis << ") requires tensor to be shaped like a bowtie.");

    const vector<int>& old_shape = shape();
    auto old_strides             = normalized_strides();

    vector<int> new_shape;
    vector<int> new_strides;
    for (int i = 0; i < old_shape.size(); ++i) {
        if (i == axis) {
            continue;
        }
        new_shape.push_back(old_shape[i]);
        new_strides.push_back(old_strides[i]);
    }

    compact_strides(new_strides, new_shape);

    return Array(new_shape,
                 memory(),
                 offset(),
                 new_strides,
                 dtype());
}


Array Array::pluck_axis(int axis, int pluck_idx) const {
    auto single_item_slice = pluck_axis(axis, Slice(pluck_idx, pluck_idx + 1));
    return single_item_slice.collapse_axis(axis);
}

Array Array::pluck_axis(int axis, const Slice& slice_unnormalized) const {
    ASSERT2(axis < shape().size(),
            utils::MS() << "pluck_axis dimension (" << axis << ") must be less the dimensionality of plucked tensor (" << shape().size() << ")");

    Slice slice = Slice::normalize_and_check(slice_unnormalized, shape()[axis]);

    const vector<int>& old_shape = shape();
    auto old_strides             = normalized_strides();

    vector<int> new_shape(old_shape);
    vector<int> new_strides(old_strides);

    new_shape[axis]    = slice.size();
    new_strides[axis] *= slice.step;

    int new_offset;
    if (slice.step > 0) {
        new_offset = offset() + old_strides[axis] * slice.start;
    } else {
        new_offset = offset() + old_strides[axis] * (slice.end - 1);
    }

    compact_strides(new_strides, new_shape);

    return Array(new_shape,
                 memory(),
                 new_offset,
                 new_strides,
                 dtype());
}
Array Array::broadcast_axis(int new_axis) const {
    vector<int> new_shape   = shape();
    vector<int> new_strides = normalized_strides();
    new_shape.insert(  new_shape.begin()   + new_axis, 1);
    new_strides.insert(new_strides.begin() + new_axis, 0);
    return Array(new_shape,
                 memory(),
                 offset(),
                 new_strides,
                 dtype());
}


// TODO(jonathan,szymon): add axis argument to sum + write tests
AssignableArray Array::sum() const {
    return op::sum_all(*this);
}

// TODO(jonathan,szymon): add axis argument to mean + write tests
AssignableArray Array::mean() const {
    return op::mean_all(*this);
}

Array::operator float() const {
    return scalar_value<float>();
}
Array::operator double() const {
    return scalar_value<double>();
}
Array::operator int() const {
    return scalar_value<int>();
}

Array::operator float&() {
    return scalar_value<float>();
}
Array::operator double&() {
    return scalar_value<double>();
}
Array::operator int&() {
    return scalar_value<int>();
}

template<typename T>
Array& Array::assign_constant(const T& other) {
    static_assert(std::is_arithmetic<T>::value,
            "Scalar value can only be assigned from arithmetic type.");

    if (dtype() == DTYPE_FLOAT) {
        scalar_value<float>() = other;
    } else if(dtype() == DTYPE_DOUBLE) {
        scalar_value<double>() = other;
    } else if(dtype() == DTYPE_INT32) {
        scalar_value<int>() = other;
    }
    return *this;
}

Array& Array::operator=(const AssignableArray& assignable) {
    assignable.assign_to(*this, OPERATOR_T_EQL);
    return *this;
}

Array& Array::operator+=(const AssignableArray& assignable) {
    assignable.assign_to(*this, OPERATOR_T_ADD);
    return *this;
}

Array& Array::operator-=(const AssignableArray& assignable) {
    assignable.assign_to(*this, OPERATOR_T_SUB);
    return *this;
}

Array& Array::operator*=(const AssignableArray& assignable) {
    assignable.assign_to(*this, OPERATOR_T_MUL);
    return *this;
}

Array& Array::operator/=(const AssignableArray& assignable) {
    assignable.assign_to(*this, OPERATOR_T_DIV);
    return *this;
}

void Array::print(std::basic_ostream<char>& stream, int indent) const {
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
    } else if (ndim() == 1) {
        stream << std::string(indent, ' ');
        stream << "[";

        for(int i = 0; i < state->shape[0]; i += 1) {
            stream << std::fixed
                      << std::setw( 7 ) /* keep 7 digits*/
                      << std::setprecision( 3 ) /* use 3 decimals*/
                      << std::setfill( ' ' );
            Array scalar = (*this)[i];
            scalar.print(stream);
            if (i != state->shape[0] - 1) stream << " ";
        }
        stream << "]";
        stream << std::endl;
    } else {
        stream << std::string(indent, ' ') << "[" << std::endl;
        for (int i = 0; i < state->shape[0]; ++i) {
            Array subtensor = (*this)[i];
            subtensor.print(stream, indent + 4);
        }
        stream << std::string(indent, ' ') <<"]" << std::endl;
    }
}

void Array::debug_memory(bool print_contents) const {
    memory()->debug_info(std::cout, print_contents, dtype());
}

void Array::clear() {
    if (spans_entire_memory()) {
        memory()->lazy_clear();
    } else {
        *this = initializer::fill(0.0);
    }
}


////////////////////////////////////////////////////////////////////////////////
//                        ARRAY SLICE                                         //
////////////////////////////////////////////////////////////////////////////////

ArraySlice::ArraySlice(const Array& input_) :
        input(input_),
        slice({}),
        collapse({}) {
}

ArraySlice::ArraySlice(const ArraySlice& other) :
        input(other.input),
        slice(other.slice.begin(), other.slice.end()),
        collapse(other.collapse.begin(), other.collapse.end()) {
}


ArraySlice ArraySlice::operator[](Slice s) {
    ASSERT2(slice.size() < input.ndim(),
        "Slicing a scalar array is not allowed.");
    ArraySlice res(*this);
    res.slice.push_back(s);
    res.collapse.push_back(false);
    return res;
}

ArraySlice ArraySlice::operator[](int idx) {
    ASSERT2(slice.size() < input.ndim(),
        "Slicing a scalar array is not allowed.");
    ArraySlice res(*this);
    res.slice.push_back(Slice(idx, idx+1));
    res.collapse.push_back(true);
    return res;
}

ArraySlice::operator Array() {
    Array out = input;
    ASSERT2(slice.size() == collapse.size(),
            "Email szymon.sidor@gmail.com.");
    int dim = 0;
    for (int i = 0; i < slice.size(); ++i) {
        out = out.pluck_axis(dim, slice[i]);
        if (collapse[i]) {
            out = out.collapse_axis(dim);
        } else {
            dim++;
        }
    }
    return out;
}
