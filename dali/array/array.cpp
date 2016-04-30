#include "array.h"

#include <iostream>
#include <ostream>
#include <type_traits>

#include "dali/array/op/other.h"
#include "dali/array/op/reducers.h"
#include "dali/utils/print_utils.h"
#include "dali/array/op/initializer.h"

using std::vector;
using memory::SynchronizedMemory;

////////////////////////////////////////////////////////////////////////////////
//               MISCELANEOUS UTILITIES (NOT EXPOSED)                         //
////////////////////////////////////////////////////////////////////////////////

int hypercube_volume(const vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
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
    assignable.assign_to(*this);
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


Array Array::operator[](index_t idx) const {
    ASSERT2(contiguous_memory(),
            "at the moment slicing is only supported for contiguous_memory");
    ASSERT2(shape().size() > 0, "Slicing a scalar array is not allowed.");
    ASSERT2(0 <= idx && idx < shape()[0], utils::MS() << "Index " << idx << " must be in [0," << shape()[0] << "].");
    return Array(subshape(),
                 state->memory,
                 state->offset + hypercube_volume(subshape()) * idx,
                 vector<int>(),
                 state->dtype);
}

Array Array::operator()(index_t idx) const {
    ASSERT2(contiguous_memory(),
            "at the moment slicing is only supported for contiguous_memory");
    ASSERT2(0 <= idx && idx <= number_of_elements(),
            utils::MS() << "Index " << idx << " must be in [0," << number_of_elements() << "].");
    return Array(vector<int>(),
                 state->memory,
                 state->offset + idx,
                 vector<int>(),
                 state->dtype);
}

Array Array::ravel() const {
    ASSERT2(contiguous_memory(),
            "at the moment ravel is only supported for contiguous_memory");
    return Array({number_of_elements()},
                 state->memory,
                 state->offset,
                 vector<int>(),
                 state->dtype);
}

Array Array::reshape(const vector<int>& new_shape) const {
    ASSERT2(contiguous_memory(),
            "at the moment reshape is only supported for contiguous_memory");
    ASSERT2(hypercube_volume(new_shape) == number_of_elements(),
            utils::MS() << "New shape (" << new_shape << ") must have the same nubmer of elements as previous shape (" << shape() << ")");
    return Array(new_shape,
                 state->memory,
                 state->offset,
                 vector<int>(),
                 state->dtype);
}
// TODO(jonathan,szymon): add axis argument to sum + write tests
AssignableArray Array::sum() const {
    return op::sum_all(*this);
}

// TODO(jonathan,szymon): add axis argument to mean + write tests
AssignableArray Array::mean() const {
    return op::mean_all(*this);
}


template<typename T>
Array::operator T() const {
    return scalar_value<T>();
}

template Array::operator float() const;
template Array::operator double() const;
template Array::operator int() const;

template<typename T>
Array::operator T&() {
    return scalar_value<T>();
}

template Array::operator float&();
template Array::operator double&();
template Array::operator int&();

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
    assignable.assign_to(*this);
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

        for(int i = 0; i < state->shape[0]; ++i) {
            stream << std::fixed
                      << std::setw( 7 ) /* keep 7 digits*/
                      << std::setprecision( 3 ) /* use 3 decimals*/
                      << std::setfill( ' ' );
            (*this)[i].print(stream);
            if (i != state->shape[0] - 1) stream << " ";
        }
        stream << "]";
        stream << std::endl;
    } else {
        stream << std::string(indent, ' ') << "[" << std::endl;
        for (int i = 0; i < state->shape[0]; ++i)
            (*this)[i].print(stream, indent + 4);
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
