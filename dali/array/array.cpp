#include "array.h"

#include <iostream>
#include <ostream>
#include <type_traits>

#include "dali/utils/print_utils.h"
#include "dali/array/assignable_array.h"

using std::vector;
using memory::SynchronizedMemory;

////////////////////////////////////////////////////////////////////////////////
//               MISCELANEOUS UTILITIES (NOT EXPOSED)                         //
////////////////////////////////////////////////////////////////////////////////

int hypercube_volume(const vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

////////////////////////////////////////////////////////////////////////////////
//                              ARRAY STATE                                   //
////////////////////////////////////////////////////////////////////////////////


ArrayState::ArrayState(const std::vector<int>& _shape,
                       std::shared_ptr<SynchronizedMemory> _memory,
                       int _offset,
                       DType _dtype) :
        shape(_shape),
        memory(_memory),
        offset(_offset),
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
        "Scalar value can only be requested for scalar (dimension zero) Array."
    );
    void* data = memory()->data(memory::Device::cpu());
    if (dtype() == DTYPE_FLOAT) {
        float res = *((float*)(data) + offset());
        return (T)res;
    } else if (dtype() == DTYPE_DOUBLE) {
        double res = *((double*)(data) + offset());
        return (T)res;
    } else if (dtype() == DTYPE_INT32) {
        int res = *((int*)(data) + offset());
        return (T)res;
    }
}

template<typename T>
T& Array::scalar_value() {
    static_assert(std::is_arithmetic<T>::value,
            "Scalar value only available for arithmetic types (integer or real).");
    ASSERT2(
        shape().size() == 0,
        "Scalar value can only be requested for scalar (dimension zero) Array."
    );
    ASSERT2(dtype_is<T>(dtype()), "Scalar assign attempted with wrong type.");
    void* data = memory()->mutable_data(memory::Device::cpu());
    return *(((T*)(data)) + offset());
}


Array::Array() {}

Array::Array(const std::vector<int>& shape, DType dtype) {
    initialize(shape, dtype);
}

Array::Array(std::initializer_list<int> shape_, DType dtype) :
        Array(vector<int>(shape_), dtype) {
}

Array::Array(const std::vector<int>& shape, std::shared_ptr<SynchronizedMemory> memory, int offset, DType dtype) {
    state = std::make_shared<ArrayState>(shape, memory, offset, dtype);
}

Array::Array(const Array& other, bool copy_memory) {
    if (copy_memory) {
        state = std::make_shared<ArrayState>(*(other.state));
        state->memory = std::make_shared<SynchronizedMemory>(*(other.state->memory));
    } else {
        state = other.state;
    }
}

Array::Array(const AssignableArray& assignable) {
    assignable.assign_to(*this);
}

Array Array::zeros(const std::vector<int>& shape, DType dtype) {
    Array ret(shape, dtype);
    ret.memory()->lazy_clear();
    return ret;
}

Array Array::zeros_like(const Array& other) {
    return zeros(other.shape(), other.dtype());
}

bool Array::is_stateless() const {
    return state == nullptr;
}

void Array::initialize(const std::vector<int>& shape, DType dtype) {
    int number_of_elements = hypercube_volume(shape);

    auto memory = std::make_shared<SynchronizedMemory>(
            number_of_elements * size_of_dtype(dtype),
            (shape.size() > 0) ? shape[shape.size()-1] : 1);

    state = std::make_shared<ArrayState>(shape, memory, 0, dtype);
}

Array& Array::reset() {
    state = nullptr;
    return *this;
}


static vector<int> empty_vector;

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

DType Array::dtype() const {
    ASSERT2(state != nullptr, "dtype must not be called on Array initialled with empty constructor");
    return state->dtype;
}


int Array::dimension() const {
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
    ASSERT2(shape().size() > 0, "Slicing a scalar array is not allowed.");
    ASSERT2(0 <= idx && idx < shape()[0], utils::MS() << "Index " << idx << " must be in [0," << shape()[0] << "].");
    return Array(subshape(),
                 state->memory,
                 state->offset + hypercube_volume(subshape()) * idx,
                 state->dtype);
}

Array Array::operator()(index_t idx) const {
    ASSERT2(0 <= idx && idx <= number_of_elements(),
            utils::MS() << "Index " << idx << " must be in [0," << number_of_elements() << "].");
    return Array(vector<int>(),
                 state->memory,
                 state->offset + idx,
                 state->dtype);
}

Array Array::ravel() const {
    return Array({number_of_elements()},
                 state->memory,
                 state->offset,
                 state->dtype);
}

Array Array::reshape(const vector<int>& new_shape) const {
    ASSERT2(hypercube_volume(new_shape) == number_of_elements(),
            utils::MS() << "New shape (" << new_shape << ") must have the same nubmer of elements as previous shape (" << shape() << ")");
    return Array(new_shape,
                 state->memory,
                 state->offset,
                 state->dtype);
}

template<typename T>
Array::operator const T&() const {
    return scalar_value<T>();
}

template Array::operator const float&() const;
template Array::operator const double&() const;
template Array::operator const int&() const;

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

Array& Array::operator=(const float& other) {
    return assign_constant<float>(other);
}

Array& Array::operator=(const double& other) {
    return assign_constant<double>(other);
}

Array& Array::operator=(const int& other) {
    return assign_constant<int>(other);
}

Array& Array::operator=(const AssignableArray& assignable) {
    assignable.assign_to(*this);
    return *this;
}

void Array::print(std::basic_ostream<char>& stream, int indent) const {
    if (dimension() == 0) {
        if (dtype() == DTYPE_FLOAT) {
            stream << (const float&)(*this);
        } else if (dtype() == DTYPE_DOUBLE) {
            stream << (const double&)(*this);
        } else if (dtype() == DTYPE_INT32) {
            stream << (const int&)(*this);
        } else {
            ASSERT2(false, "Wrong dtype for Array.");
        }
    } else if (dimension() == 1) {
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
