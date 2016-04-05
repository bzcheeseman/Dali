#include "array.h"

#include <iostream>
#include <ostream>


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
T Array::scalar_value() {
    ASSERT2(
        shape().size() == 0,
        "Scalar value can only be requested for scalar (dimension zero) Array."
    );
    void* data = memory()->data(memory::Device::cpu());
    if (dtype() == DTYPE_FLOAT) {
        float res = *((float*)(data));
        return (T)res;
    } else if (dtype() == DTYPE_DOUBLE) {
        double res = *((double*)(data));
        return (T)res;
    } else if (dtype() == DTYPE_INT32) {
        int res = *((int*)(data));
        return (T)res;
    }
}


Array::Array() {}

Array::Array(const std::vector<int>& shape, DType dtype) {
    int number_of_elements = hypercube_volume(shape);

    auto memory = std::make_shared<SynchronizedMemory>(
            number_of_elements * size_of_dtype(dtype),
            (shape.size() > 0) ? shape[shape.size()-1] : 1);

    state = std::make_shared<ArrayState>(shape, memory, 0, dtype);
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

std::shared_ptr<memory::SynchronizedMemory> Array::memory() const {
    if (state == nullptr) {
        return nullptr;
    } else {
        return state->memory;
    }
}

DType Array::dtype() const {
    ASSERT2(state != nullptr, " dtype must not be called on Array initialled with empty constructor");
    return state->dtype;
}


int Array::dimension() const {
    return (state == nullptr) ? 0 : state->shape.size();

}

int Array::number_of_elements() const {
    return (state == nullptr) ? 0 : hypercube_volume(state->shape);

}

static vector<int> empty_vector;

const vector<int>& Array::shape() const {
    if (state == nullptr) {
        return empty_vector;
    } else {
        return state->shape;
    }
}

vector<int> Array::subshape() const {
    if (state == nullptr) return vector<int>();
    if (state->shape.size() == 0) return vector<int>();
    return vector<int>(state->shape.begin() + 1, state->shape.end());
}


Array Array::operator[](index_t idx) const {
    ASSERT2(0 <= idx && idx <= shape()[0], utils::MS() << "Index " << idx << " must be in [0," << shape()[0] << "].");
    return Array(subshape(),
                 state->memory,
                 state->offset + hypercube_volume(subshape()) * idx,
                 state->dtype);
}

Array Array::operator()(index_t idx) const {
    ASSERT2(0 <= idx && idx <= number_of_elements(), utils::MS() << "Index " << idx << " must be in [0," << number_of_elements() << "].");
    return Array(vector<int>(),
                 state->memory,
                 state->offset + idx,
                 state->dtype);
}

Array::operator float() {
    return scalar_value<float>();
}
Array::operator double() {
    return scalar_value<double>();
}
Array::operator int() {
    return scalar_value<int>();
}

void Array::print(std::basic_ostream<char>& stream, int indent) const {
    if (dimension() == 0) {
        print_dtype(stream, state->dtype, memory()->data(memory::Device::cpu())); /* pad values with blanks this->w(i,j)*/
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
