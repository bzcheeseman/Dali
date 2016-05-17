#include "array.h"

#include <iostream>
#include <ostream>
#include <type_traits>

#include "dali/array/op/other.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/unary.h"
#include "dali/utils/print_utils.h"
#include "dali/array/op/initializer.h"
#include "dali/array/function/operator.h"

using std::vector;
using memory::SynchronizedMemory;

////////////////////////////////////////////////////////////////////////////////
//               MISCELANEOUS UTILITIES (NOT EXPOSED)                         //
////////////////////////////////////////////////////////////////////////////////
// TODO(szymon): create a namespace internal as you did elsewhere?

int hypercube_volume(const vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// if strides are trivial (such that they would arrise from shape normally)
// we remove them.
void compact_strides(vector<int>& strides, const vector<int>& shape) {
    if (strides.size() == 0)
        return;
    ASSERT2(strides.size() == shape.size(),
            "Invalid strides passed to compact_strides.");
    if (shape_to_trivial_strides(shape) == strides) {
        strides.clear();
    }
}

// shape makes sense
bool shape_strictly_positive(const std::vector<int>& shape) {
    return std::all_of(shape.begin(), shape.end(), [](int x) {
        return x > 0;
    });
}


////////////////////////////////////////////////////////////////////////////////
//                        ASSIGNABLE ARRAY                                    //
////////////////////////////////////////////////////////////////////////////////

AssignableArray::AssignableArray(assign_t&& _assign_to) :
        assign_to(_assign_to) {
}

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
        utils::MS() << "Attempting to case array of shape " << shape() << " to a scalar,"
                    << " which is only allowed for a zero-dimensional array.");
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
        utils::MS() << "Attempting to case array of shape " << shape() << " to a scalar,"
                    << " which is only allowed for a zero-dimensional array.");
    ASSERT2(template_to_dtype<T>() == dtype(), "Scalar assign attempted with wrong type.");
    void* data = memory()->mutable_data(memory::Device::cpu());

    return *(((T*)(data)) + offset());
}

void Array::broadcast_axis_internal(int axis) {
    ASSERT2(0 <= axis && axis < ndim(),
            utils::MS() << "broadcast dimension (" << axis << ") must be less the dimensionality of broadcasted tensor (" << ndim() << ")");

    vector<int> new_strides = normalized_strides();
    new_strides[axis] = 0;
    // this will never be needed:
    // compact_strides(new_strides, shape());

    state->strides = new_strides;
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
    ASSERT2(shape_strictly_positive(shape),
            "Shape elements must be strictly positive");
    vector<int> new_strides(strides);
    compact_strides(new_strides, shape);
    ASSERT2(new_strides.size() == 0 || new_strides.size() == shape.size(),
            "Stride and shape size must be the same (unless strides are compacted)");
    state = std::make_shared<ArrayState>(shape, memory, offset, new_strides, dtype);
}

Array::Array(const Array& other, bool copy_memory) {
    if (copy_memory) {
        // TODO(jonathan, szymon):
        // surely we can do better.
        // if memory is broadcasted we do not want to copy
        // entire underlying memory!
        //state = std::make_shared<ArrayState>(*(other.state));
        //state->memory = std::make_shared<SynchronizedMemory>(*(other.state->memory));

        *this = op::identity(other);
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

bool Array::is_scalar() const {
    return ndim() == 0;
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

Array Array::ascontiguousarray() const {
    if (contiguous_memory()) {
        return *this;
    }
    return op::identity(*this);
}



void Array::initialize(const std::vector<int>& shape, DType dtype, memory::Device preferred_device) {
    ASSERT2(shape_strictly_positive(shape),
            "Shape elements must be strictly positive");
    int number_of_elements = hypercube_volume(shape);

    auto memory = std::make_shared<SynchronizedMemory>(
            number_of_elements * size_of_dtype(dtype),
            (shape.size() > 0) ? shape[shape.size()-1] : 1,
            preferred_device
        );

    state = std::make_shared<ArrayState>(shape, memory, 0, vector<int>(), dtype);
}

void Array::initialize_with_bshape(const std::vector<int>& bshape, DType dtype, memory::Device preferred_device) {
    initialize(bshape2shape(bshape), dtype, preferred_device);
    for (int i = 0; i < bshape.size(); ++i) {
        if (bshape[i] < 0) {
            ASSERT2(bshape[i] == -1,
                    "Currently only one-sized broadcasting is supported");
            broadcast_axis_internal(i);
        }
    }
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


std::vector<int> Array::normalized_strides() const {
    return (strides().size() > 0) ? strides() : shape_to_trivial_strides(shape());
}


std::vector<int> Array::bshape() const {
    if (strides().size() == 0) {
        return shape();
    } else {
        vector<int> result(shape());
        for (int i=0; i < ndim(); ++i) {
            if (strides()[i] == 0) {
                // broadcasted dimensions are negated.
                result[i] = -abs(result[i]);
            }
        }
        return result;
    }
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

ArraySlice Array::operator[](const Slice& s) const {
    auto ret = ArraySlice(*this);
    return ret[s];
}

ArraySlice Array::operator[](const Broadcast& b) const {
    auto ret = ArraySlice(*this);
    return ret[b];
}

Array Array::operator()(index_t idx) const {
    ASSERT2(0 <= idx && idx <= number_of_elements(),
            utils::MS() << "Index " << idx << " must be in [0," << number_of_elements() << "].");

    index_t delta_offset;
    if (contiguous_memory()) {
        delta_offset = idx;
    } else {
        vector<int> ns = normalized_strides();
        delta_offset = 0;
        for (int dim = ndim() - 1; dim >= 0; --dim) {
            index_t index_at_dim  = idx % shape()[dim];
            idx /= shape()[dim];
            index_t stride_at_dim = ns[dim];
            delta_offset += index_at_dim * stride_at_dim;
        }
    }

    return Array(vector<int>(),
                 memory(),
                 offset() + delta_offset,
                 vector<int>(),
                 dtype());
}


Array Array::transpose() const {
    std::vector<int> permuation(ndim(), 0);
    for (int i = 0; i < ndim(); ++i) {
        permuation[i] = ndim() - i - 1;
    }
    return transpose(permuation);
}

Array Array::transpose(const std::vector<int>& axes) const {
    const std::vector<int>& old_shape   = shape();
    std::vector<int>        old_strides = normalized_strides();

    std::vector<int> new_shape(ndim());
    std::vector<int> new_strides(ndim());

    for (int i = 0; i < ndim(); ++i) {
        new_shape[i]   = old_shape[axes[i]];
        new_strides[i] = old_strides[axes[i]];
    }

    return Array(new_shape,
                 memory(),
                 offset(),
                 new_strides,
                 dtype());
}

Array Array::ravel() const {
    ASSERT2(contiguous_memory(),
            "at the moment ravel is only supported for contiguous_memory");
    return Array({number_of_elements()},
                 memory(),
                 offset(),
                 std::vector<int>(),
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

Array Array::reshape_broadcasted(const std::vector<int>& new_shape) const {
    ASSERT2(new_shape.size() == ndim(),
            utils::MS() << "reshape_broadcasted must receive a shape with the same dimensionality (current: " <<
            shape() << ", got: " << new_shape << ")");
    auto my_bshape = bshape();

    for (int i = 0; i < my_bshape.size(); ++i) {
        ASSERT2(new_shape[i] > 0,
                utils::MS() << "reshape_broadcasted's new_shape argument must be strictly positive (got "
                            << new_shape << ")");

        ASSERT2(new_shape[i] == std::abs(my_bshape[i]) || my_bshape[i] == -1,
                utils::MS() << "reshape_broadcasted can only reshape broadcasted dimensions "
                            << "(tried to reshape array with shape: "
                            << my_bshape << " to new shape: " << new_shape << ")");
    }
    return Array(new_shape,
                 memory(),
                 offset(),
                 strides(),
                 dtype());
}


Array Array::pluck_axis(int axis, int pluck_idx) const {
    auto single_item_slice = pluck_axis(axis, Slice(pluck_idx, pluck_idx + 1));
    return single_item_slice.squeeze(axis);
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


    return Array(new_shape,
                 memory(),
                 new_offset,
                 new_strides,
                 dtype());
}
Array Array::squeeze(int axis) const {
    ASSERT2(0 <= axis && axis < shape().size(),
            utils::MS() << "squeeze dimension (" << axis << ") must be less the dimensionality of compacted tensor (" << shape().size() << ")");
    ASSERT2(shape()[axis] == 1,
            utils::MS() << "squeeze(" << axis << ") requires tensor to be shaped like a bowtie.");

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

    return Array(new_shape,
                 memory(),
                 offset(),
                 new_strides,
                 dtype());
}

Array Array::expand_dims(int new_axis) const {
    vector<int> new_shape   = shape();
    vector<int> new_strides = normalized_strides();

    new_shape.insert(  new_shape.begin()   + new_axis, 1);
    // It really does not matter what the new stride is going to be,
    // because in we are only ever going to access it at index 0,
    // so it will get cancelled out. We chose to set it to the stride that
    // naturally arises in the shape_to_trivial_strides, such that if other strides
    // were already trivial it will get compacted.
    new_strides.insert(new_strides.begin() + new_axis, shape_to_trivial_strides(new_shape)[new_axis]);
    return Array(new_shape,
                 memory(),
                 offset(),
                 new_strides,
                 dtype());
}


Array Array::broadcast_axis(int axis) const {
    Array out(*this, false);
    out.broadcast_axis_internal(axis);
    return out;
}

Array Array::insert_broadcast_axis(int new_axis) const {
    return expand_dims(new_axis).broadcast_axis(new_axis);
}


Array Array::broadcast_scalar_to_ndim(int target_ndim) const {
    ASSERT2(is_scalar(),
            utils::MS() << "broadcast_scalar_to_ndim may only be called on scalars, got shape " << shape() << ".");
    Array res = *this;
    for (int i = 0; i < target_ndim; ++i) {
        res = res.insert_broadcast_axis(0);
    }
    return res;
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

Array& Array::operator<<=(const AssignableArray& assignable) {
    assignable.assign_to(*this, OPERATOR_T_LSE);
    return *this;
}

void Array::print(std::basic_ostream<char>& stream, int indent, bool add_newlines) const {
    std::string end_line_spacing = add_newlines ? "\n" : "";
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

        for(int i = 0; i < state->shape[0]; i += 1) {
            stream << std::fixed
                      << std::setw( 7 ) /* keep 7 digits*/
                      << std::setprecision( 3 ) /* use 3 decimals*/
                      << std::setfill( ' ' );
            Array scalar = (*this)[i];
            scalar.print(stream, 0, false);
            if (i != state->shape[0] - 1) stream << " ";
        }
        stream << "]";
        stream << end_line_spacing;
    } else {
        stream << std::string(indent, ' ') << "[" << end_line_spacing;
        for (int i = 0; i < state->shape[0]; ++i) {
            Array subtensor = (*this)[i];
            subtensor.print(stream, indent + indent_increment, add_newlines);
        }
        stream << std::string(indent, ' ') << "]" << end_line_spacing;
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
        consumed_dims(0),
        slice({}),
        action({}) {
}

ArraySlice::ArraySlice(const ArraySlice& other) :
        input(other.input),
        slice(other.slice),
        consumed_dims(other.consumed_dims),
        action(other.action) {
}


ArraySlice ArraySlice::operator[](const Slice& s) {
    ASSERT2(consumed_dims < input.ndim(),
        "Slicing a scalar array is not allowed.");
    ArraySlice res(*this);
    res.slice.push_back(s);
    res.action.push_back(SLICE_RANGE);
    res.consumed_dims += 1;
    return res;
}

ArraySlice ArraySlice::operator[](const Broadcast& b) {
    ArraySlice res(*this);
    res.action.push_back(BROADCAST);
    // res.consumed_dims remains unchanged during broadcast.
    return res;
}

ArraySlice ArraySlice::operator[](int idx) {
    ASSERT2(consumed_dims < input.ndim(),
        "Slicing a scalar array is not allowed.");
    ArraySlice res(*this);
    res.slice.push_back(Slice(idx, idx+1));
    res.action.push_back(SLICE_IDX);
    res.consumed_dims += 1;
    return res;
}

ArraySlice::operator Array() {
    Array out = input;
    ASSERT2(consumed_dims <= input.ndim(),
            "Email szymon.sidor@gmail.com.");
    auto next_slice = slice.begin();
    int output_depth = 0;
    for (const auto& a: action) {
        switch(a) {
            case SLICE_RANGE:
                out = out.pluck_axis(output_depth, *(next_slice++));
                output_depth += 1;
                break;
            case SLICE_IDX:
                out = out.pluck_axis(output_depth, *(next_slice++));
                out = out.squeeze(output_depth);
                break;
            case BROADCAST:
                out = out.insert_broadcast_axis(output_depth);
                output_depth += 1;
                break;
            default:
                ASSERT2(false, "Unsupported value for ArraySliceAction.");
                break;
        }
    }
    return out;
}
