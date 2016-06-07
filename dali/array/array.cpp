#include "array.h"

#include <iostream>
#include <ostream>
#include <type_traits>

#include "dali/array/op/other.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/unary_scalar.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/reshape.h"
#include "dali/array/op/dot.h"
#include "dali/utils/print_utils.h"
#include "dali/array/op/initializer.h"
#include "dali/array/function/operator.h"
#include "dali/utils/cnpy.h"

using std::vector;
using memory::SynchronizedMemory;

////////////////////////////////////////////////////////////////////////////////
//               MISCELANEOUS UTILITIES (NOT EXPOSED)                         //
////////////////////////////////////////////////////////////////////////////////
// TODO(szymon): create a namespace internal as you did elsewhere?

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
                       const int& _offset,
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
        utils::MS() << "Attempting to cast array of shape " << shape() << " to a scalar,"
                    << " which is only allowed for a zero-dimensional array.");

    void* data = memory()->data(memory::Device::cpu());
    if (dtype() == DTYPE_FLOAT) {
        return *((float*)(data) + offset());
    } else if (dtype() == DTYPE_DOUBLE) {
        return *((double*)(data) + offset());
    } else if (dtype() == DTYPE_INT32) {
        return *((int*)(data) + offset());
    }
    return 0;
}

void Array::broadcast_axis_internal(const int& axis) {
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
             const int& offset,
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

Array::Array(const Array& other, const bool& copy_memory) {
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
Array Array::empty_like(const Array& other) {
    if (other.is_stateless()) {
        return Array();
    } else {
        Array ret;
        ret.initialize_with_bshape(
            other.bshape(), other.dtype(), other.memory()->preferred_device
        );
        return ret;
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
        Array ret = empty_like(other);
        ret = initializer::ones();
        return ret;
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
                utils::MS() << "attempted to load a npy array of floats with dtype different from float or doubles (word size = "
                            << word_size << ")."
            );
        }
    } else if (dtype == 'i') {
        if (word_size == sizeof(int32_t)) {
            return DTYPE_INT32;
        } else {
            ASSERT2(word_size == sizeof(double) || word_size == sizeof(float),
                utils::MS() << "can only load numpy arrays with dtype float or double (got word_size = "
                            << word_size << ").");
        }
    } else {
        ASSERT2(dtype == 'i' || dtype == 'f',
            utils::MS() << "attempted to load a npy array with dtype different from float or int (got dtype = "
                        << dtype << ").");
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
    return left.state == right.state;
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

bool Array::is_nan() const {
    return op::is_nan(*this);
}

bool Array::is_stateless() const {
    return state == nullptr;
}

bool Array::is_scalar() const {
    return ndim() == 0;
}

bool Array::is_vector() const {
    return ndim() == 1;
}

bool Array::is_matrix() const {
    return ndim() == 2;
}

Array Array::vectorlike_to_vector() const {
    int noe = number_of_elements();
    for (int dim: shape()) {
        ASSERT2(dim == 1 || dim == noe,
                utils::MS() << "Tensor with shape" << shape() << " cannot be interpreted as a vector");
    }
    return ravel();
}

bool Array::spans_entire_memory() const {
    ASSERT2(!is_stateless(), "spans_entire_memory has undefined meaning on a stateless Array.");
    int noe = number_of_elements();
    if (offset() == 0 && noe * size_of_dtype(dtype()) == memory()->total_memory) {
        return true;
    }
    if (offset() == noe - 1) {
        const auto& arr_strides = strides();
        const auto& arr_shape = shape();
        for (int i = 0; i < arr_strides.size(); i++) {
            if (std::abs(arr_strides[i]) == 1 && arr_shape[i] == noe) {
                return true;
            }
        }
    }
    return false;
}

bool Array::contiguous_memory() const {
    ASSERT2(!is_stateless(), "contiguous_memory must not be called with stateless Array.");

    if (strides().empty()) {
        // strides are trivial, stored as an empty vector
        // hence memory can be accessed contiguously
        return true;
    } else {
        // the strides may actually be set to non
        // trivial values, however if on the dimensions
        // where strides are set to a different value
        // than what shape_to_trivial_strides would dictate
        // (e.g. contiguous access, lowest stride as last dimension)
        // the dimension happens to be 1, the memory access
        // is still contiguous
        const auto& ns = strides();
        auto ts = shape_to_trivial_strides(shape());

        for (int i = 0; i < ndim(); ++i) {
            if (shape()[i] > 1 && ts[i] != ns[i]) {
                return false;
            }
        }
        return true;
    }
}

Array Array::ascontiguousarray() const {
    Array ret;
    if (contiguous_memory()) {
        ret = *this;
    } else {
        debug::array_as_contiguous.activate(*this);
        ret = op::identity(*this);
    }
    ret.state->strides.clear();
    return ret;
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

Array Array::operator[](const int& idx) const {
    return pluck_axis(0, idx);
}

ArraySubtensor Array::take_from_rows(const Array& indices) const {
    return ArraySubtensor(*this, indices);
}

AssignableArray Array::operator[](const Array& indices) const {
    return op::take(*this, indices);
}

SlicingInProgress<Array> Array::operator[](const Slice& s) const {
    auto ret = SlicingInProgress<Array>(*this);
    return ret[s];
}

SlicingInProgress<Array> Array::operator[](const Broadcast& b) const {
    auto ret = SlicingInProgress<Array>(*this);
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


bool Array::is_transpose() {
    if (ndim() <= 1) {
        // dims 0 and 1 do not change if we call a transpose
        return true;
    } else {
        if (strides().size() == 0) {
            // for dim greater than 2 if strides are trivial,
            // then we are definitely not transposed.
            return false;
        }
        // the condidtion for the transpose is that strides are in the order
        // opposite to the trivial strides.
        vector<int> reversed_shape(shape());
        std::reverse(reversed_shape.begin(), reversed_shape.end());
        auto reversed_strides = shape_to_trivial_strides(reversed_shape);

        for (int i = 0; i < ndim(); ++i) {
            if (reversed_strides[i] != strides()[ndim() - 1 - i]) {
                return false;
            }
        }
        return true;
    }
}

Array Array::transpose() const {
    std::vector<int> permutation(ndim(), 0);
    for (int i = 0; i < ndim(); ++i) {
        permutation[i] = ndim() - i - 1;
    }
    return transpose(permutation);
}

Array Array::transpose(const std::vector<int>& axes) const {
    return dimshuffle(axes);
}

Array Array::swapaxes(int axis1, int axis2) const {
    axis1 = normalize_axis(axis1);
    axis2 = normalize_axis(axis2);
    ASSERT2(0 <= axis1 && axis1 < ndim(),
        utils::MS() << "swapaxes axis1 (" << axis1 << ") must be less ndim (" << ndim() << ")");
    ASSERT2(0 <= axis2 && axis2 < ndim(),
        utils::MS() << "swapaxes axis2 (" << axis2 << ") must be less ndim (" << ndim() << ")");
    vector<int> axis_permuation;
    for (int i = 0; i < ndim(); ++i) {
        if (i == axis1) {
            axis_permuation.push_back(axis2);
        } else if (i == axis2) {
            axis_permuation.push_back(axis1);
        } else {
            axis_permuation.push_back(i);
        }
    }
    return dimshuffle(axis_permuation);
}

Array Array::dimshuffle(const std::vector<int>& pattern) const {
    int dimensionality = ndim();
    ASSERT2(pattern.size() == dimensionality,
        utils::MS() << "number of dimensions in dimshuffle does not correspond"
                    << " to the dimensionality of the array (got pattern = " << pattern
                    << " on array with dimensionality=" << dimensionality
    );
    std::vector<int> newstrides(dimensionality);
    std::vector<int> newshape(dimensionality);

    auto current_shape   = shape();
    auto current_strides = normalized_strides();

    for (int i = 0; i < dimensionality; i++) {
        const auto& pick_from = pattern[i];
        ASSERT2(current_shape[pick_from] != -1,
            utils::MS() << "duplicate dimension in dimshuffle pattern " << pattern
        );
        // grab strides and shape for the
        // relevant dimension
        newshape[i] = current_shape[pick_from];
        newstrides[i] = current_strides[pick_from];
        current_shape[pick_from] = -1;
    }

    return Array(newshape, memory(), offset(), newstrides, dtype());
}

Array Array::copyless_ravel() const {
    if (ndim() == 1) return *this;
    ASSERT2(contiguous_memory(),
            "at the moment ravel is only supported for contiguous_memory");
    return Array({number_of_elements()},
                 memory(),
                 offset(),
                 std::vector<int>(),
                 dtype());
}

Array Array::ravel() const {
    if (ndim() == 1) return *this;
    return ascontiguousarray().copyless_ravel();
}

Array Array::copyless_reshape(const vector<int>& new_shape) const {
    if (new_shape == shape()) return *this;
    ASSERT2(hypercube_volume(new_shape) == number_of_elements(),
            utils::MS() << "New shape (" << new_shape
                        << ") must have the same number of elements as previous shape ("
                        << shape() << ")");
    ASSERT2(contiguous_memory(),
            utils::MS() << "Cannot perform reshape without a copy on non-contiguous memory "
                        << "(strides() = " << strides() << ", shape=" << shape()
                        << ", new shape=" << new_shape << ").");

    return Array(new_shape,
                 memory(),
                 offset(),
                 vector<int>(),
                 dtype());
}


Array Array::reshape(const vector<int>& new_shape) const {
    if (new_shape == shape()) return *this;
    return ascontiguousarray().copyless_reshape(new_shape);
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


Array Array::pluck_axis(const int& axis, const int& pluck_idx) const {
    auto single_item_slice = pluck_axis(axis, Slice(pluck_idx, pluck_idx + 1));
    return single_item_slice.squeeze(axis);
}

Array Array::pluck_axis(const int& axis, const Slice& slice_unnormalized) const {
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
Array Array::squeeze(const int& axis) const {
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

Array Array::expand_dims(const int& new_axis) const {
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


Array Array::broadcast_axis(const int& axis) const {
    Array out(*this, false);
    out.broadcast_axis_internal(axis);
    return out;
}

Array Array::insert_broadcast_axis(const int& new_axis) const {
    return expand_dims(new_axis).broadcast_axis(new_axis);
}

int Array::normalize_axis(const int& axis) const {
    if (axis < 0) {
        return ndim() + axis;
    } else {
        return axis;
    }
}


Array Array::broadcast_scalar_to_ndim(const int& target_ndim) const {
    ASSERT2(is_scalar(),
            utils::MS() << "broadcast_scalar_to_ndim may only be called on scalars, got shape " << shape() << ".");
    Array res = *this;
    for (int i = 0; i < target_ndim; ++i) {
        res = res.insert_broadcast_axis(0);
    }
    return res;
}

#define DALI_ARRAY_DEFINE_ALL_REDUCER(FUNCTION_NAME, OPNAME)\
    AssignableArray Array::FUNCTION_NAME() const {\
        return op::OPNAME(*this);\
    }\

#define DALI_ARRAY_DEFINE_AXIS_REDUCER(FUNCTION_NAME, OPNAME)\
    AssignableArray Array::FUNCTION_NAME(const int& axis) const {\
        return op::OPNAME(*this, axis);\
    }\

#define DALI_ARRAY_DEFINE_REDUCER(FUNCTION_NAME, OPNAME)\
    DALI_ARRAY_DEFINE_ALL_REDUCER(FUNCTION_NAME, OPNAME);\
    DALI_ARRAY_DEFINE_AXIS_REDUCER(FUNCTION_NAME, OPNAME);\

DALI_ARRAY_DEFINE_REDUCER(sum, sum);
DALI_ARRAY_DEFINE_REDUCER(L2_norm, L2_norm);
DALI_ARRAY_DEFINE_REDUCER(mean, mean);
DALI_ARRAY_DEFINE_REDUCER(max, max);
DALI_ARRAY_DEFINE_REDUCER(min, min);

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

#define DALI_DEFINE_ARRAY_INTERACTION_INPLACE(OPNAME, SYMBOL)\
    Array& Array::operator SYMBOL (const Array& right) {\
        return *this = OPNAME (*this, right);\
    }

#define DALI_DEFINE_SCALAR_INTERACTION_INPLACE(OPNAME, SYMBOL)\
    Array& Array::operator SYMBOL (const double& right) {\
        return *this = OPNAME (*this, right);\
    }\
    Array& Array::operator SYMBOL (const float& right) {\
        return *this = OPNAME (*this, right);\
    }\
    Array& Array::operator SYMBOL (const int& right) {\
        return *this = OPNAME (*this, right);\
    }

DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::add, +=);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::sub, -=);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::eltmul, *=);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::eltdiv, /=);

Array& Array::operator<<=(const Array& right) {
    *this <<= op::identity(right);
    return *this;
}

DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_sub, -=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_add, +=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_mul, *=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_div, /=);


void Array::print(std::basic_ostream<char>& stream, const int& indent, const bool& add_newlines) const {
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

void Array::debug_memory(const bool& print_contents) const {
    memory()->debug_info(std::cout, print_contents, dtype());
}

void Array::clear() {
    if (spans_entire_memory()) {
        memory()->lazy_clear();
    } else {
        *this = initializer::fill(0.0);
    }
}

AssignableArray Array::dot(const Array& other) const {
    return op::dot(*this, other);
}

bool operator==(const Array& left, const Array& right) {
    return Array::state_equals(left, right);
}

ArraySubtensor::ArraySubtensor(const Array& source_, const Array& indices_) : indices(indices_), source(source_) {}

DType ArraySubtensor::dtype() const {
    return source.dtype();
}

const std::vector<int>& ArraySubtensor::shape() const {
    return source.shape();
}

ArraySubtensor& ArraySubtensor::operator=(const Array& assignable) {
    op::assign_to_rows<OPERATOR_T_EQL>(*this, assignable);
    return *this;
}

ArraySubtensor& ArraySubtensor::operator=(const AssignableArray& assignable) {
    // TODO(jonathan, szymon): make more efficient
    Array self_as_array(*this);
    assignable.assign_to(self_as_array, OPERATOR_T_EQL);
    return (*this = self_as_array);
}

void ArraySubtensor::print(std::basic_ostream<char>& stream,
                           const int& indent,
                           const bool& add_newlines) const {
    ((Array)op::take_from_rows(source, indices)).print(
        stream,
        indent,
        add_newlines
    );
}

