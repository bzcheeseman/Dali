#include "buffer.h"

#include <algorithm>

#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/shape.h"
#include "dali/array/array.h"
#include "dali/array/op/unary.h"
#include "dali/array/expression/assignment.h"

namespace {
    // shape makes sense
    bool shape_strictly_positive(const std::vector<int>& shape) {
        return std::all_of(shape.begin(), shape.end(), [](int x) {
            return x > 0;
        });
    }
}

std::shared_ptr<memory::SynchronizedMemory> Buffer::create_memory(
        const std::vector<int>& shape,
        DType dtype,
        memory::Device preferred_device) {
    int number_of_elements = hypercube_volume(shape);

    return std::make_shared<memory::SynchronizedMemory>(
        number_of_elements * size_of_dtype(dtype),
        (shape.size() > 0) ? shape[shape.size()-1] : 1,
        preferred_device
    );
}

Buffer::Buffer(std::shared_ptr<memory::SynchronizedMemory> memory,
                       const std::vector<int>& shape,
                       DType dtype,
                       int offset,
                       const std::vector<int>& strides) :
        Expression(shape, dtype, {}, offset, strides),
        memory_(memory){
    ASSERT2(shape_strictly_positive(shape), utils::make_message("Shape "
        "elements must be strictly positive (got ", shape, ")."));
}

void Buffer::broadcast_axis_internal(const int& axis) {
    ASSERT2(0 <= axis && axis < ndim(), utils::make_message("broadcast dimension (",
        axis, ") must be lower than the dimensionality of the broadcasted tensor (",
        ndim(), ")."));

    std::vector<int> new_strides = normalized_strides();
    new_strides[axis] = 0;

    strides_ = new_strides;
}

Buffer::Buffer(const std::vector<int>& shape,
                       DType dtype,
                       memory::Device preferred_device,
                       int offset,
                       const std::vector<int>& strides) :
        Buffer(create_memory(shape, dtype, preferred_device),
                   shape, dtype, offset, strides) {

}

Buffer::Buffer(const Buffer& other) :
        Expression(other),
        memory_(other.memory_) {
}

expression_ptr Buffer::copy() const {
    return std::make_shared<Buffer>(*this);
}

expression_ptr Buffer::buffer_arg() const {
    return copy();
}

memory::Device Buffer::preferred_device() const {
    return memory_->preferred_device;
}

bool Buffer::spans_entire_memory() const {
    int noe = number_of_elements();
    if (offset_ == 0 && noe * size_of_dtype(dtype_) == memory_->total_memory) {
        return true;
    }
    if (offset_ == noe - 1) {
        const auto& arr_strides = strides_;
        const auto& arr_shape = shape_;
        for (int i = 0; i < arr_strides.size(); i++) {
            if (std::abs(arr_strides[i]) == 1 && arr_shape[i] == noe) {
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<Buffer> Buffer::create_with_shape(
        const std::vector<int>& shape,
        DType dtype,
        memory::Device preferred_device,
        const std::vector<int>& broadcasted_axes) {
    auto ret = std::make_shared<Buffer>(shape, dtype, preferred_device);
    for (const auto& axis : broadcasted_axes) {
        ret->broadcast_axis_internal(axis);
    }
    return ret;
}

expression_ptr Buffer::dimshuffle(const std::vector<int>& pattern, const Array* owner) const {
    int dimensionality = ndim();
    ASSERT2(pattern.size() == dimensionality, utils::make_message("number of"
        " dimensions in dimshuffle does not correspond to the dimensionality "
        "of the array (got pattern = ", pattern, " on array with ndim = ",
        dimensionality, ")."));
    std::vector<int> newstrides(dimensionality);
    std::vector<int> newshape(dimensionality);

    auto current_shape   = shape_;
    auto current_strides = normalized_strides();

    for (int i = 0; i < dimensionality; i++) {
        int pick_from = pattern[i];
        if (pick_from < 0) {
            // allow negative transpose values
            // (e.g. to swap first and last dimensions use {-1, 0})
            pick_from = pick_from + current_shape.size();
        }
        ASSERT2(0 <= pick_from && pick_from < current_shape.size(), utils::make_message(
            "tranpose axis must be positive and less than the dimensionality "
            "of the array (got ", pattern[i], " and ndim = ", current_shape.size(), ")."));
        ASSERT2(current_shape[pick_from] != -1, utils::make_message("duplicate"
            " dimension in dimshuffle pattern ", pattern, "."));
        // grab strides and shape for the
        // relevant dimension
        newshape[i] = current_shape[pick_from];
        newstrides[i] = current_strides[pick_from];
        current_shape[pick_from] = -1;
    }
    return copy(newshape, offset_, newstrides);
}

expression_ptr Buffer::_reshape(const std::vector<int>& new_shape, const Array* owner) const {
    if (contiguous_memory()) {
        return copy(new_shape, offset_, {});
    }
    if (new_shape.size() > ndim()) {
        // check if the lowest dimensions will be identical
        bool matching_lowest = true;
        for (int i = 0; i < ndim(); i++) {
            if (new_shape[new_shape.size() - i - 1] != shape_[ndim() - i - 1]) {
                matching_lowest = false;
            }
        }
        bool is_ones_elsewhere = true;
        for (int i = 0; i < new_shape.size() - ndim(); i++) {
            if (new_shape[i] != 1) {
                is_ones_elsewhere = false;
                break;
            }
        }
        if (matching_lowest && is_ones_elsewhere) {
            auto new_strides = strides_;
            int top_most_stride = new_strides.size() > 0 ? new_strides.front() : 1;
            for (int i = 0; i < new_shape.size() - ndim(); i++) {
                new_strides.insert(new_strides.begin(), top_most_stride);
            }
            return copy(new_shape, offset_, new_strides);
        }
    }
    return op::to_assignment(copy()).reshape(new_shape).expression();
}

expression_ptr Buffer::pluck_axis(int axis, const Slice& slice_unnormalized, const Array* owner) const {
    axis = normalize_axis(axis);
    ASSERT2(axis >= 0 && axis < shape_.size(), utils::make_message("pluck_axis"
        " dimension (", axis, ") must be positive and less the dimensionality "
        "of the plucked array (", ndim(), ")."));
    Slice slice;
    if (strides_.size() > 0 && strides_[axis] == 0 &&
        slice_unnormalized.end &&
        (slice_unnormalized.end.value() - slice_unnormalized.start > 0)) {
        slice = Slice::normalize_and_check(Slice(0, 1), shape_[axis]);
    } else {
        slice = Slice::normalize_and_check(slice_unnormalized, shape_[axis]);
    }

    const std::vector<int>& old_shape = shape_;
    auto old_strides             = normalized_strides();

    std::vector<int> new_shape(old_shape);
    std::vector<int> new_strides(old_strides);

    new_shape[axis]    = slice.size();
    new_strides[axis] *= slice.step;

    int new_offset;
    if (slice.step > 0) {
        new_offset = offset_ + old_strides[axis] * slice.start;
    } else {
        new_offset = offset_ + old_strides[axis] * (slice.end.value() - 1);
    }

    return copy(new_shape, new_offset, new_strides);
}


expression_ptr Buffer::_squeeze(int axis, const Array* owner) const {
    const std::vector<int>& old_shape = shape_;
    auto old_strides             = normalized_strides();
    std::vector<int> new_shape;
    std::vector<int> new_strides;
    for (int i = 0; i < old_shape.size(); ++i) {
        if (i == axis) {
            continue;
        }
        new_shape.push_back(old_shape[i]);
        new_strides.push_back(old_strides[i]);
    }
    return copy(new_shape, offset_, new_strides);
}

expression_ptr Buffer::_expand_dims(int new_axis, const Array* owner) const {
    std::vector<int> new_shape   = shape_;
    std::vector<int> new_strides = normalized_strides();

    new_shape.insert(  new_shape.begin() + new_axis, 1);
    // It really does not matter what the new stride is going to be,
    // because in we are only ever going to access it at index 0,
    // so it will get cancelled out. We chose to set it to the stride that
    // naturally arises in the shape_to_trivial_strides, such that if other strides
    // were already trivial it will get compacted.
    new_strides.insert(
        new_strides.begin() + new_axis,
        shape_to_trivial_strides(new_shape)[new_axis]
    );
    return copy(new_shape, offset_, new_strides);
}

expression_ptr Buffer::broadcast_to_shape(const std::vector<int>& new_shape, const Array* owner) const {
    ASSERT2(new_shape.size() == shape_.size(), utils::make_message(
        "broadcast_to_shape's new_shape argument must be of the "
        "same dimensionality as the current dimensionality of the "
        "expression (got new_shape = ", new_shape, ", current shape = ", shape_, ")."));
    std::vector<int> broadcasted_axes;
    for (size_t i = 0; i < new_shape.size(); i++) {
        if (shape_[i] != new_shape[i]) {
            ASSERT2(shape_[i] == 1, utils::make_message(
                "broadcast_to_shape cannot broadcast a dimension ", i, " of size ", shape_[i],
                " to a size of ", new_shape[i], " (shape = ", shape_,
                ", new_shape = ", new_shape, ")."));
            broadcasted_axes.emplace_back(i);
        }
    }
    if (broadcasted_axes.size() > 0) {
        std::vector<int> new_strides = normalized_strides();
        for (auto& axis : broadcasted_axes) {
            new_strides[axis] = 0;
        }
        return copy(shape_, offset_, new_strides);
    } else {
        return copy();
    }
}

expression_ptr Buffer::operator()(int idx, const Array* owner) const {
    ASSERT2(0 <= idx && idx < number_of_elements(), utils::make_message(
        "Index ", idx, " must be in [0,", number_of_elements(), ")."));
    int delta_offset;
    if (contiguous_memory()) {
        delta_offset = idx;
    } else {
        std::vector<int> ns = normalized_strides();
        delta_offset = 0;
        for (int dim = ndim() - 1; dim >= 0; --dim) {
            int index_at_dim  = idx % shape_[dim];
            idx /= shape_[dim];
            int stride_at_dim = ns[dim];
            delta_offset += index_at_dim * stride_at_dim;
        }
    }
    return copy({}, offset_ + delta_offset, {});
}

expression_ptr Buffer::copy(const std::vector<int>& shape,
                                int offset,
                                const std::vector<int>& strides) const {
    auto ret      = copy();
    ret->shape_   = shape;
    ret->offset_  = offset;
    ret->strides_ = strides;
    ret->compact_strides();
    return ret;
}

bool Buffer::supports_operator(OPERATOR_T operator_t) const {
    return true;
}

bool Buffer::is_assignable() const {
    return true;
}

bool Buffer::is_buffer() const {
    return true;
}

bool Buffer::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return contiguous_memory();
}

namespace op {
Buffer* static_as_buffer(const Array& arr) {
    return static_cast<Buffer*>(arr.expression().get());
}
}
