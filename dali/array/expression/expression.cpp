#include "expression.h"

#include <algorithm>

// demangle names
#include <cxxabi.h>

#include "dali/array/array.h"
#include "dali/array/shape.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/op/unary.h"

////////////////////////////////////////////////////////////////////////////////
//               MISCELANEOUS UTILITIES (NOT EXPOSED)                         //
////////////////////////////////////////////////////////////////////////////////

namespace {
    using std::vector;

    // if strides are trivial (such that they would arrise from shape normally)
    // we remove them.
    void compact_strides(const vector<int>& shape, vector<int>* strides_ptr) {
        auto& strides = *strides_ptr;
        if (strides.size() == 0)
            return;
        ASSERT2(strides.size() == shape.size(),
                "Not the same number of strides as dimensions.");
        if (shape_to_trivial_strides(shape) == strides) {
            strides.clear();
        }
    }

    std::vector<int> normalize_shape(const std::vector<int>& current_shape, std::vector<int> new_shape) {
        int undefined_dim = -1;
        int known_shape_volume = 1;
        for (int i = 0; i < new_shape.size(); i++) {
            if (new_shape[i] < 0) {
                ASSERT2(undefined_dim == -1, utils::make_message("new shape can "
                    "only specify one unknown dimension (got ", new_shape, ")."));
                undefined_dim = i;
            } else {
                known_shape_volume *= new_shape[i];
            }
        }
        if (undefined_dim != -1) {
            if (known_shape_volume == 0) {
                return new_shape;
            } else {
                int current_volume = hypercube_volume(current_shape);
                ASSERT2(current_volume % known_shape_volume == 0, utils::make_message(
                    "cannot deduce unknown dimension (", new_shape, ") with current "
                    "shape (", current_shape, ")."));
                new_shape[undefined_dim] = current_volume / known_shape_volume;
            }
        }
        return new_shape;
    }
}

////////////////////////////////////////////////////////////////////////////////
//                           EXPRESSION                                       //
////////////////////////////////////////////////////////////////////////////////

Expression::Expression(const std::vector<int>& shape,
                       DType dtype,
                       int offset,
                       const std::vector<int>& strides) :
        shape_(shape),
        dtype_(dtype),
        offset_(offset),
        strides_(strides) {
    compact_strides(shape_, &strides_);
    ASSERT2(strides_.size() == 0 || strides_.size() == shape_.size(), "stride "
        "and shape size must be the same (unless strides are compacted).");
}

Expression::Expression(const Expression& other) :
        shape_(other.shape_),
        dtype_(other.dtype_),
        offset_(other.offset_),
        strides_(other.strides_) {
}

expression_ptr Expression::copy(const std::vector<int>& shape,
                                int offset,
                                const std::vector<int>& strides) const {
    auto ret      = copy();
    ret->shape_   = shape;
    ret->offset_  = offset;
    ret->strides_ = strides;
    compact_strides(ret->shape_, &ret->strides_);
    return ret;
}


int Expression::number_of_elements() const {
    return hypercube_volume(shape_);
}

int Expression::ndim() const {
    return shape_.size();
}

std::vector<int> Expression::normalized_strides() const {
    return (strides_.size() > 0) ? strides_ : shape_to_trivial_strides(shape_);
}

std::vector<int> Expression::bshape() const {
    if (strides_.size() == 0) {
        return shape_;
    } else {
        vector<int> result(shape_);
        for (int i=0; i < ndim(); ++i) {
            if (strides_[i] == 0) {
                // broadcasted dimensions are negated.
                result[i] = -abs(result[i]);
            }
        }
        return result;
    }
}

bool Expression::is_scalar() const {
    return ndim() == 0;
}

bool Expression::is_vector() const {
    return ndim() == 1;
}

bool Expression::is_matrix() const {
    return ndim() == 2;
}


bool Expression::contiguous_memory() const {
    if (strides_.empty()) {
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
        const auto& ns = strides_;
        auto ts = shape_to_trivial_strides(shape_);

        for (int i = 0; i < ndim(); ++i) {
            if (shape_[i] > 1 && ts[i] != ns[i]) {
                return false;
            }
        }
        return true;
    }
}

inline int Expression::normalize_axis(const int& axis) const {
    if (axis < 0) {
        return ndim() + axis;
    } else {
        return axis;
    }
}

void Expression::broadcast_axis_internal(const int& axis) {
    ASSERT2(0 <= axis && axis < ndim(), utils::make_message("broadcast dimension (",
        axis, ") must be lower than the dimensionality of the broadcasted tensor (",
        ndim(), ")."));

    vector<int> new_strides = normalized_strides();
    new_strides[axis] = 0;

    strides_ = new_strides;
}


expression_ptr Expression::operator()(int idx) const {
    ASSERT2(0 <= idx && idx < number_of_elements(), utils::make_message(
        "Index ", idx, " must be in [0,", number_of_elements(), ")."));

    int delta_offset;
    if (contiguous_memory()) {
        delta_offset = idx;
    } else {
        vector<int> ns = normalized_strides();
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

bool Expression::is_transpose() const {
    if (ndim() <= 1) {
        // dims 0 and 1 do not change if we call a transpose
        return true;
    } else {
        if (strides_.size() == 0) {
            // for dim greater than 2 if strides are trivial,
            // then we are definitely not transposed.
            return false;
        }
        // the condition for the transpose is that strides are in the order
        // opposite to the trivial strides.
        vector<int> reversed_shape(shape_);
        std::reverse(reversed_shape.begin(), reversed_shape.end());
        auto reversed_strides = shape_to_trivial_strides(reversed_shape);

        for (int i = 0; i < ndim(); ++i) {
            if (reversed_strides[i] != strides_[ndim() - 1 - i]) {
                return false;
            }
        }
        return true;
    }
}

expression_ptr Expression::transpose() const {
    std::vector<int> permutation(ndim(), 0);
    for (int i = 0; i < ndim(); ++i) {
        permutation[i] = ndim() - i - 1;
    }
    return transpose(permutation);
}

expression_ptr Expression::transpose(const std::vector<int>& axes) const {
    return dimshuffle(axes);
}

expression_ptr Expression::swapaxes(int axis1, int axis2) const {
    axis1 = normalize_axis(axis1);
    axis2 = normalize_axis(axis2);
    // no-op
    if (axis1 == axis2) return copy();

    ASSERT2(0 <= axis1 && axis1 < ndim(), utils::make_message("swapaxes"
        " axis1 (", axis1, ") must be less than ndim (", ndim(), ")."));
    ASSERT2(0 <= axis2 && axis2 < ndim(), utils::make_message("swapaxes"
        " axis2 (", axis2, ") must be less than ndim (", ndim(), ")."));

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

expression_ptr Expression::dimshuffle(const std::vector<int>& pattern) const {
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
            // (e.g. to swap first and last dimensions use {0, -1})
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

expression_ptr Expression::copyless_ravel() const {
    if (ndim() == 1) return copy();
    return copyless_reshape({-1});
}

expression_ptr Expression::ravel() const {
    if (ndim() == 1) return copy();
    return reshape({-1});
}

bool Expression::can_copyless_reshape(const vector<int>& new_shape) const {
    auto norm_shape = normalize_shape(shape_, new_shape);
    if (norm_shape == shape_ || contiguous_memory()) return true;
    if (hypercube_volume(norm_shape) != number_of_elements()) return false;
    if (norm_shape.size() > ndim()) {
        // check if the lowest dimensions will be identical
        bool matching_lowest = true;
        for (int i = 0; i < ndim(); i++) {
            if (norm_shape[norm_shape.size() - i - 1] != shape_[ndim() - i - 1]) {
                matching_lowest = false;
            }
        }
        bool is_ones_elsewhere = true;
        for (int i = 0; i < new_shape.size() - ndim(); i++) {
            if (norm_shape[i] != 1) {
                is_ones_elsewhere = false;
                break;
            }
        }
        if (matching_lowest && is_ones_elsewhere) {
            return true;
        }
    }
    return false;
}

expression_ptr Expression::copyless_reshape(const vector<int>& new_shape) const {
    auto norm_shape = normalize_shape(shape_, new_shape);
    if (norm_shape == shape_) return copy();
    ASSERT2(hypercube_volume(norm_shape) == number_of_elements(), utils::make_message(
        "New shape (", new_shape, ") must have the same number of elements as previous "
        "shape (", shape_, ")"));

    if (contiguous_memory()) {
        return copy(norm_shape, offset_, {});
    }
    if (norm_shape.size() > ndim()) {
        // check if the lowest dimensions will be identical
        bool matching_lowest = true;
        for (int i = 0; i < ndim(); i++) {
            if (norm_shape[norm_shape.size() - i - 1] != shape_[ndim() - i - 1]) {
                matching_lowest = false;
            }
        }
        bool is_ones_elsewhere = true;
        for (int i = 0; i < new_shape.size() - ndim(); i++) {
            if (norm_shape[i] != 1) {
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
            return copy(norm_shape, offset_, new_strides);
        }
    }
    ASSERT2(false, utils::make_message("Cannot perform reshape without a copy on "
        "non-contiguous memory (contiguous = ", contiguous_memory(),
        ", strides = ", strides_, ", shape=", shape_, ","
        " new shape = ", new_shape, ")."));
    return nullptr;
}

expression_ptr Expression::right_fit_ndim(int target_ndim) const {
    if (ndim() == target_ndim) return copy();
    if (ndim() > target_ndim) {
        std::vector<int> new_shape = shape_;
        // remove dimensions that will be collapsed:
        new_shape.erase(new_shape.begin(), new_shape.begin() + (ndim() - target_ndim));
        if (target_ndim > 0) {
            new_shape[0] = -1;
        }
        return reshape(new_shape);
    } else {
        std::vector<int> new_shape = shape_;
        // extend shape with ones:
        new_shape.insert(new_shape.begin(), target_ndim - ndim(), 1);
        return reshape(new_shape);
    }
}

expression_ptr Expression::copyless_right_fit_ndim(int target_ndim) const {
    if (ndim() == target_ndim) return copy();
    if (ndim() > target_ndim) {
        std::vector<int> new_shape = shape_;
        // remove dimensions that will be collapsed:
        new_shape.erase(new_shape.begin(), new_shape.begin() + (ndim() - target_ndim));
        if (target_ndim > 0) {
            new_shape[0] = -1;
        }
        return copyless_reshape(new_shape);
    } else {
        std::vector<int> new_shape = shape_;
        // extend shape with ones:
        new_shape.insert(new_shape.begin(), target_ndim - ndim(), 1);
        return copyless_reshape(new_shape);
    }
}

expression_ptr Expression::reshape(const vector<int>& new_shape) const {
    if (can_copyless_reshape(new_shape)) return copyless_reshape(new_shape);
    auto ret = op::identity(Array(copy()));
    return ret.expression()->copyless_reshape(new_shape);
}

expression_ptr Expression::reshape_broadcasted(const std::vector<int>& new_shape) const {
    ASSERT2(new_shape.size() == ndim(), utils::make_message("reshape_"
        "broadcasted must receive a shape with the same dimensionality ("
        "current shape = ", shape_, ", new shape = ", new_shape, " in expression ",
        name(), ")."));
    auto my_bshape = bshape();

    for (int i = 0; i < my_bshape.size(); ++i) {
        ASSERT2(new_shape[i] > 0, utils::make_message("reshape_broadcasted's "
            "new_shape argument must be strictly positive (got ", new_shape, ")."));

        ASSERT2(new_shape[i] == std::abs(my_bshape[i]) || my_bshape[i] == -1,
                utils::make_message("reshape_broadcasted can only reshape "
                "broadcasted dimensions (tried to reshape array with shape "
                "= ", my_bshape, " to new shape = ", new_shape, ")."));
    }
    return copy(new_shape, offset_, strides_);
}

expression_ptr Expression::pluck_axis(const int& axis, const int& pluck_idx) const {
    auto single_item_slice = pluck_axis(axis, Slice(pluck_idx, pluck_idx + 1));
    return single_item_slice->squeeze(axis);
}

expression_ptr Expression::pluck_axis(int axis, const Slice& slice_unnormalized) const {
    axis = normalize_axis(axis);
    ASSERT2(axis >= 0 && axis < shape_.size(), utils::make_message("pluck_axis"
        " dimension (", axis, ") must be positive and less the dimensionality "
        "of the plucked array (", ndim(), ")."));

    Slice slice = Slice::normalize_and_check(slice_unnormalized, shape_[axis]);

    const vector<int>& old_shape = shape_;
    auto old_strides             = normalized_strides();

    vector<int> new_shape(old_shape);
    vector<int> new_strides(old_strides);

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

expression_ptr Expression::squeeze(int axis) const {
    axis = normalize_axis(axis);
    ASSERT2(0 <= axis && axis < shape_.size(), utils::make_message("squeeze "
        "dimension (", axis, ") must be less the dimensionality of compacted "
        "tensor (", ndim(), ")."));
    ASSERT2(shape_[axis] == 1, utils::make_message("cannot select an axis to squeeze "
        "out which has size not equal to one (got axis = ", axis, ", shape[",
        axis, "] = ", shape_[axis], ")."));

    const vector<int>& old_shape = shape_;
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

    return copy(new_shape, offset_, new_strides);
}

expression_ptr Expression::expand_dims(int new_axis) const {
    new_axis = normalize_axis(new_axis);
    ASSERT2(new_axis >= 0 && new_axis <= ndim(), utils::make_message("expand_dims "
        "new_axis argument must be strictly positive and at most the dimensionality"
        " of the array (got new_axis = ", new_axis, ", ndim = ", ndim(), ")."));
    vector<int> new_shape   = shape_;
    vector<int> new_strides = normalized_strides();

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

expression_ptr Expression::broadcast_axis(int axis) const {
    auto out = copy();
    axis = normalize_axis(axis);
    ASSERT2(axis >= 0 && axis < ndim(), utils::make_message("broadcast_axis "
        "axis must be positive and less than the dimensionality of the array "
        "(got axis = ", axis, ", ndim = ", ndim(), ")."));
    ASSERT2(shape_[axis] == 1, utils::make_message("axis to be broadcasted "
        "must have dimension 1 (got shape[", axis, "] = ", shape_[axis], ")."));
    out->broadcast_axis_internal(axis);
    return out;
}

expression_ptr Expression::insert_broadcast_axis(int new_axis) const {
    new_axis = normalize_axis(new_axis);
    return expand_dims(new_axis)->broadcast_axis(new_axis);
}

expression_ptr Expression::collapse_axis_with_axis_minus_one(int axis) const {
    axis = normalize_axis(axis);
    ASSERT2(axis >= 1 && axis < ndim(), utils::make_message("collapse_axis_with_axis_minus_one "
        "axis must >= 1 and less than the dimensionality of the array "
        "(got axis = ", axis, ", ndim = ", ndim(), ")."));
    std::vector<int> newshape = bshape();
    newshape[axis - 1] = newshape[axis] * newshape[axis - 1];
    newshape.erase(newshape.begin() + axis);
    return copyless_reshape(newshape);
}

expression_ptr Expression::broadcast_scalar_to_ndim(const int& target_ndim) const {
    ASSERT2(target_ndim >= 0, utils::make_message("broadcast_scalar_to_ndim"
        " expected a non-negative integer (got ", target_ndim, ")."));
    ASSERT2(is_scalar(), utils::make_message("broadcast_scalar_to_ndim may "
        "only be called on scalars, current shape = ", shape_, "."));
    auto res = copy();
    for (int i = 0; i < target_ndim; ++i) {
        res = res->insert_broadcast_axis(0);
    }
    return res;
}

std::string Expression::name() const {
    auto hasname = typeid(*this).name();
    int status;
    char * demangled = abi::__cxa_demangle(hasname, 0, 0, &status);
    return std::string(demangled);
}

std::string Expression::full_name() const {
    std::stringstream ss;
    ss << name();
    auto args = arguments();
    if (args.size() > 0) {
        ss << "(";
        for (size_t i = 0; i < args.size(); i++) {
            ss << args[i].expression()->full_name();
            if (i + 1 != args.size()) {
                ss << ", ";
            }
        }
        ss << ")";
    }
    return ss.str();
}

void Expression::for_all_suboperations(std::function<void(const Array&)> callback) const {
    for (auto arg : arguments()) {
        callback(arg);
        arg.expression()->for_all_suboperations(callback);
    }
}

bool Expression::supports_operator(OPERATOR_T operator_t) const {
    return operator_t == OPERATOR_T_EQL;
}

bool Expression::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return false;
}
