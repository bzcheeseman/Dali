#include "expression.h"

#include <algorithm>

// demangle names
#include <cxxabi.h>

#include "dali/array/array.h"
#include "dali/array/shape.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/expression/control_flow.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/op/unary.h"

////////////////////////////////////////////////////////////////////////////////
//                           EXPRESSION                                       //
////////////////////////////////////////////////////////////////////////////////

#define CONNECT_AUTO_ASSIGN(NAME)\
    Array assignment = op::to_assignment(copy());\
    if (owner != nullptr) {\
        owner->set_expression(assignment.expression());\
        return op::control_dependency(\
            *owner, (*owner).buffer_arg().NAME).expression();\
    }\
    return op::control_dependency(\
        assignment, assignment.buffer_arg().NAME).expression();\

Expression::Expression(const std::vector<int>& shape,
                       DType dtype,
                       const std::vector<Array>& arguments,
                       int offset,
                       const std::vector<int>& strides) :
        shape_(shape),
        dtype_(dtype),
        arguments_(arguments),
        offset_(offset),
        strides_(strides) {
    compact_strides();
    ASSERT2(strides_.size() == 0 || strides_.size() == shape_.size(), "stride "
        "and shape size must be the same (unless strides are compacted).");
}

Expression::Expression(const Expression& other) :
        shape_(other.shape_),
        dtype_(other.dtype_),
        offset_(other.offset_),
        strides_(other.strides_),
        arguments_(other.arguments_) {
}

void Expression::compact_strides() {
    if (strides_.size() == 0)
        return;
    ASSERT2(strides_.size() == shape_.size(),
            "Not the same number of strides as dimensions.");
    if (shape_to_trivial_strides(shape_) == strides_) {
        strides_.clear();
    }
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

bool Expression::is_scalar() const {
    return ndim() == 0;
}

bool Expression::is_vector() const {
    return ndim() == 1;
}

bool Expression::is_matrix() const {
    return ndim() == 2;
}

bool Expression::is_assignable() const {
    return false;
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
        std::vector<int> reversed_shape(shape_);
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


std::string Expression::name() const {
    auto mangled_name = typeid(*this).name();
    int status;
    char * demangled = abi::__cxa_demangle(mangled_name, 0, 0, &status);
    return std::string(demangled);
}

const std::vector<Array>& Expression::arguments() const {
    return arguments_;
}

std::string Expression::pretty_print_full_name(Expression* highlight, int indent) const {

    std::stringstream ss;
    if (this == highlight) {
        ss << utils::red;
    }
    ss << std::string(indent, ' ') << name();
    auto args = arguments();
    if (args.size() > 0) {
        ss << "(\n";
        for (size_t i = 0; i < args.size(); i++) {
            ss << args[i].expression()->pretty_print_full_name(highlight, indent + 4);
            if (i + 1 != args.size()) {
                ss << ",\n";
            } else {
                ss << "\n";
            }
        }
        ss << std::string(indent, ' ') << ")";
    }
    if (this == highlight) {
        ss << utils::reset_color;
    }
    return ss.str();
}

std::string Expression::full_name() const {
    std::stringstream ss;
    ss << name();
    const auto& args = arguments();
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

bool Expression::spans_entire_memory() const {
    if (offset_ == 0 && strides_.size() == 0) {
        return true;
    }
    int noe = number_of_elements();
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

expression_ptr Expression::buffer_arg() const {
    return nullptr;
}

expression_ptr Expression::transpose(const Array* owner) const {
    std::vector<int> permutation(ndim(), 0);
    for (int i = 0; i < ndim(); ++i) {
        permutation[i] = ndim() - i - 1;
    }
    return transpose(permutation, owner);
}

expression_ptr Expression::transpose(const std::vector<int>& axes, const Array* owner) const {
    return dimshuffle(axes, owner);
}

expression_ptr Expression::swapaxes(int axis1, int axis2, const Array* owner) const {
    axis1 = normalize_axis(axis1);
    axis2 = normalize_axis(axis2);
    // no-op
    if (axis1 == axis2) return copy();

    ASSERT2(0 <= axis1 && axis1 < ndim(), utils::make_message("swapaxes"
        " axis1 (", axis1, ") must be less than ndim (", ndim(), ")."));
    ASSERT2(0 <= axis2 && axis2 < ndim(), utils::make_message("swapaxes"
        " axis2 (", axis2, ") must be less than ndim (", ndim(), ")."));

    std::vector<int> axis_permutation;
    for (int i = 0; i < ndim(); ++i) {
        if (i == axis1) {
            axis_permutation.push_back(axis2);
        } else if (i == axis2) {
            axis_permutation.push_back(axis1);
        } else {
            axis_permutation.push_back(i);
        }
    }
    return dimshuffle(axis_permutation, owner);
}

expression_ptr Expression::ravel(const Array* owner) const {
    if (ndim() == 1) return copy();
    return reshape({-1}, owner);
}

expression_ptr Expression::right_fit_ndim(int target_ndim, const Array* owner) const {
    if (ndim() == target_ndim) return copy();
    if (ndim() > target_ndim) {
        std::vector<int> new_shape = shape_;
        // remove dimensions that will be collapsed:
        new_shape.erase(new_shape.begin(), new_shape.begin() + (ndim() - target_ndim));
        if (target_ndim > 0) {
            new_shape[0] = -1;
        }
        return reshape(new_shape, owner);
    } else {
        std::vector<int> new_shape = shape_;
        // extend shape with ones:
        new_shape.insert(new_shape.begin(), target_ndim - ndim(), 1);
        return reshape(new_shape, owner);
    }
}

expression_ptr Expression::insert_broadcast_axis(int new_axis, const Array* owner) const {
    new_axis = normalize_axis(new_axis);
    return expand_dims(new_axis, owner)->broadcast_axis(new_axis, nullptr);
}

expression_ptr Expression::collapse_axis_with_axis_minus_one(int axis, const Array* owner) const {
    axis = normalize_axis(axis);
    ASSERT2(axis >= 1 && axis < ndim(), utils::make_message("collapse_axis_with_axis_minus_one "
        "axis must >= 1 and less than the dimensionality of the array "
        "(got axis = ", axis, ", ndim = ", ndim(), ")."));
    std::vector<int> newshape = shape_;
    newshape[axis - 1] = newshape[axis] * newshape[axis - 1];
    newshape.erase(newshape.begin() + axis);
    return reshape(newshape, owner);
}

expression_ptr Expression::broadcast_scalar_to_ndim(const int& target_ndim, const Array* owner) const {
    ASSERT2(target_ndim >= 0, utils::make_message("broadcast_scalar_to_ndim"
        " expected a non-negative integer (got ", target_ndim, ")."));
    ASSERT2(is_scalar(), utils::make_message("broadcast_scalar_to_ndim may "
        "only be called on scalars, current shape = ", shape_, "."));
    auto res = copy();
    for (int i = 0; i < target_ndim; ++i) {
        res = res->insert_broadcast_axis(0, nullptr);
    }
    return res;
}

expression_ptr Expression::pluck_axis(const int& axis, const int& pluck_idx, const Array* owner) const {
    return pluck_axis(axis, Slice(pluck_idx, pluck_idx + 1), owner)->squeeze(axis, nullptr);
}

expression_ptr Expression::broadcast_to_shape(const std::vector<int>& new_shape, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(broadcast_to_shape(new_shape))
}

expression_ptr Expression::operator()(int idx, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(operator() (idx))
}

expression_ptr Expression::dimshuffle(const std::vector<int>& pattern, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(dimshuffle(pattern))
}

expression_ptr Expression::reshape(const std::vector<int>& new_shape, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(reshape(new_shape))
}

expression_ptr Expression::pluck_axis(int axis, const Slice& slice_unnormalized, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(pluck_axis(axis, slice_unnormalized))
}

expression_ptr Expression::squeeze(int axis, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(squeeze(axis))
}

expression_ptr Expression::expand_dims(int new_axis, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(expand_dims(new_axis))
}

expression_ptr Expression::broadcast_axis(int axis, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(broadcast_axis(axis))
}
