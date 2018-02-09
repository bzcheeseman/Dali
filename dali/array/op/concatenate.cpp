#include "concatenate.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"
#include "dali/utils/print_utils.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/control_flow.h"
#include "dali/array/jit/reshape.h"

namespace op {
    Array concatenate(const std::vector<Array>& arrays, int axis) {
        ASSERT2(arrays.size() > 0, "concatenate must receive at least one array.");
        if (arrays.size() == 1) return arrays[0];
        int ndim = -1;
        std::vector<int> common_shape;
        int total_concat_dim = 0;
        for (auto& array : arrays) {
            if (ndim == -1) {
                ndim = array.ndim();
                ASSERT2(ndim > 0, utils::make_message(
                    "cannot concatenate scalar arrays (got ndim = ", ndim, ")."));
                if (axis < 0) {
                    axis = axis + ndim;
                }
                ASSERT2(axis >= 0 && axis < ndim, utils::make_message(
                    "concatenation axis must be >= 0 and < arrays[0].ndim = ", ndim,
                    ", but got axis = ", axis, "."));
                common_shape = array.shape();
                total_concat_dim = array.shape()[axis];
            } else {
                ASSERT2(ndim == array.ndim(), utils::make_message(
                    "All arrays in concatenate must have the same dimensionality, but got ",
                    array.expression_name(), " with ndim = ", array.ndim(), " != ",
                    arrays[0].expression_name(), ".ndim = ", arrays[0].ndim(), "."));
                for (int i = 0; i < common_shape.size(); i++) {
                    if (i != axis) {
                        if (common_shape[i] != array.shape()[i] && common_shape[i] == 1) {
                            common_shape[i] = array.shape()[i];
                        }
                        ASSERT2(common_shape[i] == array.shape()[i] | array.shape()[i] == 1, utils::make_message(
                            "All shapes must have length-1 or equal dimensions on all axes except "
                            "the concatenation axis, but got ", array.expression_name(),
                            " with shape = ", array.shape(), " which differs on shape[", i, "] = ",
                            array.shape()[i], " from expected = ", common_shape[i], "."));
                    }
                }
                total_concat_dim += array.shape()[axis];
            }
        }
        auto concated_shape = common_shape;
        concated_shape[axis] = total_concat_dim;
        Array result(concated_shape, arrays[0].dtype());
        std::vector<Array> assignments;
        assignments.reserve(arrays.size());
        int so_far = 0;
        for (auto array : arrays) {
            common_shape[axis] = array.shape()[axis];
            if (array.shape() != common_shape) {
                array = jit::broadcasted_reshape(array, common_shape);
            }
            assignments.emplace_back(op::assign(
                result.pluck_axis(axis, Slice(so_far, so_far + array.shape()[axis])),
                OPERATOR_T_EQL, array));
            so_far += array.shape()[axis];
        }
        return Array(std::make_shared<ControlFlow>(result, assignments));
    }
    Array hstack(const std::vector<Array>& arrays) {
        return concatenate(arrays, -1);
    }
    Array vstack(const std::vector<Array>& arrays) {
        return concatenate(arrays, 0);
    }
}
