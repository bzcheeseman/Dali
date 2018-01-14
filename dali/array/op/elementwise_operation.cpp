#include "elementwise_operation.h"

#include <vector>
#include <numeric>
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/jit/elementwise_kernel_utils.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/reshape.h"
#include "dali/array/jit/scalar_view.h"
#include "dali/array/expression/buffer_view.h"

namespace {
    int compute_min_computation_rank(const std::vector<Array>& arguments) {
        return std::accumulate(arguments.begin(),
           arguments.end(),
           0,
           [](int so_far, const Array& op) {
               return std::max(so_far, op::jit::min_computation_rank(op));
           }
        );
    }
}

namespace op {
namespace jit {
    bool ndim_compatible(const Array& a, const Array& b) {
        int a_ndim = a.ndim();
        int b_ndim = b.ndim();
        return a_ndim == 0 || b_ndim == 0 || a_ndim == b_ndim;
    }

    struct ElementwiseExpression : public JITNode {
        static const hash_t optype_hash;
        const std::string functor_name_;

        ElementwiseExpression(const std::string& functor_name,
                              const std::vector<Array>& arguments,
                              DType dtype) :
                JITNode(compute_min_computation_rank(arguments),
                        arguments[0].shape(),
                        dtype, arguments),
                functor_name_(functor_name) {
            ASSERT2(arguments.size() > 0,
                "Elementwise expression state must have at least one argument.");
        }

        ElementwiseExpression(const ElementwiseExpression& other) :
            ElementwiseExpression(other.functor_name_, other.arguments_, other.dtype_) {}

        virtual DType dtype() const {
            return dtype_;
        }

        virtual std::string name() const {
            return functor_name_;
        }

        virtual expression_ptr copy() const {
            return std::make_shared<ElementwiseExpression>(*this);
        }

        virtual expression_ptr _reshape(const std::vector<int>& new_shape, const Array* owner) const {

            std::vector<Array> reshaped_arguments;
            for (const auto& arg : arguments_) {
                reshaped_arguments.emplace_back(arg.reshape(new_shape));
            }
            return std::make_shared<ElementwiseExpression>(
                functor_name_,
                reshaped_arguments,
                dtype_);
        }

        virtual void compute_node_compilation_info(int desired_computation_rank,
                                                   const std::vector<int>& desired_computation_shape,
                                                   SymbolTable& symbol_table,
                                                   node_to_info_t* node_to_info) const {
            (*node_to_info)[this].computation_rank = desired_computation_rank;
            for (auto& arg: arguments_) {
                op::jit::compute_node_compilation_info(arg,
                                                       desired_computation_rank,
                                                       desired_computation_shape,
                                                       symbol_table,
                                                       node_to_info);
            }
            utils::Hasher hasher;
            hasher.add(optype_hash).add(desired_computation_rank).add(functor_name_);
            for (auto& arg : arguments_) {
                hasher.add(node_to_info->at(arg.expression().get()).hash);
            }
            hasher.add(dtype_);
            (*node_to_info)[this].hash = hasher.value();
        }

        virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const {
            bool is_contig = true;
            for (auto& arg : arguments_) {
                is_contig = is_contig && arg.is_axis_collapsible_with_axis_minus_one(axis);
            }
            return is_contig;
        }

        virtual expression_ptr collapse_axis_with_axis_minus_one(int axis) const {
            std::vector<Array> new_arguments;
            for (auto& arg : arguments_) {
                new_arguments.emplace_back(arg.collapse_axis_with_axis_minus_one(axis));
            }
            return std::make_shared<ElementwiseExpression>(functor_name_, new_arguments, dtype_);
        }

        virtual expression_ptr transpose(const std::vector<int>& permutation) const {
            std::vector<Array> new_arguments;
            for (auto& arg : arguments_) {
                new_arguments.emplace_back(arg.transpose(permutation));
            }
            return std::make_shared<ElementwiseExpression>(functor_name_, new_arguments, dtype_);
        }

        virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                             const node_to_info_t& node_to_info,
                                             memory::DeviceT device_type) const {
            return generate_call_code_nd(this,
                                         utils::make_message(elementwise_kernel_name(arguments_.size(), node_to_info.at(this).computation_rank), "<", functor_name_, ", ", dtype_to_cpp_name(dtype()), ">"),
                                         symbol_table, node_to_info, device_type,
                                         /*has_shape=*/false);
        }

        virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                        memory::DeviceT device_type) const {
            return create_elementwise_kernel_caller(arguments_.size(), node_to_info.at(this).computation_rank);
        }
    };
    const hash_t ElementwiseExpression::optype_hash = std::hash<std::string>()(typeid(ElementwiseExpression).name());
    }  // namespace jit
    Array elementwise(Array a, const std::string& functor_name) {
        return Array(std::make_shared<jit::ElementwiseExpression>(functor_name, std::vector<Array>({a}), a.dtype()));
    }

    Array elementwise(Array a, Array b, const std::string& functor_name) {
        std::tie(a, b) = ensure_arguments_compatible(a, b, functor_name, true);
        return Array(std::make_shared<jit::ElementwiseExpression>(
            functor_name, std::vector<Array>({a, b}), a.dtype()
        ));
    }

    Array astype(Array a, DType type) {
        return type == DTYPE_INT32 ? round(a) : unsafe_cast(a, type);
    }

    Array unsafe_cast(Array a, DType type) {
        if (a.dtype() == type) {
            return a;
        }
        return Array(std::make_shared<jit::ElementwiseExpression>("functor::cast", std::vector<Array>({a}), type));
    }

    Array round(Array a) {
        if (a.dtype() == DTYPE_INT32) {
            return a;
        }
        return Array(std::make_shared<jit::ElementwiseExpression>("functor::round", std::vector<Array>({a}), DTYPE_INT32));
    }

    std::tuple<Array, Array> ensure_arguments_compatible(
            Array a, Array b, const std::string& functor_name,
            bool update_shape) {

        if (update_shape) {
            ASSERT2(jit::ndim_compatible(a, b), utils::make_message(
                    "Arguments to binary operation '",
                    functor_name, "' must have the same rank (got left.ndim = ",
                    a.ndim(), ", left.shape = ", a.shape(), ", and right.ndim = ",
                    b.ndim(), ", right.shape = ", b.shape(), ")."));
        }

        // perform type promotion:
        if (a.dtype() != b.dtype()) {
            auto new_type = type_promotion(a, b);
            a = astype(a, new_type);
            b = astype(b, new_type);
        }
        if (update_shape && a.shape() != b.shape()) {
            auto common_shape = get_common_shape({a, b});
            a = a.is_scalar() ? jit::tile_scalar(a, common_shape) : jit::broadcasted_reshape(a, common_shape);
            b = b.is_scalar() ? jit::tile_scalar(b, common_shape) : jit::broadcasted_reshape(b, common_shape);
        }
        return std::tuple<Array,Array>(a, b);
    }
}  // namespace op
