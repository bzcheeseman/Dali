#include "gather.h"

#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

struct GatherState : public RtcExpression {
    static const hash_t optype_hash;

    std::shared_ptr<const RtcExpression> source_;
    std::shared_ptr<const RtcExpression> indices_;

    GatherState(std::shared_ptr<const RtcExpression> source, std::shared_ptr<const RtcExpression> indices) :
            RtcExpression(
                std::max(
                    2,
                    source->ndim() + indices->min_computation_rank_ - 1
                )
            ),
            source_(source),
            indices_(indices) {
    }

    virtual std::string name() const {
        return "gather";
    }

    std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
        std::string access;
        bool is_2d = node_to_info.at(this).computation_rank == 2 &&
                     node_to_info.at(source_.get()).computation_rank == 2;
        bool use_references = is_assignable();
        if (is_2d) {
            // use a simpler set of functions when access pattern is well understood (no
            // loops or unrolled loops needed).
            // TODO(jonathan): auto-generate full access pattern without any loops
            access = utils::make_message(
                "    XINLINE T", use_references ? "&" : "", " operator[](const Shape<ndim>& query) ", use_references ? "" : "const", " {\n"
                "        return source_[{indices_(query[0]), query[1]}];\n"
                "    }\n");
        } else {
            access = utils::make_message(
                "    XINLINE T", use_references ? "&" : "", " operator[](const Shape<ndim>& query) ", use_references ? "" : "const", " {\n"
                "        Shape<C1::ndim> source_query = query.template axis_reduced_shape<C2::ndim, C1::ndim - 1, 1>();\n"
                "        source_query[0] = indices_[query.template axis_reduced_shape<0, C2::ndim>()];\n"
                "        return source_[source_query];\n"
                "    }\n");
        }
        std::string name = utils::make_message("GatherKernel", is_2d ? "2D" : "ND");

        std::stringstream shape_assignments_ss;
        for (int i = 0; i < node_to_info.at(indices_.get()).computation_rank; i++) {
            shape_assignments_ss << "    res[" << i << "] = indices_shape[" << i << "];\n";
        }
        std::string shape_assignments = shape_assignments_ss.str();

        return utils::make_message("template<typename C1, typename C2>\n"
        "struct ", name, " {\n"
        "    C1 source_;\n"
        "    C2 indices_;\n"
        "    static const int ndim = C1::ndim + C2::ndim - 1;\n"
        "    typedef typename C1::T T;\n"
        "    XINLINE Shape<ndim> shape() const {\n"
        "        auto res = source_.shape().template axis_reduced_shape<1, C1::ndim-1, C2::ndim>();\n",
        "        auto indices_shape = indices_.shape();\n", shape_assignments,
        "        return res;\n"
        "    }\n"
        "    XINLINE ", name, "(const C1& source, const C2& indices)\n"
        "        : source_(source), indices_(indices) {}\n", access,
        "};\n"
        "template<typename C1, typename C2>\n",
        name, "<C1, C2> gather_kernel", is_2d ? "2D" : "ND", "(const C1& a, const C2& b) {\n"
        "    return ", name, "<C1, C2>(a, b);\n"
        "}\n");
    }

    DType dtype() const {
        return source_->dtype();
    }

    std::vector<int> bshape() const {
        auto res = indices_->bshape();
        auto source_bshape = source_->bshape();
        res.insert(res.end(), source_bshape.begin() + 1, source_bshape.end());
        return res;
    }

    virtual int ndim() const {
        return indices_->ndim() + source_->ndim() - 1;
    }

    virtual bool is_assignable() const {
        return source_->is_assignable();
    }

    std::vector<std::shared_ptr<const ExpressionState>> arguments() const {
        return {source_, indices_};
    }

    bool is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        int indices_ndim = indices_->ndim();
        if (dim < indices_ndim) {
            return indices_->is_dim_collapsible_with_dim_minus_one(dim);
        }
        if (dim == indices_ndim) {
            // this is the dimensionality of the output's dimension just after
            // the leading dimension. Collapsing this dimension means losing track
            // of what is being gathered.
            return false;
        }
        if (dim >= indices_ndim + 1) {
            // because dim is being observed from the output, we must
            // subtract all the index dimensions, and add back a dimension
            // hidden by gather
            return source_->is_dim_collapsible_with_dim_minus_one(
                dim - indices_ndim + 1
            );
        }
        return false;
    }

    std::shared_ptr<const RtcExpression> collapse_dim_with_dim_minus_one(const int& dim) const {
        int indices_ndim = indices_->ndim();
        if (dim < indices_ndim) {
            return std::make_shared<GatherState>(
                source_,
                indices_->collapse_dim_with_dim_minus_one(dim)
            );
        } else {
            return std::make_shared<GatherState>(
                source_->collapse_dim_with_dim_minus_one(dim - indices_ndim + 1),
                indices_
            );
        }
    }

    std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const {
        // TODO(jonathan): there is a way to transpose the index dimensions of
        // gather, or the non-leading dimension of the source.
        throw std::runtime_error(
            "Cannot transpose gather (yet)."
        );
        return jit_shared_from_this();
    }

    void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            std::vector<const ArrayWrapper*>* arrays,
            std::vector<const ScalarWrapper*>* scalars,
            node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        int source_ndim = source_->ndim();
        // indices dim 1, dim2, etc... source dim 2, dim 3, etc...
        std::vector<int> source_shape(
            desired_computation_shape.end() - (source_ndim - 1),
            desired_computation_shape.end()
        );
        // add dim 1 of source back in (hidden from output by gather operation).
        source_shape.insert(source_shape.begin(), std::abs(source_->bshape()[0]));
        std::vector<int> indices_shape(
            desired_computation_shape.begin(),
            desired_computation_shape.end() - (source_ndim - 1)
        );

        source_->compute_node_compilation_info(source_ndim, source_shape, arrays, scalars, node_to_info);
        indices_->compute_node_compilation_info(desired_computation_rank - (source_ndim - 1), indices_shape, arrays, scalars, node_to_info);
        bool is_2d = desired_computation_rank == 2 && source_ndim == 2;
        (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                    .add(desired_computation_rank)
                                                    .add(is_2d)
                                                    .add(node_to_info->at(source_.get()).hash)
                                                    .add(node_to_info->at(indices_.get()).hash)
                                                    .value();
    }


    std::string get_call_code_nd(
            const symbol_table_t& symbol_table,
            const node_to_info_t& node_to_info,
            memory::DeviceT device_type) const {
        bool is_2d = node_to_info.at(this).computation_rank == 2 &&
                     node_to_info.at(source_.get()).computation_rank == 2;
        return utils::make_message("gather_kernel", is_2d ? "2D" : "ND", "(",
                                    source_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    indices_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ")");
    }
};

const hash_t GatherState::optype_hash = std::hash<std::string>()("GatherState");

namespace op {
    Expression gather(const Expression& source, const Expression& indices) {
        ASSERT2(
            source.ndim() > 0,
            utils::make_message("gather must be called on source with ndim >="
                " 1 (got ndim=", source.ndim(), ").")
        );
        ASSERT2(
            indices.dtype() == DTYPE_INT32,
            utils::make_message("indices must be integers (got dtype=", indices.dtype(), ").")
        );
        return Expression(std::make_shared<GatherState>(source.state_->as_jit(), indices.state_->as_jit()));
    }
}  // namespace op
