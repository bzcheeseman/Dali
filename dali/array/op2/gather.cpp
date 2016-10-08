#include "circular_convolution.h"

#include "dali/array/op2/operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

struct GatherState : public OperationState {
    static const hash_t optype_hash;

    operation_state_ptr source_;
    operation_state_ptr indices_;

    GatherState(operation_state_ptr source, operation_state_ptr indices) :
            OperationState(
                std::max(
                    2,
                    source->ndim() + indices->min_computation_rank_ - 1
                )
            ),
            source_(source),
            indices_(indices) {
    }

    std::string prefix_code(const node_to_info_t& node_to_info) const {
        std::string access;
        bool is_2d = node_to_info.at(this).computation_rank == 2 &&
                     node_to_info.at(source_.get()).computation_rank == 2;
        if (is_2d) {
            // use a simpler set of functions when access pattern is well understood (no
            // loops or unrolled loops needed).
            // TODO(jonathan): auto-generate full access pattern without any loops
            access = (
                "    XINLINE T operator[](const Shape<ndim>& query) {\n"
                "        return source_[{indices_(query[0]), query[1]}];\n"
                "    }\n");
        } else {
            access = (
                "    XINLINE T operator[](const Shape<ndim>& query) {\n"
                "        Shape<C1::ndim> source_query = query.template axis_reduced_shape<C2::ndim, C1::ndim - 1, 1>();\n"
                "        source_query[0] = indices_[query.template axis_reduced_shape<0, C2::ndim>()];\n"
                "        return source_[source_query];\n"
                "    }\n");
        }
        std::string name = utils::make_message("GatherKernel", is_2d ? "2D" : "ND");

        return utils::make_message("template<typename C1, typename C2>\n"
        "struct ", name, " {\n"
        "    const C1& source_;\n"
        "    const C2& indices_;\n"
        "    static const int ndim = C1::ndim + C2::ndim - 1;\n"
        "    typedef typename C1::T T;\n"
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

    std::vector<operation_state_ptr> arguments() const {
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

    operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const {
        int indices_ndim = indices_->ndim();
        if (dim < indices_ndim) {
            return std::make_shared<GatherState>(
                source_,
                indices_->collapse_dim_with_dim_minus_one(dim)
            );
        } else {
            return std::make_shared<GatherState>(
                source_->collapse_dim_with_dim_minus_one(dim - indices_ndim + 1),
                indices_
            );
        }
    }

    operation_state_ptr transpose(const std::vector<int>& permutation) const {
        // TODO(jonathan): there is a way to transpose the index dimensions of
        // gather, or the non-leading dimension of the source.
        throw std::runtime_error(
            "Cannot transpose gather (yet)."
        );
        return shared_from_this();
    }

    void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            std::vector<const ArrayOperationState*>* arrays,
            std::vector<const ScalarOperationState*>* scalars,
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
            const node_to_info_t& node_to_info) const {
        bool is_2d = node_to_info.at(this).computation_rank == 2 &&
                     node_to_info.at(source_.get()).computation_rank == 2;
        return utils::make_message("gather_kernel", is_2d ? "2D" : "ND", "(",
                                    source_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    indices_->get_call_code_nd(symbol_table, node_to_info),
                                    ")");
    }
};

const hash_t GatherState::optype_hash = std::hash<std::string>()("GatherState");

namespace op2 {
    Operation gather(const Operation& source, const Operation& indices) {
        ASSERT2(
            source.ndim() > 0,
            utils::make_message("gather must be called on source with ndim >="
                " 1 (got ndim=", source.ndim(), ").")
        );
        ASSERT2(
            indices.dtype() == DTYPE_INT32,
            utils::make_message("indices must be integers (got dtype=", indices.dtype(), ").")
        );
        return Operation(std::make_shared<GatherState>(source.state_, indices.state_));
    }
}  // namespace op2
