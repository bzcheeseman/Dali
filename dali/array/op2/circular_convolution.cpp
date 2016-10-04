#include "circular_convolution.h"

#include "dali/array/op2/operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"



struct CircularConvolutionOperationState : public OperationState {
    static const hash_t optype_hash;

    operation_state_ptr content_;
    operation_state_ptr weights_;


    CircularConvolutionOperationState(operation_state_ptr content, operation_state_ptr weights) :
            OperationState(std::max(2, std::max(content->min_computation_rank_, weights->min_computation_rank_))),
            content_(content),
            weights_(weights) {
    }

    std::string prefix_code(const node_to_info_t& node_to_info) const {
        // TODO(jonathan, szymon): clearly kernel writing is repetitive, a method could
        //                         be designed here to factor out all the boilerplate
        //                         to instantiate easily 2, 3, etc... arg templates.
        return"template<typename C1, typename C2>\n"
        "struct CircularConvolutionKernel {\n"
        "    const C1& a_view_;\n"
        "    const C2& b_view_;\n"
        "    static const int ndim = C1::ndim;\n"
        "    typedef typename C1::T T;\n"
        "    XINLINE CircularConvolutionKernel(const C1& a_view, const C2& b_view)\n"
        "        : a_view_(a_view), b_view_(b_view) {}\n"
        "    XINLINE T operator[](Shape<ndim> query) {\n"
        "        T res = static_cast<T>(0);\n"
        "        const int conv_size = b_view_.shape()[ndim - 1];\n"
        "        const int x = query[ndim - 1];\n"
        "        Shape<ndim> a_query = query;\n"
        "        Shape<ndim> b_query = query;\n"
        "        int& shift_idx = b_query[ndim - 1];\n"
        "        int& offset = a_query[ndim - 1];\n"
        "        #pragma clang loop vectorize(enable)\n"
        "        #pragma clang loop interleave(enable)\n"
        "        for (shift_idx = 0; shift_idx < conv_size; shift_idx++) {\n"
        "            offset = x + shift_idx;\n"
        "            if (offset >= conv_size) {\n"
        "                offset -= conv_size;\n"
        "            }\n"
        "            res += a_view_[a_query] * b_view_[b_query];\n"
        "        }\n"
        "        return res;\n"
        "    }\n"
        "};\n"
        "template<typename C1, typename C2>\n"
        "CircularConvolutionKernel<C1, C2> circular_convolution_kernel(const C1& a, const C2& b) {\n"
        "    return CircularConvolutionKernel<C1, C2>(a, b);\n"
        "}\n";
    }

    DType dtype() const {
        return content_->dtype();
    }

    std::vector<int> bshape() const {
        return get_common_bshape({content_->bshape(), weights_->bshape()});
    }

    int ndim() const {
        return bshape().size();
    }

    std::vector<operation_state_ptr> arguments() const {
        return {content_, weights_};
    }

    bool is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        if (dim == ndim() - 1) {
            return false;
        }
        return weights_->is_dim_collapsible_with_dim_minus_one(dim) &&
               content_->is_dim_collapsible_with_dim_minus_one(dim);
    }

    operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const {
        return std::make_shared<CircularConvolutionOperationState>(
            content_->collapse_dim_with_dim_minus_one(dim),
            weights_->collapse_dim_with_dim_minus_one(dim)
        );
    }

    operation_state_ptr transpose(const std::vector<int>& permutation) const {
        throw std::runtime_error("this code is untested.");
        return shared_from_this();
    }

    void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            std::vector<const ArrayOperationState*>* arrays,
            std::vector<const ScalarOperationState*>* scalars,
            node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        content_->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
        weights_->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
        (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                    .add(desired_computation_rank)
                                                    .add(node_to_info->at(content_.get()).hash)
                                                    .add(node_to_info->at(weights_.get()).hash)
                                                    .value();
    }


    std::string get_call_code_nd(
            const symbol_table_t& symbol_table,
            const node_to_info_t& node_to_info) const {
        return utils::make_message("circular_convolution_kernel(",
                                    content_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    weights_->get_call_code_nd(symbol_table, node_to_info),
                                    ")");
    }
};

const hash_t CircularConvolutionOperationState::optype_hash = std::hash<std::string>()("CircularConvolutionOperationState");


namespace op2 {
    Operation circular_convolution(const Operation& x, const Operation& weights) {
        return Operation(std::make_shared<CircularConvolutionOperationState>(
            x.state_,
            weights.state_
        ));
    }
}  // namespace op2
