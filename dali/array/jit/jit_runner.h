#ifndef DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
#define DALI_ARRAY_EXPRESSION_JIT_RUNNER_H

#include "dali/array/array.h"
#include "dali/array/expression/expression.h"
#include "dali/array/expression/buffer_view.h"
#include "dali/array/expression/scalar_view.h"
#include "dali/utils/hash_utils.h"

namespace op {
    namespace jit {
        struct CompilationInfo {
            int  computation_rank;
            std::string name;
            std::vector<int> computation_shape;
            hash_t hash;
        };

        typedef std::unordered_map<const Expression*, std::string>     symbol_table_t;
        typedef std::unordered_map<const Expression*, CompilationInfo> node_to_info_t;

        struct JITNode : public Expression {
            // implement these
            virtual void compute_node_compilation_info(int desired_computation_rank,
                                                       const std::vector<int>& desired_computation_shape,
                                                       std::vector<std::shared_ptr<BufferView>>* arrays,
                                                       std::vector<std::shared_ptr<ScalarView>>* scalars,
                                                       node_to_info_t* node_to_info) const = 0;
            virtual std::string get_call_code_nd(const symbol_table_t& symbol_table,
                                                 const node_to_info_t& node_to_info,
                                                 memory::DeviceT device_type) const = 0;


            ///////////////////////////////////////////////////////////////////////////////
            //            REIMPLEMENT AS YOU SEE FIT                                     //
            ///////////////////////////////////////////////////////////////////////////////

            virtual bool is_axis_collapsible_with_axis_minus_one(const int& axis) const = 0;
            virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;
            virtual memory::Device preferred_device() const;

            // internals:
            const int min_computation_rank_;
            JITNode(int min_computation_rank,
                    const std::vector<int>& shape,
                    DType dtype,
                    int offset=0,
                    const std::vector<int>& strides={});
            JITNode(const JITNode& other);
        };

        std::shared_ptr<JITNode> as_jit_node(Array array);
        hash_t node_hash(const node_to_info_t& node_to_info, const Array& arr);

        struct JITRunner : public JITNode {
            Array root_;
            std::vector<Array> leaves_;

            virtual expression_ptr copy() const;
            JITRunner(Array root, const std::vector<Array>& leaves);
            virtual std::vector<Array> arguments() const;
            virtual memory::Device preferred_device() const;
            virtual void compute_node_compilation_info(int desired_computation_rank,
                                                       const std::vector<int>& desired_computation_shape,
                                                       std::vector<std::shared_ptr<BufferView>>* arrays,
                                                       std::vector<std::shared_ptr<ScalarView>>* scalars,
                                                       node_to_info_t* node_to_info) const;
            virtual bool is_axis_collapsible_with_axis_minus_one(const int& axis) const;
            virtual std::string get_call_code_nd(const symbol_table_t& symbol_table,
                                                 const node_to_info_t& node_to_info,
                                                 memory::DeviceT device_type) const;
        };
    }
}


#endif // DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
