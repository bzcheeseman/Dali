#ifndef DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
#define DALI_ARRAY_EXPRESSION_JIT_RUNNER_H

#include "dali/array/array.h"
#include "dali/array/expression/expression.h"
#include "dali/array/expression/buffer_view.h"

#include "dali/utils/hash_utils.h"

namespace op {
    namespace jit {

        struct ScalarView;

        struct CompilationInfo {
            int computation_rank = -1;
            std::string name;
            std::vector<int> computation_shape;
            hash_t hash;
        };

        typedef std::unordered_map<const Expression*, CompilationInfo> node_to_info_t;


        struct SymbolTable {
            std::vector<const BufferView*> arrays_;
            std::vector<const ScalarView*> scalars_;
            std::vector<const Expression*> shapes_;
            mutable std::unordered_map<const Expression*, std::string> declaration_table_;
            mutable std::unordered_map<const Expression*, std::string> shape_declaration_table_;
            std::string get_name(const Expression*) const;
            std::string get_shape(const Expression*) const;
            void declare_array(const BufferView*);
            void declare_scalar(const ScalarView*);
            void declare_shape(const Expression*);
            std::string variable_declarations(const node_to_info_t& node_to_info) const;
            std::vector<Array> collect_buffers(const node_to_info_t& node_to_info) const;
            std::vector<const void*> collect_scalars(const node_to_info_t& node_to_info) const;
            std::vector<const int*> collect_shapes(const node_to_info_t& node_to_info) const;
        };

        struct JITNode : public Expression {
            static const hash_t optype_hash;
            // implement these
            virtual void compute_node_compilation_info(int desired_computation_rank,
                                                       const std::vector<int>& desired_computation_shape,
                                                       SymbolTable& symbol_table,
                                                       node_to_info_t* node_to_info) const = 0;
            virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                                 const node_to_info_t& node_to_info,
                                                 memory::DeviceT device_type) const = 0;


            ///////////////////////////////////////////////////////////////////////////////
            //            REIMPLEMENT AS YOU SEE FIT                                     //
            ///////////////////////////////////////////////////////////////////////////////

            virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const;
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

            virtual bool supports_operator(OPERATOR_T operator_t) const;
        };

        std::shared_ptr<JITNode> as_jit_node(Array array);
        hash_t node_hash(const node_to_info_t& node_to_info, const Array& arr);
        bool is_jit(const Array& arr);

        void compute_node_compilation_info(const Array& a,
                                           int desired_computation_rank,
                                           const std::vector<int>& desired_computation_shape,
                                           SymbolTable& symbol_table,
                                           node_to_info_t* node_to_info);

        std::string get_call_code_nd(const Array& a,
                                     const SymbolTable& symbol_table,
                                     const node_to_info_t& node_to_info,
                                     memory::DeviceT device_type);

        int min_computation_rank(const Array& array);
    }
}

#include "dali/array/jit/scalar_view.h"


#endif // DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
