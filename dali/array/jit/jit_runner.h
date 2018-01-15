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

        // Convenience class for keeping track of shapes, names, declarations
        // and later using them in the template engine
        struct SymbolTable {
            std::vector<const BufferView*> arrays_;
            std::vector<const ScalarView*> scalars_;
            std::vector<const Expression*> shapes_;

            // temporary storage:
            std::vector<Array> temporaries_;
            mutable std::vector<bool> temporary_status_;
            std::vector<const Expression*> temporary_assigns_expressions_;

            mutable std::unordered_map<const Expression*, std::string> declaration_table_;
            mutable std::unordered_map<const Expression*, std::string> shape_declaration_table_;
            std::string get_name(const Expression*) const;
            std::string get_shape(const Expression*) const;
            // retrieve the value of a computation (callcode). If the value is
            // stored in a temporary, return it, else return the original call code.
            std::string get_temporary(const Expression*, std::string callcode) const;

            void declare_array(const BufferView*);
            void declare_scalar(const ScalarView*);
            void declare_shape(const Expression*);

            std::string variable_declarations(const node_to_info_t& node_to_info) const;
            std::vector<Array> collect_buffers(const node_to_info_t& node_to_info) const;
            std::vector<const void*> collect_scalars(const node_to_info_t& node_to_info) const;
            std::vector<std::vector<int>> collect_shapes(const node_to_info_t& node_to_info) const;
            // mark that a value will be re-used multiple times and should be stored into
            // a temporary storage (if the value is a Buffer, then this is ignored)
            void store_into_temporary(const Expression* stored, node_to_info_t* node_to_info);
            // During template generation, keep track of which temporaries are ready for use
            // and which still need to be computed
            void mark_temporary_as_ready(const Expression* expr) const;
        };

        struct JITNode : public Expression {
            static const hash_t optype_hash;
            ////////////////////
            // MUST IMPLEMENT //
            ////////////////////
            virtual void compute_node_compilation_info(int desired_computation_rank,
                                                       const std::vector<int>& desired_computation_shape,
                                                       SymbolTable& symbol_table,
                                                       node_to_info_t* node_to_info) const = 0;
            virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                                 const node_to_info_t& node_to_info,
                                                 memory::DeviceT device_type) const = 0;


            ////////////////////////////////
            // REIMPLEMENT AS YOU SEE FIT //
            ////////////////////////////////

            virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const;
            virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;
            virtual memory::Device preferred_device() const;

            ///////////////////////////////////////////////////////
            // REIMPLEMENT IF YOU WANT TO MAKE A NODE ASSIGNABLE //
            ///////////////////////////////////////////////////////

            virtual std::string assignment_code(const std::vector<Array>& dest,
                                                const std::vector<std::string>& root,
                                                const std::vector<OPERATOR_T>& operators,
                                                const SymbolTable& symbol_table,
                                                const node_to_info_t& node_to_info,
                                                memory::DeviceT device_type,
                                                const std::vector<int>& computation_ranks) const;
            virtual std::string assignment_code_nd(OPERATOR_T operator_t, memory::DeviceT device_type,
                                                   std::string dst, std::string src) const;
            virtual std::string assignment_prefix_code(const std::vector<OPERATOR_T>& operators,
                                                       const node_to_info_t& node_to_info,
                                                       memory::DeviceT device_type,
                                                       const std::vector<int>& computation_ranks) const;
            // internals:
            const int min_computation_rank_;
            JITNode(int min_computation_rank,
                    const std::vector<int>& shape,
                    DType dtype,
                    const std::vector<Array>& arguments,
                    int offset=0,
                    const std::vector<int>& strides={});
            JITNode(const JITNode& other);
            virtual bool supports_operator(OPERATOR_T operator_t) const;

            virtual expression_ptr _reshape(const std::vector<int>& new_shape, const Array* owner) const override;
            virtual expression_ptr _expand_dims(int new_axis, const Array* owner) const override;
            virtual expression_ptr _squeeze(int axis, const Array* owner) const override;
        };

        // return a shared pointer to the underlying jit node, checks for dynamic_cast
        std::shared_ptr<JITNode> as_jit_node(Array array);
        // return a pointer to the underlying jit node, does not dynamic_cast
        JITNode* static_as_jit_node(const Array& array);
        hash_t node_hash(const node_to_info_t& node_to_info, const Array& arr);
        // does dynamic_cast to check if an expression is a jit node:
        bool is_jit_node(const Array& arr);

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


#endif // DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
