#ifndef DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
#define DALI_ARRAY_EXPRESSION_JIT_RUNNER_H

#include "dali/array/array.h"
#include "dali/array/expression/expression.h"
#include "dali/array/expression/buffer_view.h"
#include "dali/utils/hash_utils.h"

namespace op {
    namespace jit {

        enum PARALLELISM_T {
            INDEPENDENT_BLOCK_WARP  = 0,
            INDEPENDENT_BLOCK = 1
        };

        struct ScalarView;

        struct CompilationInfo {
            int computation_rank = -1;
            std::string name;
            std::vector<int> computation_shape;
            hash_t hash;
            hash_t data_hash;
        };

        typedef std::unordered_map<const Expression*, CompilationInfo> node_to_info_t;

        // Convenience class for keeping track of shapes, names, declarations
        // and later using them in the template engine
        struct SymbolTable {
            std::vector<const BufferView*> arrays_;
            std::vector<const ScalarView*> scalars_;
            std::vector<const Expression*> shapes_;
            std::unordered_map<const BufferView*, int> arrays_visited_;
            utils::Hasher array_order_;

            std::unordered_map<const ScalarView*, int> scalars_visited_;
            utils::Hasher scalar_order_;

            // temporary storage:
            std::vector<Array> temporaries_;
            std::vector<expression_ptr> temporary_assigns_expressions_;
            std::vector<hash_t> temporary_assigns_expression_hashes_;

            mutable std::unordered_map<const Expression*, std::string> declaration_table_;
            mutable std::unordered_map<const Expression*, std::string> shape_declaration_table_;
            std::string get_name(const Expression*) const;
            std::string get_shape(const Expression*) const;

            void declare_array(const BufferView*);
            // each unique array gets an index for its insertion time
            int get_array_index(const BufferView*) const;
            int get_scalar_index(const ScalarView*) const;
            void declare_scalar(const ScalarView*);
            void declare_shape(const Expression*);

            std::string variable_declarations(const node_to_info_t& node_to_info) const;
            std::vector<Array> collect_buffers(const node_to_info_t& node_to_info) const;
            std::vector<const void*> collect_scalars(const node_to_info_t& node_to_info) const;
            std::vector<std::vector<int>> collect_shapes(const node_to_info_t& node_to_info) const;
            // mark that a value will be re-used multiple times and should be stored into
            // a temporary storage (if the value is a Buffer, then this is ignored)
            void store_into_temporary(expression_ptr stored, node_to_info_t& node_to_info);
        };

        struct JITNode : public Expression {
            static const hash_t optype_hash;
            // ////////////////////
            // // MUST IMPLEMENT //
            // ////////////////////
            virtual void compute_node_compilation_info(int desired_computation_rank,
                                                       const std::vector<int>& desired_computation_shape,
                                                       SymbolTable& symbol_table,
                                                       node_to_info_t& node_to_info) const = 0;
            virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                                 const node_to_info_t& node_to_info,
                                                 memory::DeviceT device_type) const = 0;

            // ////////////////////////////////
            // // REIMPLEMENT AS YOU SEE FIT //
            // ////////////////////////////////

            // whether this node will need a shape argument (and thus one
            // should be kept and given to the kernel)
            virtual bool shape_required() const;
            // whether to allow aliasing of a node
            virtual bool antialias() const;
            virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const override;
            virtual memory::Device preferred_device() const override;

            virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;
            virtual PARALLELISM_T parallelism_type() const;
            virtual hash_t compute_node_data_hash(const node_to_info_t& node_to_info, const SymbolTable&) const;
            virtual expression_ptr jit_right_fit_ndim(int ndim) const;



            virtual int min_computation_rank() const = 0;

            ///////////////////////////////////////////////////////
            // REIMPLEMENT IF YOU WANT TO MAKE A NODE ASSIGNABLE //
            ///////////////////////////////////////////////////////

            virtual std::string assignment_code(hash_t hash,
                                                const std::vector<Array>& dest,
                                                const std::vector<std::string>& root,
                                                const std::vector<OPERATOR_T>& operators,
                                                const SymbolTable& symbol_table,
                                                const node_to_info_t& node_to_info,
                                                memory::DeviceT device_type,
                                                const std::vector<int>& computation_ranks,
                                                const std::vector<PARALLELISM_T>& parallelism_types) const;
            virtual std::string assignment_code_nd(OPERATOR_T operator_t, memory::DeviceT device_type,
                                                   std::string dst, std::string src) const;
            virtual std::string assignment_prefix_code(hash_t hash,
                                                       const std::vector<OPERATOR_T>& operators,
                                                       const node_to_info_t& node_to_info,
                                                       memory::DeviceT device_type,
                                                       const std::vector<int>& computation_ranks,
                                                       const std::vector<PARALLELISM_T>& parallelism_types) const;
            // internals:
            JITNode(const std::vector<int>& shape,
                    DType dtype,
                    const std::vector<Array>& arguments,
                    int offset=0,
                    const std::vector<int>& strides={});
            JITNode(const JITNode& other);
            virtual bool supports_operator(OPERATOR_T operator_t) const override;
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
                                           node_to_info_t& node_to_info);

        std::string get_call_code_nd(const Array& a,
                                     const SymbolTable& symbol_table,
                                     const node_to_info_t& node_to_info,
                                     memory::DeviceT device_type);

        int min_computation_rank(const Array& array);
        expression_ptr jit_right_fit_ndim(const Array& array, int ndim);
        // describe whether a jit node is using the full warp internally
        // or if it is intra-warp (gpu kernel generation)
        PARALLELISM_T parallelism_type(const Array& array);

        // create a condition under which one jit node can be replaced by another
        // during code generation (e.g. to promote computation to be over warps)
        int register_jit_optimization(int priority,
                                      std::function<bool(const Array&, memory::DeviceT, const node_to_info_t&)> condition,
                                      std::function<Array(const Array&)> transformation,
                                      const std::string& name);
    }
}


#endif // DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
