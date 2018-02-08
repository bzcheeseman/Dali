#include "arange.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/utils/make_message.h"

namespace op {
namespace jit {

struct Arange : public JITNode {
    Arange(Array start, Array step, int size) :
        JITNode({size}, start.dtype(), {start, step}) {
    }

    virtual int min_computation_rank() const override {
        return 1;
    }

    virtual memory::Device preferred_device() const override {
        return arguments_[0].preferred_device();
    }
    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
        return generate_call_code_nd(this, kernel_name(), symbol_table, device_type,
                                     /*has_shape=*/true);
    }

    virtual bool shape_required() const override {return true;}

    virtual std::string kernel_name() const {
        return utils::make_message("arange", ndim(), "d");
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        define_kernel(/*ndim=*/ndim(),
                      /*has_shape=*/true,
                      /*arguments=*/{"start", "step"},
                      /*kernel=*/"start_[0] + indices_to_offset(shape_, query) * step_[0]",
                      /*name=*/kernel_name(),
                      /*is_assignable=*/false,
                      insert);
    }

    virtual expression_ptr copy() const override {
        return std::make_shared<Arange>(arguments_[0], arguments_[1], shape_[0]);
    }
};
}  // namespace jit

Array arange(int size) {
    return arange(0, size, 1);
}

Array arange(int start, int stop) {
    return arange(start, stop, 1);
}

Array arange(int start, int stop, int step) {
    ASSERT2(start != stop, utils::make_message(
        "arange's start and stop must be different "
        "(got start = ", start, ", stop = ", stop, ")."));
    int size = std::max((stop - start) / step, 1);
    ASSERT2(size > 0, utils::make_message(
        "arange's size must be strictly positive "
        "(got start = ", start, ", stop = ", stop,
        ", step = ", step, ")."));
    return Array(std::make_shared<jit::Arange>(start, step, size));
}

}  // namespace op
