#include "arange.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
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

Array arange(Array start, Array stop, Array step) {
    ASSERT2(start.ndim() == 0, utils::make_message("arange's start argument must be a scalar but got start.shape = ", start.shape(), "."));
    ASSERT2(stop.ndim() == 0, utils::make_message("arange's stop argument must be a scalar but got stop.shape = ", stop.shape(), "."));
    ASSERT2(step.ndim() == 0, utils::make_message("arange's step argument must be a scalar but got step.shape = ", step.shape(), "."));
    int size = (int)op::ceil((stop - start) / step);
    ASSERT2(size > 0, utils::make_message(
        "arange's size must be strictly positive "
        "(got start = ", (double)start, ", stop = ", (double)stop,
        ", step = ", (double)step, ")."));
    return Array(std::make_shared<jit::Arange>(start, step, size));
}

}  // namespace op
