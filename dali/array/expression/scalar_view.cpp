#include "scalar_view.h"

#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/shape.h"
#include "dali/array/array.h"

ScalarView::ScalarView(double value) :
        Expression({}, DTYPE_DOUBLE, 0, {}),
        value_(value){
}

ScalarView::ScalarView(const ScalarView& other) : ScalarView(other.value_) {}

expression_ptr ScalarView::copy() const {
    return std::make_shared<ScalarView>(*this);
}

memory::Device ScalarView::preferred_device() const {
    return memory::default_preferred_device;
}

std::vector<Array> ScalarView::arguments() const {
    return {};
}

bool ScalarView::spans_entire_memory() const {
    return true;
}
