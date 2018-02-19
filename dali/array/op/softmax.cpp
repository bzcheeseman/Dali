#include "softmax.h"

namespace op {
    // TODO(jonathan): ensure cudnn softmax can be suggested as replacement here
    Array softmax(const Array& logits, int axis) {
        auto exped = (logits - logits.max({axis}, true)).exp();
        return exped / exped.sum({axis}, true);
    }

    Array softmax_temperature(const Array& logits, const Array& temperature, int axis) {
        auto exped = (logits - logits.max({axis}, true)).exp();
        return exped / exped.sum({axis}, true);
    }
}
