#include "core/Tape.h"

namespace graph {
    __thread bool backprop_enabled = true;
    thread_local Tape tape;

    void emplace_back(std::function<void()>&& f) {
        tape.backprop.emplace_back(f);
    }

    void backward() {
        tape.backward();
    }


    /* Tape */

    void Tape::backward () {
        for (auto it = backprop.rbegin(); it != backprop.rend(); ++it)
            (*it)();
        backprop.clear();
    }

    /* NoBackprop */

    NoBackprop::NoBackprop() : old_value(backprop_enabled) {
        backprop_enabled = false;
    }

    NoBackprop::~NoBackprop() {
        backprop_enabled = old_value;
    }

}
