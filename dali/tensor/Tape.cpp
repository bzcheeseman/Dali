#include "Tape.h"
#include <iostream>

namespace graph {
    thread_local bool _backprop_enabled = true;
    thread_local Tape tape;

    void emplace_back(std::function<void()>&& f) {
        tape.backprop.emplace_back(f);
    }

    void backward() {
        tape.backward();
    }

    void clear() {
        tape.backprop.clear();
    }

    bool backprop_enabled() {
        return _backprop_enabled;
    }

    void _set_backprop_enabled(bool value) {
        _backprop_enabled = value;
    }

    size_t size() {
        return tape.backprop.size();
    }


    /* Tape */

    void Tape::backward () {
        for (auto it = backprop.rbegin(); it != backprop.rend(); ++it)
            (*it)();
        backprop.clear();
    }

    /* NoBackprop */
    NoBackprop::NoBackprop() : NoBackprop(true) {
    }

    NoBackprop::NoBackprop(bool condition) : old_value(_backprop_enabled), enabled(condition) {
        if(enabled)
            _backprop_enabled = false;
    }

    NoBackprop::~NoBackprop() {
        if (enabled)
            _backprop_enabled = old_value;
    }

}
