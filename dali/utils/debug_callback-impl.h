
template<typename... Args>
auto DebugCallback<Args...>::register_callback(callback_t callback) -> callback_handle_t {
    return callbacks.insert(callbacks.end(), callback);
}

template<typename... Args>
void DebugCallback<Args...>::deregister_callback(callback_handle_t handle) {
    callbacks.erase(handle);
}

template<typename... Args>
int DebugCallback<Args...>::activate(Args... args) {
    int num_called = 0;
    for (callback_t c: callbacks) {
        c(args...);
        num_called++;
    }
    return num_called;
}


template<typename... Args>
ScopedCallback<Args...>::ScopedCallback(
        typename DebugCallback<Args...>::callback_t callback,
        DebugCallback<Args...>* dc_) :
                dc_handle(dc_->register_callback(callback)),
                dc(dc_),
                owns_handle(true) {
}

template<typename... Args>
ScopedCallback<Args...>::~ScopedCallback() {
    if (owns_handle) {
        dc->deregister_callback(dc_handle);
    }
}

template<typename... Args>
ScopedCallback<Args...>::ScopedCallback(ScopedCallback&& other) :
        dc(std::move(other.dc)),
        dc_handle(std::move(other.dc_handle)),
        owns_handle(std::move(other.owns_handle)) {
    other.owns_handle = false;
}


template<typename... Args>
ScopedCallback<Args...> make_scoped_callback(
        typename DebugCallback<Args...>::callback_t callback,
        DebugCallback<Args...>* dc) {
    return ScopedCallback<Args...>(callback, dc);
}
