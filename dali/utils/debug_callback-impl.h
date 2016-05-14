
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
