
template<typename... Args>
auto Observation<Args...>::observe(callback_t callback) -> callback_handle_t {
    return callbacks.insert(callbacks.end(), callback);
}

template<typename... Args>
void Observation<Args...>::lose_interest(callback_handle_t handle) {
    callbacks.erase(handle);
}

template<typename... Args>
int Observation<Args...>::notify(Args... args) {
    int num_called = 0;
    for (callback_t c: callbacks) {
        c(args...);
        num_called++;
    }
    return num_called;
}


template<typename... Args>
ObserverGuard<Args...>::ObserverGuard(
        typename Observation<Args...>::callback_t callback,
        Observation<Args...>* dc_) :
                dc_handle(dc_->observe(callback)),
                dc(dc_),
                owns_handle(true) {
}

template<typename... Args>
ObserverGuard<Args...>::~ObserverGuard() {
    if (owns_handle) {
        dc->lose_interest(dc_handle);
    }
}

template<typename... Args>
ObserverGuard<Args...>::ObserverGuard(ObserverGuard&& other) :
        dc(std::move(other.dc)),
        dc_handle(std::move(other.dc_handle)),
        owns_handle(std::move(other.owns_handle)) {
    other.owns_handle = false;
}


template<typename... Args>
ObserverGuard<Args...> make_observer_guard(
        typename Observation<Args...>::callback_t callback,
        Observation<Args...>* dc) {
    return ObserverGuard<Args...>(callback, dc);
}
