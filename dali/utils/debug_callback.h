#ifndef DALI_ARRAY_DEBUG_CALLBACKS_H
#define DALI_ARRAY_DEBUG_CALLBACKS_H

#include <list>
#include <functional>

template<typename... Args>
struct DebugCallback {
    typedef std::function<void(Args...)>    callback_t;
    typedef typename std::list<callback_t>::iterator callback_handle_t;


    callback_handle_t register_callback(callback_t callback);
    void deregister_callback(callback_handle_t handle);
    int activate(Args...);
  private:
    std::list<callback_t> callbacks;
};

template<typename... Args>
struct ScopedCallback {
    DebugCallback<Args...>* dc;
    typename DebugCallback<Args...>::callback_handle_t dc_handle;

    ScopedCallback(typename DebugCallback<Args...>::callback_t callback,
                   DebugCallback<Args...>* dc_);

    ~ScopedCallback();

    ScopedCallback(ScopedCallback&&);
    ScopedCallback(const ScopedCallback&) = delete;
    ScopedCallback& operator=(const ScopedCallback&) = delete;
};

template<typename... Args>
ScopedCallback<Args...> make_scoped_callback(
        typename DebugCallback<Args...>::callback_t callback,
        DebugCallback<Args...>* dc);


#include "debug_callback-impl.h"

#endif
