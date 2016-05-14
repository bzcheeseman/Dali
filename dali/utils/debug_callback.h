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

#include "debug_callback-impl.h"

#endif
