#ifndef DALI_ARRAY_DEBUG_CALLBACKS_H
#define DALI_ARRAY_DEBUG_CALLBACKS_H

#include <list>
#include <functional>

template<typename... Args>
struct Observation {
    typedef std::function<void(Args...)>    callback_t;
    typedef typename std::list<callback_t>::iterator callback_handle_t;


    callback_handle_t observe(callback_t callback);
    void lose_interest(callback_handle_t handle);
    int notify(Args...);
  private:
    std::list<callback_t> callbacks;
};

template<typename... Args>
struct ObserverGuard {
    Observation<Args...>* dc;
    typename Observation<Args...>::callback_handle_t dc_handle;

    ObserverGuard(typename Observation<Args...>::callback_t callback,
                   Observation<Args...>* dc_);

    ~ObserverGuard();

    ObserverGuard(ObserverGuard&&);
    ObserverGuard(const ObserverGuard&) = delete;
    ObserverGuard& operator=(const ObserverGuard&) = delete;
  private:
    bool owns_handle;
};

template<typename... Args>
ObserverGuard<Args...> make_observer_guard(
        typename Observation<Args...>::callback_t callback,
        Observation<Args...>* dc);


#include "debug_callback-impl.h"

#endif
