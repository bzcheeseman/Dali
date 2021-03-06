#include "scope.h"

#include "dali/utils/assert2.h"

using namespace std::placeholders;

Observation<Scope::name_t> Scope::obs_enter;
Observation<Scope::name_t> Scope::obs_exit;

Scope::Scope(name_t name_) : name(name_) {
    obs_enter.notify(name);
}

Scope::~Scope() {
    obs_exit.notify(name);
}

bool Scope::has_observers() {
    return obs_enter.num_observers() > 0 && obs_exit.num_observers() > 0;
}

ScopeObserver::ScopeObserver(callback_t on_enter_, callback_t on_exit_) :
        on_enter(on_enter_),
        on_exit(on_exit_),
        enter_guard(std::bind(&ScopeObserver::on_enter_wrapper, this, _1),
                    &Scope::obs_enter),
        exit_guard(std::bind(&ScopeObserver::on_exit_wrapper, this, _1),
                    &Scope::obs_exit) {
}

void ScopeObserver::on_enter_wrapper(Scope::name_t name) {
    state.trace.emplace_back(name);
    if ((bool)on_enter) {
        on_enter(state);
    }
}

void ScopeObserver::on_exit_wrapper(Scope::name_t name) {
    if ((bool)on_exit) {
        on_exit(state);
    }
    ASSERT2(*(state.trace.back()) == *name,
            "Scope exit called out of order.");
    state.trace.pop_back();
}
