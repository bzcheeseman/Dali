#ifndef DALI_ARRAY_ASSIGNABLE_ARRAY_H
#define DALI_ARRAY_ASSIGNABLE_ARRAY_H

#include <functional>

class Array;

struct AssignableArray {
    typedef std::function<void(Array&)> assign_t;
    assign_t assign_to;
    AssignableArray(assign_t&& _assign_to);
};

#endif
