#ifndef DALI_VISUALIZER_VISUALIZER_H
#define DALI_VISUALIZER_VISUALIZER_H

#include <redox.hpp>
#include <string>

#include "dali/visualizer/EventQueue.h"


class Visualizer {
    private:
        std::string my_namespace;
        std::shared_ptr<redox::Redox> rdx;
        EventQueue eq;
        EventQueue::repeating_t pinging;

    public:
        Visualizer(std::string my_namespace, std::shared_ptr<redox::Redox> rdx=nullptr);
};


#endif
