#ifndef CORE_NEW_GRAPH_H
#define CORE_NEW_GRAPH_H

#include <functional>
#include <vector>

namespace graph {
    void emplace_back(std::function<void()>&& f);

    void backward();

    extern thread_local bool backprop_enabled;

    class Tape {
        public:
            std::vector<std::function<void()>>  backprop;

            void backward ();
    };

    extern thread_local Tape tape;

    class NoBackprop {
        private:
            const bool old_value;
        public:
            NoBackprop();
            ~NoBackprop();
    };
}

#endif
