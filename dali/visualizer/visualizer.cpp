#include "visualizer.h"


#include "dali/utils/core_utils.h"

using utils::MS;
using std::string;

Visualizer::Visualizer(std::string my_namespace, std::shared_ptr<redox::Redox> rdx):
        my_namespace(my_namespace),
        rdx(rdx) {
    if(!rdx) {
        rdx = std::make_shared<redox::Redox>();
    }
    if (!rdx->connect()) {
        throw string(MS() << "Can't connect to redis server");
    }
    pinging = eq.run_every([this, rdx]() {
        // Expire at 2 seconds
        rdx->command<string>({"SET", MS() << "namespace_" << this->my_namespace, "1", "EX", "2"});
    }, std::chrono::seconds(1));
}
