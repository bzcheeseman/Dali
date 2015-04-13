#include "visualizer.h"
#include "dali/utils/core_utils.h"

using utils::MS;
using std::string;
using json11::Json;

Visualizer::Visualizer(std::string my_namespace, std::shared_ptr<redox::Redox> other_rdx):
        my_namespace(my_namespace) {
    if(!other_rdx) {
        // if no redox connection was passed we create
        // one on the fly:
        rdx = std::make_shared<redox::Redox>(std::cout, redox::log::Off);
        // next we try to connect:
        connected = rdx->connect();
    } else {
        rdx = other_rdx;
        connected = true;
    }
    if (!connected) {
        std::cout << "WARNING: Visualizer off (can't connect to redis)" << std::endl;
    }
    // then we ping the visualizer regularly:
    pinging = eq.run_every([this]() {
        if (!connected) return;
        // Expire at 2 seconds
        rdx->command<string>({"SET", MS() << "namespace_" << this->my_namespace, "1", "EX", "2"});
    }, std::chrono::seconds(1));
}

void Visualizer::feed(const json11::Json& obj) {
    if (!connected) return;
    rdx->publish(MS() << "feed_" << my_namespace, obj.dump());
}

void Visualizer::feed(const std::string& str) {
    Json str_as_json = Json::object {
        { "type", "report" },
        { "data", str },
    };
    feed(str_as_json);
}
