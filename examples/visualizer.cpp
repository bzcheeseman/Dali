#include <iostream>
#include <redox.hpp>
#include <chrono>
#include <thread>
#include <json11.hpp>
#include <sole.hpp>

#include "dali/visualizer/visualizer.h"

using json11::Json;

int main(int argc, char* argv[]) {
    Visualizer v("vis_demo");

    std::cout << sole::uuid4().str() << std::endl;
    std::cout << sole::uuid4().str() << std::endl;
    std::cout << sole::uuid4().str() << std::endl;
    std::cout << sole::uuid4().str() << std::endl;

    Json my_json = Json::object {
        { "key1", "value1" },
        { "key2", false },
        { "key3", Json::array { 1, 2, 3 } },
    };
    std::string json_str = my_json.dump();
    std::cout << json_str << std::endl;
    v.feed(std::string("o hai"));
    v.feed(my_json);

    std::this_thread::sleep_for(std::chrono::seconds(3));
}
