#include "visualizer.h"
#include "dali/utils/core_utils.h"

using utils::MS;
using std::string;
using json11::Json;

DEFINE_string(visualizer_hostname, "127.0.0.1", "Default hostname to be used by visualizer.");
DEFINE_int32(visualizer_port, 6379, "Default port to be used by visualizer.");

namespace visualizable {
    const std::vector<double> FiniteDistribution::empty_vec;
}
Visualizer::Visualizer(std::string my_namespace, std::shared_ptr<redox::Redox> other_rdx) : Visualizer(my_namespace, false, other_rdx) {}

Visualizer::Visualizer(std::string my_namespace, bool rename_if_needed, std::shared_ptr<redox::Redox> other_rdx) :
        my_namespace(my_namespace) {
    if(!other_rdx) {
        // if no redox connection was passed we create
        // one on the fly:
        rdx = std::make_shared<redox::Redox>(std::cout, redox::log::Off);
        // next we try to connect:
        std::cout << "Visualizer connecting to redis "
                  << FLAGS_visualizer_hostname << ":" << FLAGS_visualizer_port << std::endl;
        connected = rdx->connect(FLAGS_visualizer_hostname, FLAGS_visualizer_port);
    } else {
        rdx = other_rdx;
        connected = true;
    }
    if (!connected) {
        throw std::runtime_error("VISUALIZER ERROR: can't connect to redis.");
        return;
    }

    std::cout << "my_namespace = " << this->my_namespace << std::endl;
    std::string namespace_key = MS() << "namespace_" << this->my_namespace;
    auto& key_exists = rdx->commandSync<int>({"EXISTS", namespace_key});
    assert(key_exists.ok());
    bool taken = key_exists.reply() == 1;
    key_exists.free();
    if (taken) {
        if (rename_if_needed) {
            int increment = 1;
            if (!other_rdx) {
                std::cout << "Duplicate Visualizer name : \"" << this->my_namespace
                          << "\". Retrying with \"" << this->my_namespace << "_" << increment << "\"" << std::endl;
            }
            // iterate until a suitable name is found
            while (true) {
                if (increment > 500) {
                    throw Visualizer::duplicate_name_error(MS() << "VISUALIZER ERROR: visualizer name already in use: " << my_namespace);
                }
                namespace_key = MS() << "namespace_" << this->my_namespace << "_" << increment;
                auto& alternate_key_exists = rdx->commandSync<int>({"EXISTS", namespace_key});
                assert(alternate_key_exists.ok());
                taken = alternate_key_exists.reply() == 1;
                alternate_key_exists.free();
                if (taken) {
                    if (!other_rdx) {
                        std::cout << "Duplicate Visualizer name : \"" << this->my_namespace  << "_" << increment
                                  << "\". Retrying with \"" << this->my_namespace << "_" << increment + 1 << "\"" << std::endl;
                    }
                    increment++;
                    continue;
                } else {
                    // make note of new namespace:
                    this->my_namespace = MS() << "_" << increment;
                    break;
                }
            }
        } else {
            // give up immediately on renaming
            // and throw duplicate name error:
            throw Visualizer::duplicate_name_error(MS() << "VISUALIZER ERROR: visualizer name already in use: " << my_namespace);
        }
    }
    // then we ping the visualizer regularly:
    pinging = eq.run_every([this, namespace_key]() {
        if (!connected) return;
        // Expire at 2 seconds
        rdx->command<string>({"SET", namespace_key, "1", "EX", "2"});
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

Visualizer::duplicate_name_error::duplicate_name_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
Visualizer::duplicate_name_error::duplicate_name_error(const char* what_arg) : std::runtime_error(what_arg) {}

