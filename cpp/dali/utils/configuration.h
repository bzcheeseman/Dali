#ifndef UTILS_CONFIGURATION_H
#define UTILS_CONFIGURATION_H

#include <memory>
#include <vector>
#include <random>
#include <string>


#include "dali/utils/core_utils.h"

namespace conf_internal {
    struct ConfItem {
        std::string name;
        // We need at least one virtual function
        // to be considered polymorphic.
        virtual ~ConfItem();
    };

    struct Choice : public ConfItem {
        std::vector<std::string> choices;
        std::string default_value;

        std::string value;
    };

    struct Float : public ConfItem {
        double lower_bound;
        double upper_bound;
        double default_value;

        double value;
    };

    struct Bool : public ConfItem {
        bool default_value;
        bool value;
    };

    class Conf {
        public:
            std::unordered_map<std::string, std::shared_ptr<ConfItem>> items;

            std::string ch(std::string name);
            double f(std::string name);
            bool b(std::string name);

            std::shared_ptr<Choice> get_choice(std::string name);
            std::shared_ptr<Float> get_float(std::string name);
            std::shared_ptr<Bool> get_bool(std::string name);

            Conf& def_choice(std::string name,
                             std::vector<std::string> choices,
                             std::string default_value);

            Conf& def_float(std::string name,
                            double lower_bound,
                            double upper_bound,
                            double default_value);

            Conf& def_bool(std::string name, bool default_value);
    };
}

typedef conf_internal::Conf Conf;

namespace std {
    std::string to_string (const conf_internal::Choice* c);
    std::string to_string (const conf_internal::Float* c);
    std::string to_string (const conf_internal::Bool* c);
    std::string to_string (const conf_internal::ConfItem* c);
    std::string to_string (const Conf& c, bool indented=false);
}

#endif
