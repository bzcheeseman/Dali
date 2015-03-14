#ifndef UTILS_CONFIGURATION_H
#define UTILS_CONFIGURATION_H

#include <memory>
#include <vector>
#include <random>
#include <string>


#include "dali/utils/core_utils.h"

struct ConfItem {
    std::string name;
    // We need at least one virtual function
    // to be considered polymorphic.
    virtual ~ConfItem();
};

struct Choice : public ConfItem {
    std::vector<std::string> choices;
    std::string default_value;

    std::string before_perturbation;
    std::string value;
};

struct Float : public ConfItem {
    double lower_bound;
    double upper_bound;
    double default_value;

    double before_perturbation;
    double value;

};

class Conf {
    public:
        std::unordered_map<std::string, std::shared_ptr<ConfItem>> items;

        std::string ch(std::string name);
        double f(std::string name);

        std::shared_ptr<Choice> get_choice(std::string name);
        std::shared_ptr<Float> get_float(std::string name);

        Conf& def_choice(std::string name,
                         std::vector<std::string> choices,
                         std::string default_value);

        Conf& def_float(std::string name,
                        double lower_bound,
                        double upper_bound,
                        double default_value);
};

namespace std {
    std::string to_string (const Choice* c);
    std::string to_string (const Float* c);
    std::string to_string (const ConfItem* c);
    std::string to_string (const Conf& c, bool indented=false);
}

#endif
