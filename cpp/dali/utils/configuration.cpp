#include "configuration.h"

#include "dali/utils/core_utils.h"

using utils::MS;
using utils::assert2;
using std::shared_ptr;
using std::string;
using utils::in_vector;
using std::make_shared;

namespace conf_internal {
    ConfItem::~ConfItem() {
        // hello!
    }
}

using conf_internal::ConfItem;
using conf_internal::Choice;
using conf_internal::Float;
using conf_internal::Bool;

std::string Conf::ch(std::string name) {
    return get_choice(name)->value;
}

double Conf::f(std::string name) {
    return get_float(name)->value;
}

bool Conf::b(std::string name) {
    return get_bool(name)->value;
}

shared_ptr<Choice> Conf::get_choice(std::string name) {
    return std::static_pointer_cast<Choice>(items[name]);
}
shared_ptr<Float> Conf::get_float(std::string name) {
    return std::static_pointer_cast<Float>(items[name]);
}
shared_ptr<Bool> Conf::get_bool(std::string name) {
    return std::static_pointer_cast<Bool>(items[name]);
}


Conf& Conf::def_choice(std::string name,
                       std::vector<std::string> choices,
                       std::string default_value) {
    assert2(in_vector(choices, default_value),
        MS() << default_value << " is not an option for " << name);
    assert2(choices.size() >= 2,
        MS() << "At least two choices are needed for " << name);
    auto c = make_shared<Choice>();
    c->name = name;
    c->choices = choices;
    c->default_value = default_value;
    c->value = default_value;


    items[name] = c;
    return *this;
}

Conf& Conf::def_float(std::string name,
            double lower_bound,
            double upper_bound,
            double default_value) {
    assert2(lower_bound <= default_value && default_value <= upper_bound,
                MS() << "Default value for " << name << "not in range.");
    auto f = make_shared<Float>();
    f->name = name;
    f->lower_bound = lower_bound;
    f->upper_bound = upper_bound;
    f->default_value = default_value;
    f->value = default_value;

    items[name] = f;
    return *this;
}

Conf& Conf::def_bool(std::string name, bool default_value) {
    auto b = make_shared<Bool>();
    b->default_value = default_value;
    b->value = default_value;
    items[name] = b;
    return *this;
}

namespace std {
    std::string to_string (Choice* choice) {
        return choice->value;
    }

    std::string to_string (Float* ffloat) {
        return to_string(ffloat->value);
    }

    std::string to_string (Bool* bbool) {
        return bbool->value ? "true" : "false";
    }

    std::string to_string (ConfItem* c) {
        if (Choice* choice = dynamic_cast<Choice*>(c)) {
            return to_string(choice);
        } else if (Float* ffloat = dynamic_cast<Float*>(c)) {
            return to_string(ffloat);
        } else if (Bool* bbool = dynamic_cast<Bool*>(c)) {
            return to_string(bbool);
        }
    }

    std::string to_string (const Conf& conf, bool indented) {
        std::stringstream ss;
        ss << "{";
        if (indented) ss << std::endl;
        string name;
        std::shared_ptr<ConfItem> item;
        for (auto kv: conf.items) {
            std::tie(name, item) = kv;
            if (indented) ss << "    ";
            ss << name << ":" << (indented ? " ": "") << to_string(item.get()) << ",";
            if (indented) ss << std::endl;
        }
        ss << "}";
        return ss.str();
    }
}
