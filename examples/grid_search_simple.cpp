#include <chrono>
#include <iostream>
#include <memory>

#include "dali/layers/model.h"
#include "dali/utils/configuration.h"
#include "dali/utils/grid_search.h"
#include "dali/utils/core_utils.h"
#include "dali/utils/SaneCrashes.h"

using std::shared_ptr;
using std::make_shared;
using std::chrono::seconds;

class Siema : public Model {
    public:
        Siema() : Model(default_conf()) {
        }
        virtual Conf default_conf() const {
            Conf conf;
            // Size can be small or large (small by default)
            conf.def_choice("size", {"large", "small"}, "small");
            // lol can be between 0 and 1 (0 by default).
            conf.def_float("lol", 0.0, 1.0, 0.0);
            // negative is a logical value (true by default)
            conf.def_bool("add5", false);
            return conf;
        }

        // SMALL: (lol)
        // LARGE: (lol+5) if add5 else lol
        virtual double activate() const {
            double extra = c().b("add5") ? 5.0 : 0.0;
            if (c().ch("size") == "small") {
                return c().f("lol") + extra;
            } else if (c().ch("size") == "large") {
                return (6*c().f("lol") + extra);
            }
        }
};

// how close are we to 10.3
double model_performance(const Siema& s) {
    double pred = s.activate();
    double target = 10.3;
    return (pred - target) * (pred - target);
}

int main() {
    sane_crashes::activate();
    Siema siema;
    std::cout << "Initial objective " << model_performance(siema)
              << " achieved by " << std::to_string(siema.c()) << std::endl;
    perturb_for(seconds(1), siema.c(), [&siema]() { return model_performance(siema); });
    // perturbX(4, siema, [&siema]() { return model_performance(siema); });

    std::cout << "Optimized objective " << model_performance(siema)
              << " achieved by " << std::to_string(siema.c()) << std::endl;
}
