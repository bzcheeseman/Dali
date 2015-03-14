#include <iostream>
#include <memory>

#include "dali/mat/model.h"
#include "dali/utils/configuration.h"
#include "dali/utils/grid_search.h"
#include "dali/utils/core_utils.h"
#include "dali/utils/SaneCrashes.h"

using std::shared_ptr;
using std::make_shared;

class Siema : public Model {
    public:
        virtual shared_ptr<Conf> default_conf() const {
            shared_ptr<Conf> conf = make_shared<Conf>();
            conf->def_choice("size", {"large", "small"}, "small");
            conf->def_float("lol", 0.0, 1.0, 0.0);
            return conf;
        }

        virtual double activate() const {
            if (conf().ch("size") == "small") {
                return conf().f("lol");
            } else if (conf().ch("size") == "large") {
                return conf().f("lol") + 10.0;
            }
        }
};

double model_performance(const Siema& s) {
    double pred = s.activate();
    double target = 10.3;
    return (pred - target) * (pred - target);
}

int main() {
    sane_crashes::activate();
    Siema siema;
    std::cout << "Default configuration values:" << std::endl;
    std::cout << std::to_string(siema.conf()) << std::endl;
    std::cout << "Commencing grid search" << std::endl;
    for (int i=0; i<10; ++i) {
        double obj = perturbation_round(siema, model_performance(siema),
                                        [&siema]() { return model_performance(siema); });
        std::cout << "Objective after round " << i+1 << " is " << obj <<std::endl;
        std::cout << std::to_string(siema.conf()) << std::endl;
    }

}
