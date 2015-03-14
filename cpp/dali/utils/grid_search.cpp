#include "grid_search.h"

using std::shared_ptr;


double perturbation_round(const Model& model,
                          double current_objective,
                          std::function<double()> objective) {
    Conf& c = model.conf();
    // select random param;
    int param_idx = rand() % c.items.size();
    auto item_kv = c.items.begin();
    std::advance(item_kv, param_idx);
    auto item = item_kv->second;

    if (shared_ptr<Choice> ch = std::dynamic_pointer_cast<Choice>(item)) {
        auto before_perturbation = ch->value;
        // choosing any index except current one;
        // this way of drawing ensures fairness.
        auto value_idx = std::find(ch->choices.begin(), ch->choices.end(), ch->value)
                       - ch->choices.begin();

        auto idx = rand() % (ch->choices.size() - 1);
        if (idx >= value_idx)
            idx += 1;
        ch->value = ch->choices[idx];
        auto obj = objective();
        if (obj < current_objective)
            return obj;
        ch->value = before_perturbation;
    } else if (shared_ptr<Float> f = std::dynamic_pointer_cast<Float>(item)) {
        auto before_perturbation = f->value;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(f->lower_bound, f->upper_bound);
        f->value = dis(gen);
        auto obj = objective();
        if (obj < current_objective)
            return obj;
        f->value = before_perturbation;
    }
    return current_objective;
}
