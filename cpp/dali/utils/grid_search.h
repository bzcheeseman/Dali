#ifndef UTILS_GRID_SEARCH_H
#define UTILS_GRID_SEARCH_H

#include <functional>
#include <memory>

#include "dali/utils/configuration.h"
#include "dali/mat/model.h"



double perturbation_round(const Model& model,
                          double current_objective,
                          std::function<double()> objective);


#endif
