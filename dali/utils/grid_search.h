#ifndef UTILS_GRID_SEARCH_H
#define UTILS_GRID_SEARCH_H

#include <chrono>
#include <functional>
#include <memory>

#include "dali/utils/configuration.h"
#include "dali/mat/model.h"


typedef std::chrono::high_resolution_clock grid_clock_t;


double perturbation_round(Conf& model,
                          double current_objective,
                          std::function<double()> objective);

double perturb_for(grid_clock_t::duration duration,
                   Conf& model,
                   std::function<double()> objective);

double perturbX(int times,
                Conf& model,
                std::function<double()> objective);
#endif
