#ifndef UTILS_GRID_SEARCH_H
#define UTILS_GRID_SEARCH_H

#include <chrono>
#include <functional>
#include <memory>

#include "dali/utils/configuration.h"
#include "dali/mat/model.h"


typedef std::chrono::high_resolution_clock grid_clock_t;


double perturbation_round(const Model& model,
                          double current_objective,
                          std::function<double()> objective);

void perturb_for(grid_clock_t::duration duration,
                 const Model& model,
                 std::function<double()> objective);

void perturbX(int times,
              const Model& model,
              std::function<double()> objective);
#endif
