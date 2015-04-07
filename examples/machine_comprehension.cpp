#include <tuple>
#include <vector>

#include "dali/data_processing/machine_comprehension.h"

using mc::Section;
using mc::Question;
using std::vector;

vector<Section> training_data, test_data;


int main() {
    std::tie(training_data, test_data) = mc::load();
    training_data[0].print();
}
