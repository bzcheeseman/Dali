#include <iostream>
#include <vector>
#include "dali/data_processing/Glove.h"
using std::vector;

typedef float R;

int main() {
    auto embedding = glove::load<double>( STR(DALI_DATA_DIR) "/glove/test_data.txt");
}
