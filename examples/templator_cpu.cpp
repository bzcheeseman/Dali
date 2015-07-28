#include <iostream>
#include <vector>
#include "dali/data_processing/Glove.h"
#include "dali/math/TensorInternal.h"
#include "dali/math/LazyTensor.h"

using std::vector;

typedef float R;

int main() {
    auto embedding = glove::load<double>( STR(DALI_DATA_DIR) "/glove/test_data.txt");

    TensorInternal<R,1> t2(mshadow::Shape1(5));
    for (int i=0; i < 5; ++i) {
        t2(i) = i;
    }

    TensorInternal<R,2> out(mshadow::Shape2(3,5));

    out = t2.wrapper().template broadcast<1>(out.shape);
    out.print();


}
