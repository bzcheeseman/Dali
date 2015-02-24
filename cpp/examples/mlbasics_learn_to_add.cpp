#include <cstring>
#include <gflags/gflags.h>
#include <memory>

#include "core/Mat.h"
#include "core/Graph.h"

typedef std::shared_ptr<Mat<double> > shared_mat;


/*class RNN {
    shared_mat activate(Graph<double> G&, shared_mat input, shared output) {

    }
};*/


int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
    "RNN Kindergarden - Lesson 1 - Learning to add.");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    const int NUM_EXAMPLES = 10;
    const int EXAMPLE_SIZE = 3;
    shared_mat X = std::make_shared<Mat<double> >(NUM_EXAMPLES,
                                                  EXAMPLE_SIZE,
                                                  0.0,
                                                  1.0/EXAMPLE_SIZE);

    shared_mat Y = std::make_shared<Mat<double> >(NUM_EXAMPLES, 1);

    Y->w = X->w.rowwise().sum().matrix();

    X->print();
    Y->print();

    shared_mat W = std::make_shared<Mat<double> >(EXAMPLE_SIZE, 1, -1.0, 1.0);


    Graph<double> G(true);
    shared_mat predY = G.mul(X,W);
    shared_mat error = G.sum(G.square(G.sub(predY, Y)));
    error->grad();
    G.backward();

    //G.
}
