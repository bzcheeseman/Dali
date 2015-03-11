#include <iostream>
#include <vector>

#include "core/Layers.h"
#include "core/Mat.h"
#include "core/Tape.h"
#include "core/SaneCrashes.cpp"
//#include "core/StackedGatedModel.h"

using std::vector;

// Test file for LSTM
int main () {
    sane_crashes::activate();
    typedef double R;
std::cout << "SIEMA" << std::endl;
    LSTM<R> lstm(30, 50);
std::cout << "SIEMA" << std::endl;

    Mat<R> embedding(1000, 30, 2.0);
    Mat<R> prev_cell(50, 1);
    Mat<R> prev_hidden(50, 1);

    Mat<R> hidden, memory;

    std::tie(memory, hidden) = lstm.activate(embedding.rows_pluck({0, 1, 10, 2, 1, 3}), prev_cell, prev_hidden);
    hidden.print();

        // load numpy matrix from file:

    auto name = "numpy_test.npy";
    std::cout << "loading a numpy matrix \"" << name << "\" from the disk" << std::endl;
    Mat<R> numpy_mat;
    if (utils::file_exists(name)) {
        numpy_mat = Mat<R>(name);
    } else {
        numpy_mat = Mat<R>(3, 3);
        for (int i = 0; i < 9; i++) numpy_mat.w()(i) = i;
        numpy_mat.npy_save(name);
    }
    std::cout << "\"" << name << "\"=" << std::endl;
    // print it
    numpy_mat.print();
    // take softmax
    std::cout << "We now take a softmax of this matrix:" << std::endl;
    auto softmaxed = MatOps<R>::softmax(numpy_mat);
    softmaxed.print();
    uint idx = 2;
    std::cout << "let us now compute the Kullback-Leibler divergence\n"
              << "between each column in this Softmaxed matrix and a\n"
              << "one-hot distribution peaking at index " << idx + 1 << "." << std::endl;

    // print softmax:
    auto divergence = MatOps<R>::cross_entropy(softmaxed, idx);
    divergence.print();

    std::cout << "Press Enter to continue" << std::endl;
    getchar();
    //std::cin.ignore( std::numeric_limits<std::streamsize>::max(), '\n' );
    //std::cin.get();

    Mat<R> A(3, 5);
    A.w() = (A.w().array() + 1.2).matrix();
    // build random matrix of double type with standard deviation 2:
    Mat<R> B(A.dims(0), A.dims(1), 2.0);
    Mat<R> C(A.dims(1), 4,    2.0);

    A.print();
    B.print();

    auto A_times_B    = A * B;
    auto A_plus_B_sig = (A+B).sigmoid();
    auto A_dot_C_tanh = A.dot(C).tanh();
    auto A_plucked    = A.row_pluck(2);

    A_times_B.print();
    A_plus_B_sig.print();
    A_dot_C_tanh.print();
    A_plucked.print();

    // add some random singularity and use exponential
    // normalization:
    A_plucked.w()(2,0) += 3.0;
    auto A_plucked_normed = MatOps<R>::softmax(A_plucked);
    auto A_plucked_normed_t = MatOps<R>::softmax(A_plucked.T());
    A_plucked_normed.print();
    A_plucked_normed_t.print();

    // backpropagate to A and B
    auto params = lstm.parameters();
    utils::save_matrices(params, "lstm_params");

    StackedInputLayer<R> superclassifier({20, 20, 10, 2}, 5);

    vector<Mat<R>> inputs;
    inputs.emplace_back(20, 5, -2.0, 2.0);
    inputs.emplace_back(20, 5, -2.0, 2.0);
    inputs.emplace_back(10, 5, -2.0, 2.0);
    inputs.emplace_back(2,  5, -2.0, 2.0);

    auto out2 = superclassifier.activate(inputs);


    auto stacked = MatOps<R>::vstack(inputs);
    auto stacked_a_b = MatOps<R>::vstack(inputs[0], inputs[1]);

    stacked.print();
    stacked_a_b.print();

    out2.print();

    // Now vstacks:
    Mat<R> upper(4, 1, -2.0, 2.0);
    Mat<R> lower(4, 3, -2.0, 2.0);
    std::cout << "Stacking \"upper\": " << std::endl;
    upper.print();
    std::cout << "with \"lower\": " << std::endl;
    lower.print();
    std::cout << "using MAT::hstack(\"upper\", \"lower\") :" << std::endl;
    MatOps<R>::hstack(upper, lower).print();

    inputs[0].w().fill(3);
    inputs[0].w().row(1).fill(1);
    inputs[0].w().row(2).fill(2);
    inputs[0].w().row(3).fill(3);

    Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> bob_indices = Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>::Zero(3,3);

    bob_indices(0,0) = 1;
    bob_indices(1,0) = 2;
    bob_indices(2,0) = 3;

    auto bob = inputs[0].rows_pluck(bob_indices.col(0) );

    bob.print();

    auto dropped_bob = MatOps<R>::dropout(bob, 0.2);

    dropped_bob.print();

    auto fast_dropped_bob = MatOps<R>::fast_dropout(bob);

    fast_dropped_bob.print();

    graph::backward();
/*
    auto some_model = StackedGatedModel<R>(20, 10, 20, 2, 1, false, 0.3);

    some_model.save("some_model");


    auto loaded_model = StackedGatedModel<R>::load("some_model");

    auto some_model2 = StackedModel<R>(20, 10, 20, 2, 1, false);

    some_model.save("some_model");

    auto loaded_model2 = StackedModel<R>::load("some_model");
*/
    return 0;
}
