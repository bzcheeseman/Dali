#include <iostream>
#include <vector>

#include "dali/core.h"
#include "dali/utils.h"

using std::vector;
using std::string;

// Test file for LSTM
int main () {

    string tsv_file = STR(DALI_DATA_DIR) "/CoNLL_NER/NER_dummy_dataset.tsv";
    auto dataset = utils::load_tsv(tsv_file, 4, '\t');
    utils::assert2(dataset.size() == 7, "ne");
    utils::assert2(dataset.back().front().front() == ".", "ne");


    typedef double R;
    sane_crashes::activate();
    auto U = weights<R>::uniform(-2.0, 2.0);
    LSTM<R> lstm(30, 50,
        true   // do use Alex Graves' 2013 LSTM
               // where memory connects to gates
    );
    Mat<R> embedding(1000, 30, U);
    auto prev_state = lstm.initial_states();
    Mat<R> hidden, memory;
    std::tie(memory, hidden) = lstm.activate(embedding[{0, 1, 10, 2, 1, 3}], prev_state);
    hidden.print();

    // load numpy matrix from file:
    auto name = "numpy_test.npy";
    std::cout << "loading a numpy matrix \"" << name << "\" from the disk" << std::endl;
    Mat<R> numpy_mat;
    if (utils::file_exists(name)) {
        numpy_mat = Mat<R>(name);
    } else {
        numpy_mat = Mat<R>(3, 3);
        for (int i = 0; i < 9; i++) numpy_mat.w(i) = i;
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

    Mat<R> A(3, 5);
    A += 1.2;
    A = A + Mat<R>(3, 5, weights<R>::uniform(-0.5, 0.5));
    // build random matrix of double type with standard deviation 2:
    Mat<R> B(A.dims(0), A.dims(1), U);
    Mat<R> C(A.dims(1), 4,         U);

    A.print();
    B.print();

    auto A_times_B    = A * B;
    auto A_plus_B_sig = (A+B).sigmoid();
    auto A_dot_C_tanh = A.dot(C).tanh();
    auto A_plucked    = A[2];

    A_times_B.print();
    A_plus_B_sig.print();
    A_dot_C_tanh.print();
    A_plucked.print();

    // add some random singularity and use exponential
    // normalization:
    A_plucked.w(2,0) += 3.0;
    auto A_plucked_normed = MatOps<R>::softmax(A_plucked);
    auto A_plucked_normed_t = MatOps<R>::softmax(A_plucked.T());
    A_plucked_normed.print();
    A_plucked_normed_t.print();

    // backpropagate to A and B
    auto params = lstm.parameters();
    utils::save_matrices(params, "lstm_params");

    StackedInputLayer<R> superclassifier({20, 20, 10, 2}, 5);

    vector<Mat<R>> inputs;
    inputs.emplace_back(20, 5, U);
    inputs.emplace_back(20, 5, U);
    inputs.emplace_back(10, 5, U);
    inputs.emplace_back(2,  5, U);

    auto out2 = superclassifier.activate(inputs);


    auto stacked = MatOps<R>::vstack(inputs);
    auto stacked_a_b = MatOps<R>::vstack(inputs[0], inputs[1]);

    stacked.print();
    stacked_a_b.print();

    out2.print();

    // Now vstacks:
    Mat<R> upper(4, 1, U);
    Mat<R> lower(4, 3, U);
    std::cout << "Stacking \"upper\": " << std::endl;
    upper.print();
    std::cout << "with \"lower\": " << std::endl;
    lower.print();
    std::cout << "using MAT::hstack(\"upper\", \"lower\") :" << std::endl;
    MatOps<R>::hstack(upper, lower).print();

    inputs[0] = MatOps<R>::fill(inputs[0], 3);
    // TODO(jonathan): put in a test.
    // inputs[0].w().row(1).fill(1);
    // inputs[0].w().row(2).fill(2);
    // inputs[0].w().row(3).fill(3);

    Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> bob_indices = Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>::Zero(3,3);

    bob_indices(0,0) = 1;
    bob_indices(1,0) = 2;
    bob_indices(2,0) = 3;

    auto bob = inputs[0][bob_indices.col(0)];

    bob.print();

    auto dropped_bob = MatOps<R>::dropout(bob, 0.2);

    dropped_bob.print();

    auto fast_dropped_bob = MatOps<R>::fast_dropout(bob);

    fast_dropped_bob.print();

    graph::backward();

    std::cout << "Press Enter to continue to Orthogonal Initialization" << std::endl;
    getchar();

    /**
    Orthogonal Initialization:
    Code sourced from:
    https://github.com/IndicoDataSolutions/Passage/blob/fcc4037632721e63e5e2473e577e3e6f367e64f8/passage/inits.py

    def orthogonal(shape, scale=1.1):
        """ benanne lasagne ortho init (faster than qr approach)"""
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return sharedX(scale * q[:shape[0], :shape[1]])
    **/
    std::default_random_engine generator;
    std::normal_distribution<R> distribution(0.0, 1.0);
    std::random_device rd;
    generator.seed(rd());
    auto randn = [&] (int) {return distribution(generator);};
    auto mymat = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(10, 20, randn);
    auto svd = mymat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

    std::cout << "U dimensions = (" << svd.matrixU().cols() << " " << svd.matrixU().rows() << ")" << std::endl;
    std::cout << "V dimensions = (" << svd.matrixV().cols() << " " << svd.matrixV().rows() << ")" << std::endl;

    return 0;
}
