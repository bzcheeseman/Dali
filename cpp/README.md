RecurrentJS in C++
------------------

This is an reimagination of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/)'s recurrentjs in C++. It has the same API names and structure (so far), but the backbones are using **Eigen** and C++11's standard library. Callbacks are gone (following the Python implementation of the same idea), leaving a one stop shop for all backprop operations handled by `<functional>`'s lambda (anonymous) functions.


### What is this readme about?

This readme contains technical details about the internal implementation of the C++ code. For high level overview checkout the readme one level above.


### Usage

is dead simple (except for the crazy use of `std::shared_ptr`, that's
because C++ is antsy and really desperately wants to kill these matrices,
on the plus side: free garbage collection ;).

	typedef double REAL_t;
    typedef Mat<REAL_t> mat;

	auto A = std::make_shared<mat>(3, 5);
    A->w = (A->w.array() + 1.2).matrix();
    // build random matrix of double type with standard deviation 2:
    auto B = std::make_shared<mat>(A->n, A->d, 2.0);
    auto C = std::make_shared<mat>(A->d, 4,    2.0);

    A->print();
    B->print();

    Graph<REAL_t> graph;
    auto A_times_B    = graph.eltmul(A, B);
    auto A_plus_B_sig = graph.sigmoid(graph.add(A, B));
    auto A_dot_C_tanh = graph.tanh( graph.mul(A, C) );
    auto A_plucked    = graph.row_pluck(A, 2);

    A_times_B   ->print();
    A_plus_B_sig->print();
    A_dot_C_tanh->print();
    A_plucked   ->print();

    forward_model(graph, A, C);

    auto prod  = graph.mul(A, C);
    auto activ = graph.tanh(prod);

    // add some random singularity and use exponential
    // normalization:
    A_plucked->w(2,0) += 3.0;
    auto A_plucked_normed = softmax(A_plucked);
    A_plucked_normed->print();

    // backpropagate to A and B
    graph.backward();
    return 0;

Just as in the Javascript / Python versions the backward step takes care of all the necessary backpropagation through a graph.


#### Utils

In the utilities namespace you will find several tools to make data processing and saving easier.

To create folders similar to how `os.makedirs` works in Python, you can do:


    utils::makedirs("folder/subfolder/");

Random integer between 0 and 2 (included):


    utils::randint(0, 2);

Check whether a file is gzipped:


    utils::is_gzip("folder/suspicious.gz");

Sort the arguments of a list `np.argsort` style:


    auto sorted_lengths = utils::argsort(lengths);



Linking
-------

To fix linking issues follow the steps in the shell script:

    sh ./fix_dylib.sh

This uses `install_name_tool` to [cure the sickness from C++](http://stackoverflow.com/questions/23777191/dyld-library-not-loaded-when-trying-to-run-fortran-executable-from-objective-c).

Safety / Memory
---------------

Zealous use of `std::shared_ptr` appears to be the way of the future for managing
both the backpropagation; `<functional>`'s lambda functions keep track of previous memory use in other steps, and the overall forward structure of the operations.

### Multithreading

One potential area of concern is during multithreaded code where shared_ptr will need to keep track of updates across threads. If this is the case, then perhaps some better bookkeeping will need to happen to ensure that shared memory across threads isn't garbage collected by other threads. Hopefully this won't happen since the `std:shared_ptr`s will only be scoped outside the main threading loop.

MKL Zaziness Problems
---------------------

On Mac OSX, or more generally when using [Intel's gracious MKL Library](https://software.intel.com/en-us/intel-mkl) you may encounter an interesting bug with [`Eigen`](http://eigen.tuxfamily.org/bz/show_bug.cgi?id=874) where `MKL_BLAS` is shown as undefined during compilation.

To fix this bug (feature?) make the modifications listed [here](https://bitbucket.org/eigen/eigen/pull-request/82/fix-for-mkl_blas-not-defined-in-mkl-112/diff) to your Eigen header files and everything should be back to normal.


Future steps
------------

* Who knows ?
