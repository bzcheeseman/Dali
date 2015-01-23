RecurrentJS in C++
------------------

This is an reimagination of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/)'s recurrentjs in C++. It has the same API names and structure (so far), but the backbones are using **Eigen** and C++11's standard library. Callbacks are gone (following the Python implementation of the same idea), leaving a one stop shop for all backprop operations handled by `Backward`.


### Features

* Automatic differentiation
* Matrix Broadcasting (elementwise multiply, elementwise product)
* Multiple index slicing
* Speed
* Clarity of API

### Installation
	
Grad hold of the latest copy of **[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)** ([Download Link](http://bitbucket.org/eigen/eigen/get/3.2.4.tar.bz2)) Place the downloaded Eigen header folder in the `cpp` directory of this repo. Then:

	> make

That's it.

Or for optimizations turned on (slower compilation 3x result):

    > make optimized

and to do character recognition as was done in Javascript:

    > make character_predict

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

Safety / Memory
---------------

Zealous use of `std::shared_ptr` appears to be the way of the future for managing
both the backpropagation `Backward` classes that keep track of previous memory use in other steps, and the overall forward structure of the operations.

### Multithreading

One potential area of concern is during multithreaded code where shared_ptr will need to keep track of updates across threads. If this is the case, then perhaps some better bookkeeping will need to happen to ensure that shared memory across threads isn't garbage collected by other threads. Hopefully this won't happen since the `std:shared_ptr`s will only be scoped outside the main threading loop.

Future steps
------------

* parallelized code
* Adagrad / Adadelta
