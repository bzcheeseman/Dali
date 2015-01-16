RecurrentJS in C++
------------------

This is an reimagination of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/)'s recurrentjs in C++. It has the same API names and structure (so far), but the backbones are using **Eigen** and C++11's standard library. Callbacks are gone (following the Python implementation of the same idea), leaving a one stop shop for all backprop operations handled by `Backward`.

### Installation
	
Grad hold of the latest copy of **Eigen** (no linking, installation, or nothing, just download and you're ready to go). Then:

	> make

That's it.

### Usage

is dead simple (except for the crazy use of `std::shared_ptr`, that's
because C++ is antsy and really desperately wants to kill these matrices,
on the plus side: free garbage collection ;).

	typedef double REAL_t;

	// build blank matrix of double type:
    std::shared_ptr< Mat<REAL_t>> A(new Mat<REAL_t>(3, 5) );
    // build random matrix of double type with standard deviation 2:
    std::shared_ptr< Mat<REAL_t> > B(new Mat<REAL_t>(A->n, A->d, 2.0));
    std::shared_ptr< Mat<REAL_t> > C(new Mat<REAL_t>(A->d, 4, 2.0));

    A->w = (A->w.array() + 1.2).matrix();

    A->print();
    B->print();

    Graph<REAL_t> graph;
    auto A_plus_B     = graph.add(A, B);
    auto A_times_B    = graph.eltmul(A, B);
    auto A_plus_B_sig = graph.sigmoid(A_plus_B);
    auto A_dot_C      = graph.mul(A, C);

    auto A_dot_C_tanh = graph.tanh(A_dot_C);

    A_plus_B    ->print();
    A_times_B   ->print();
    A_plus_B_sig->print();
    A_dot_C     ->print();

    auto A_plucked = graph.row_pluck(A, 2);
    A_plucked->print();
    forward_model(graph, A, C);

    auto prod = graph.mul(A, C);
    auto activ = graph.tanh(prod);

    // add some random singularity and use exponential
    // normalization:
    A_plucked->w(2,0) += 3.0;
    auto A_plucked_normed = softmax(A_plucked);
    A_plucked_normed->print();

    // backpropagate to A and B
    graph.backward();

Just as in the Javascript / Python versions the backward step takes care of all the necessary backpropagation through a graph.

Safety / Memory
---------------

Zealous use of `std::shared_ptr` appears to be the way of the future for managing
both the backpropagation `Backward` classes that keep track of previous memory use in other steps, and the overall forward structure of the operations.

### Multithreading

One potential area of concern is during multithreaded code where shared_ptr will need to keep track of updates across threads. If this is the case, then perhaps some better bookkeeping will need to happen to ensure that shared memory across threads isn't garbage collected by other threads. Hopefully this won't happen since the `std:shared_ptr`s will only be scoped outside the main threading loop.


Future steps
------------

* Working LSTM example, doing character reading from the same Paul Graham startup wisdom corpus. 
* parallelized code
* Adagrad / Adadelta
