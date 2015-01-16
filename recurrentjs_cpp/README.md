RecurrentJS in C++
------------------

This is an reimagination of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/)'s recurrentjs in C++. It has the same API names and structure (so far), but the backbones are using **Eigen** and C++11's standard library. Callbacks are gone (following the Python implementation of the same idea), leaving a one stop shop for all backprop operations handled by `Backward`.

### Installation
	
Grad hold of the latest copy of **Eigen** (no linking, installation, or nothing, just download and you're ready to go). Then:

	> make

That's it.

### Usage

is dead simple:

	typedef double REAL_t;

	// build blank matrix of double type:
    Mat<REAL_t> A(3, 5);
    A.w = (A.w.array() + 1.2).matrix();

    // build random matrix of double type with standard deviation 2:
    Mat<REAL_t> B = Mat<REAL_t>::RandMat(A.n, A.d, 2.0);
    Mat<REAL_t> C = Mat<REAL_t>::RandMat(A.d, 4, 2.0);

    A.print();
    B.print();

    Graph<REAL_t> graph;
	Mat<REAL_t> A_plus_B     = graph.add(A, B);
	Mat<REAL_t> A_times_B    = graph.eltmul(A, B);
	Mat<REAL_t> A_plus_B_sig = graph.sigmoid(A_plus_B);
	Mat<REAL_t> A_dot_C      = graph.mul(A, C);

    Mat<REAL_t> A_dot_C_tanh = graph.tanh(A_dot_C);

    A_plus_B    .print();
    A_times_B   .print();
    A_plus_B_sig.print();
    A_dot_C     .print();

    Mat<REAL_t> A_plucked = graph.row_pluck(A, 2);
    A_plucked.print();

    // add some random singularity and use exponential
    // normalization:
    A_plucked.w(2,0) += 3.0;
    Mat<REAL_t> A_plucked_normed = softmax(A_plucked);
    A_plucked_normed.print();

    // backpropagate to A and B
    graph.backward();

Just as in the Javascript / Python versions the backward step takes care of all the necessary backpropagation through a graph.

Safety / Memory
---------------

In the future used of shared pointers might be a better way of moving matrices around, but until then it appears that no segfaults have occured during the runtime. There is possibly an issue with Backward classes holding on to matrices that the original scope removed. If this is the case then shared pointers should be introduced inside backwards to notify of the ownership of this memory.

Future steps
------------

* Working LSTM example, doing character reading from the same Paul Graham startup wisdom corpus. 
* parallelized code
* Adagrad / Adadelta
