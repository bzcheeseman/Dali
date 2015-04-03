RecurrentJS in C++
------------------

This is an reimagination of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/)'s [recurrentJS](http://cs.stanford.edu/people/karpathy/recurrentjs/) ([Github](https://github.com/karpathy/recurrentjs)) in C++. It has similar API names but the backbones are using **Eigen** and C++11's standard library. This automatic differentiation library uses reverse-mode differentation (backpropgation) which allows it to differentiate mathematical expressions but also through control flow, while loops, recursion.

@authors **Jonathan Raiman** and **Szymon Sidor**

### Features

* Automatic differentiation
* Matrix Broadcasting (elementwise multiply, elementwise product)
* Multiple index slicing
* Speed
* Clarity of API

### Why ?

While there are existing great automatic differentiation libraries, a fast and no-compile version is lacking. In particular recurrentJS makes great use of callbacks and garbage collection to enable backprop through time. In this implementation the goal is to reduce reliance on these abstractions and have a simple backprop step class.

In Python use of a specialized `Backward` class wraps backpropagation steps, while C++ uses the `<functional>` lambda functions but this time garbage collection and tracking is done using `C++11`'s excellent `std::shared_ptr`.

## Usage

#### Run a super duper simple example

Create two 3x3 matrices filled with uniform random noise between -2 and 2:

    Mat<float> A(3,3, -2.0, 2.0);
    Mat<float> B(3,3, -2.0, 2.0);

Now let's multiply them:

    auto C = A * B;

Now's let take the gradient of the squared sum of this operation:

    auto error = (C ^ 2).sum();

And get the gradient of error with respect to A and B:

    error.grad();
    graph::backward();

    auto A_gradient = A.dw();
    auto B_gradient = B.dw();


##### Behind the scenes:

Each matrix has another matrix called `dw` that holds the elementwise gradients for each
matrix. When we multiply the matrices together we create a new output matrix called `C`,
**and** we also add this operation to our computational graph (held by a thread local
variable in `graph::tape`). When we reach `C.sum()` we also add this operation to our graph.

Computing the gradient is done in 2 steps, first we tell our graph what the objective
function is:

    error.grad();

`error` needs to be a scalar (a 1x1 matrix in this implementation) to use `grad()`.
Step 2 is to call `graph::backward()` and go through every operation executed so far
in reverse using `graph::tape`'s record. When we run through the operations backward
we update the gradients of each intermediary object until `A` and `B`'s `dw`s get
updated. Those are now [the gradients we we're looking for](http://youtu.be/DIzAaY2Jm-s?t=3m12s).

#### Run a simple (yet advanced) example

Let's run a simple example. We will use data from [Paul Graham's blog](http://paulgraham.com) to train a language model. This way we can generate random pieces of startup wisdom at will! After about 5-10 minutes of training time you should see it generate sentences that sort of make sense. To do this go to cpp/build and execute

    examples/language_model --flagfile ../flags/language_model_simple.flags

That's it. Don't forget to checkout `examples/language_model.cpp`. It's not that scary!

## Installation

Get **[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)** ([Download Link](http://bitbucket.org/eigen/eigen/get/3.2.4.tar.bz2)), **Clang**, and **protobuf**, then head to the `cpp/build` folder and use `cmake` to configure and create the appropriate Makefiles.

You need the latest version of [Clang](http://llvm.org/releases/download.html) (>= 3.6.0).

    > brew install eigen
    > brew install cmake
    > HOMEBREW_CC=clang HOMEBREW_CXX=clang++ brew install protobuf
    > cmake ..


The run `make` to compile the code:


    > make -j 9


That's it. Now built examples will be stored in `cpp/build/examples`.
For instance a character prediction model using Stacked LSTMs is built under `cpp/build/examples/character_prediction`.

## Tests

To compile and run tests you need [Google Tests](https://code.google.com/p/googletest/). Download it [here](https://code.google.com/p/googletest/downloads/detail?name=gtest-1.7.0.zip).

#### 1. Compile and run tests

From the build folder do the following:

    cmake ..
    make -j 9 run_tests

###### 2.a Install Gtest on Mac OSX

Homebrew does not offer a way of installing gtest, however in a few steps you can get it running. First go to the directory where you downloaded Gtests:

    cd gtest-1.7.0
    mkdir mybuild
    cd mybuild
    cmake ..
    cp libgtest_main.a /usr/local/lib/libgtest_main.a
    cp libgtest.a /usr/local/lib/libgtest.a
    cp -R ../include/* /usr/local/include/

Now cmake should be able to find gtest (go back to step 1).

###### 2.b Install Gtest on Fedora Linux

Using `yum` it's a piece of cake:

    sudo yum install gtest gtest-devel

#### MKL Zaziness Problems

On Mac OSX, or more generally when using [Intel's gracious MKL Library](https://software.intel.com/en-us/intel-mkl) you may encounter an interesting bug with [`Eigen`](http://eigen.tuxfamily.org/bz/show_bug.cgi?id=874) where `MKL_BLAS` is shown as undefined during compilation.

To fix this bug (feature?) make the modifications listed [here](https://bitbucket.org/eigen/eigen/pull-request/82/fix-for-mkl_blas-not-defined-in-mkl-112/diff) to your Eigen header files and everything should be back to normal.


### Utils

In the utilities namespace you will find several tools to make data processing and saving easier.

To create folders similar to how `os.makedirs` works in Python, you can do:


    utils::makedirs("folder/subfolder/");

Random integer between 0 and 2 (included):


    utils::randint(0, 2);

Check whether a file is gzipped:


    utils::is_gzip("folder/suspicious.gz");

Sort the arguments of a list `np.argsort` style:


    auto sorted_lengths = utils::argsort(lengths);


### Future steps

* Switching matrix backend from **Eigen** to **[MatrixShadow](https://github.com/dmlc/mshadow)**.
* Adding ImageNet and broader convolutional network support (currently supports `conv2d` and `conv1d`, but no pooling yet)
* Web interface for visualization of progress and reporting.

## Additional Notes

### Safety / Memory

Zealous use of `std::shared_ptr` appears to be the way of the future for managing
both the backpropagation; `<functional>`'s lambda functions keep track of previous memory use in other steps, and the overall forward structure of the operations.

### MKL Zaziness Problems

On Mac OSX, or more generally when using [Intel's gracious MKL Library](https://software.intel.com/en-us/intel-mkl) you may encounter an interesting bug with [`Eigen`](http://eigen.tuxfamily.org/bz/show_bug.cgi?id=874) where `MKL_BLAS` is shown as undefined during compilation.

To fix this bug (feature?) make the modifications listed [here](https://bitbucket.org/eigen/eigen/pull-request/82/fix-for-mkl_blas-not-defined-in-mkl-112/diff) to your Eigen header files and everything should be back to normal.
