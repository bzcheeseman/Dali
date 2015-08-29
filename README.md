# Dali

An automatic differentiation library that uses reverse-mode differentation (backpropagation) to differentiate recurrent neural networks, or most mathematical expressions through control flow, while loops, recursion.

<img src="https://raw.github.com/JonathanRaiman/Dali/master/misc/salvador.jpg" width="50%" />

This is an reimagination of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/)'s [recurrentJS](http://cs.stanford.edu/people/karpathy/recurrentjs/) ([Github](https://github.com/karpathy/recurrentjs)) in C++. It has similar API names but the backbones are using **Eigen** and C++11's standard library.

@authors **Jonathan Raiman** and **Szymon Sidor**

### Features

* Automatic differentiation
* Matrix Broadcasting (elementwise multiply, elementwise product)
* Multiple index slicing
* Speed
* Clarity of API
* [Visualize Neural Network output in real time](https://github.com/JonathanRaiman/dali-visualizer)

### Why ?

While there are existing great automatic differentiation libraries, a fast simple **multithreaded** **compile-less** version is missing. Hogwild-style training provides huge speedups in data-parallel model training, and so far Torch and Theano do not provide an accessible way of having this.

## Usage

#### Run a super duper simple example

Create two 3x3 matrices filled with uniform random noise between -2 and 2:

```cpp
Mat<float> A(3,3, weights<float>::uniform(-2.0, 2.0));
Mat<float> B(3,3, weights<float>::uniform(-2.0, 2.0));
```

Now let's multiply them:

```cpp
auto C = A * B;
```

Now's let take the gradient of the squared sum of this operation:

```cpp
auto error = (C ^ 2).sum();
```

And get the gradient of error with respect to A and B:

```cpp
error.grad();
graph::backward();

auto A_gradient = A.dw();
auto B_gradient = B.dw();
```

##### Behind the scenes:

Each matrix has another matrix called `dw` that holds the elementwise gradients for each
matrix. When we multiply the matrices together we create a new output matrix called `C`,
**and** we also add this operation to our computational graph (held by a thread local
variable in `graph::tape`). When we reach `C.sum()` we also add this operation to our graph.

Computing the gradient is done in 2 steps, first we tell our graph what the objective
function is:

```cpp
error.grad();
```

`error` needs to be a scalar (a 1x1 matrix in this implementation) to use `grad()`.
Step 2 is to call `graph::backward()` and go through every operation executed so far
in reverse using `graph::tape`'s record. When we run through the operations backward
we update the gradients of each intermediary object until `A` and `B`'s `dw`s get
updated. Those are now [the gradients we we're looking for](http://youtu.be/DIzAaY2Jm-s?t=3m12s).

#### Run a simple (yet advanced) example

Let's run a simple example. We will use data from [Paul Graham's blog](http://paulgraham.com) to train a language model. This way we can generate random pieces of startup wisdom at will! After about 5-10 minutes of training time you should see it generate sentences that sort of make sense. To do this go to cpp/build and execute

```bash
examples/language_model --flagfile ../flags/language_model_simple.flags
```

That's it. Don't forget to checkout `examples/language_model.cpp`. It's not that scary!

For a more extensive description of usage see the [character model tutorial](docs/CharacterModel.md)

For a funny example where you teach stacked LSTMs about multiplication, substraction, and addition [check this out](docs/Arithmetic.md).

## Installation

Get **[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)** ([Download Link](http://bitbucket.org/eigen/eigen/get/3.2.4.tar.bz2)), **GFlags**, **HiRedis**, **Clang**, and **protobuf**, then head to the `cpp/build` folder and use `cmake` to configure and create the appropriate Makefiles.

You need the latest version of [Clang](http://llvm.org/releases/download.html) (>= 3.6.0).

##### 1. Dependency Installation

###### 1.a on Mac OSX

```bash
brew install eigen
brew install cmake
brew install gflags
HOMEBREW_CC=clang HOMEBREW_CXX=clang++ brew install protobuf
brew install libev
HOMEBREW_CC=clang HOMEBREW_CXX=clang++ brew install hiredis
cmake ..
```

###### 1.b on Fedora Linux

```bash
yum install eigen3-devel
yum install clang
yum install gflags gflags-devel
yum install sqlite-devel
yum install protobuf protobuf-devel protobuf-compiler
yum install libev libev-devel
yum install hiredis hiredis-devel
```

##### 2. Compilation

Then use `cmake` to create the `make` targets, and run `make` to compile the code:

```bash
cmake ..
make -j 9
```

That's it. Now built examples will be stored in `cpp/build/examples`.
For instance a character prediction model using Stacked LSTMs is built under `cpp/build/examples/character_prediction`.

## Tests

To compile and run tests you need [Google Tests](https://code.google.com/p/googletest/). Download it [here](https://code.google.com/p/googletest/downloads/detail?name=gtest-1.7.0.zip).

#### 1. Compile and run tests

From the build folder do the following:

```bash
cmake ..
make -j 9 run_tests
```

###### 2.a Install Gtest on Mac OSX

Homebrew does not offer a way of installing gtest, however in a few steps you can get it running:

```bash
wget https://googletest.googlecode.com/files/gtest-1.7.0.zip
cd gtest-1.7.0
mkdir mybuild
cd mybuild
cmake ..
make -j 9
cp libgtest_main.a /usr/local/lib/libgtest_main.a
cp libgtest.a /usr/local/lib/libgtest.a
cp -R ../include/* /usr/local/include/
cd ../..
rm -rf gtest-1.7.0
```

###### 2.b Install Gtest on Fedora Linux

Using `yum` it's a piece of cake:

```bash
sudo yum install gtest gtest-devel
```

#### Mac OSX compiler issues

Until Apple decides to fully embrace thread_local abstraction we are sadly forced to update our compilers manually (and no replacing with __thread is not enough...). Here are steps for updating your compiler:

```bash
    # Go to http://llvm.org/releases/download.html
    # Download "Clang for OSX" (tarball). Use version
    # 3.6.0 or above
    # Unpack .tar.xz (which will by default be in ~/Downloads)
    tar xf CLANG.tar.xz
    # Then cd into clang and copy to /usr/local:
    cd CLANG
    cp -R ./* /usr/local/
```


#### MKL Zaziness Problems

On Mac OSX, or more generally when using [Intel's gracious MKL Library](https://software.intel.com/en-us/intel-mkl) you may encounter an interesting bug with [`Eigen`](http://eigen.tuxfamily.org/bz/show_bug.cgi?id=874) where `MKL_BLAS` is shown as undefined during compilation.

To fix this bug (feature?) make the modifications listed [here](https://bitbucket.org/eigen/eigen/pull-request/82/fix-for-mkl_blas-not-defined-in-mkl-112/diff) to your Eigen header files and everything should be back to normal.


### Utils

In the utilities namespace you will find several tools to make data processing and saving easier.

To create folders similar to how [os.makedirs](https://docs.python.org/2/library/os.html#os.makedirs) works in Python, you can do:

```cpp
utils::makedirs("folder/subfolder/");
```

Random integer between 0 and 2 (included):

```cpp
utils::randint(0, 2);
```

Check whether a file is gzipped:

```cpp
utils::is_gzip("folder/suspicious.gz");
```

Sort the arguments of a list [np.argsort](http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html) style:

```cpp
auto sorted_lengths = utils::argsort(lengths);
```


### Future steps

* Switch matrix backend from **Eigen** to **[MShadow](https://github.com/dmlc/mshadow)**.
* Add ImageNet, Caffe loading, broader ConvNet support (currently have `conv2d` and `conv1d`, but no pooling)
* Web interface for managing experiments (today [Dali-visualizer](https://github.com/JonathanRaiman/dali-visualizer) only shows progress and sample predictions). 
* Web interface for visualizing network activity.
* Add some mathematical expressions from [Deepmind's Torch Cephes module](http://deepmind.github.io/torch-cephes/).
* Improve hogwild with Natural Gradient Descent
* Distribute training over multiple machines.
* Add **Python** extension

## Additional Notes

### Safety / Memory

Zealous use of `std::shared_ptr` appears to be the way of the future for managing
both the backpropagation; `<functional>`'s lambda functions keep track of previous memory use in other steps, and the overall forward structure of the operations.

### Intel MKL Support

On Mac OSX, or more generally when using [Intel's MKL Library](https://software.intel.com/en-us/intel-mkl) you may encounter an interesting bug with [`Eigen`](http://eigen.tuxfamily.org/bz/show_bug.cgi?id=874) where `MKL_BLAS` is shown as undefined during compilation.

To fix this bug make the modifications listed [here](https://bitbucket.org/eigen/eigen/pull-request/82/fix-for-mkl_blas-not-defined-in-mkl-112/diff) to your Eigen header files and everything should return to normal.

### Debugging Assertion Failures

You can use [gdb](http://www.gnu.org/software/gdb/) to debug assertion failures in **Dali**. The majority of the assertions in Dali use `utils::assert2` instead of the usual `assert` method to provide more informative error messages. It is easy to catch and trace these errors using **gdb**:

```bash
gdb --args example/dali_code.o arg1 arg2
...
catch throw
run
...
backtrace
```

A stack trace for the assertion error should now appear.


### Theme song

*Warning: for development purposes only!*

[![The theme song](https://i.ytimg.com/vi/c7BS4jbA_hw/mqdefault.jpg)](https://www.youtube.com/watch?v=c7BS4jbA_hwA)
