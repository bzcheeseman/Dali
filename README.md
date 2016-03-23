*If you are new to Dali, you are most likely to enjoy the [Python version](https://github.com/JonathanRaiman/dali-cython).*

# Dali

[![Build Status](https://travis-ci.org/JonathanRaiman/Dali.svg?branch=master)](https://travis-ci.org/JonathanRaiman/Dali)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

An automatic differentiation library that uses reverse-mode differentation (backpropagation) to differentiate recurrent neural networks, or most mathematical expressions through control flow, while loops, recursion.

<img src="https://raw.github.com/JonathanRaiman/Dali/master/misc/salvador.jpg" width="25%" />

This is an reimagination of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/)'s [recurrentJS](http://cs.stanford.edu/people/karpathy/recurrentjs/) ([Github](https://github.com/karpathy/recurrentjs)) in C++. It has similar API names but the backbones are using **MShadow** and C++11's standard library.

@authors **Jonathan Raiman** and **Szymon Sidor**

## Installation

### Homebrew (MAC OS X)

```bash
# with cuda
brew install https://github.com/JonathanRaiman/Dali/releases/download/v1.0.0/dali-gpu.rb
# without cuda
brew install https://github.com/JonathanRaiman/Dali/releases/download/v1.0.0/dali-cpu.rb
```

### Fedora Linux

For 22 or newer replace `yum` with `dnf`.

```bash
# with cuda
sudo yum install https://github.com/JonathanRaiman/Dali/releases/download/v1.0.0/dali-1.0.0-Linux-x86_64-gpu.rpm
# without cuda
sudo yum install https://github.com/JonathanRaiman/Dali/releases/download/v1.0.0/dali-1.0.0-Linux-x86_64-cpu.rpm
```

### Ubuntu (14.04 or newer)

... and maybe some other debians.

```bash
# with cuda
URL='https://github.com/JonathanRaiman/Dali/releases/download/v1.0.0/dali-1.0.0-Linux-x86_64-gpu.deb'; FILE=`mktemp`; wget "$URL" -O $FILE && sudo dpkg -i $FILE; rm $FILE
# without cuda
URL='https://github.com/JonathanRaiman/Dali/releases/download/v1.0.0/dali-1.0.0-Linux-x86_64-cpu.deb'; FILE=`mktemp`; wget "$URL" -O $FILE && sudo dpkg -i $FILE; rm $FILE
```


### Features

* Automatic differentiation
* Broadcasting between matrices and vectors
* Speed (Language model trained using a 2-Layer LSTM processes 25,000 words per second on a Nvidia GTX 780 TI -- vs. [15,000 words per second on Russel Stewart's NLP-Caffe](https://github.com/Russell91/NLPCaffe))
* Clarity of API
* Lazy evaluation of matrix operations
* Hybrid GPU-CPU computation, with **best device for each operation selected at runtime**
* [Visualize Neural Network output in real time](https://github.com/JonathanRaiman/dali-visualizer)

### Why not use Theano?

Theano is a fantastic tensor and automatic differentiation library, with excellent packages for Deep Learning. Unfortunately, it cannot differentiate through control flow, and computation graphs with many nodes and recurrence require long compilation time (this may somewhat change with the arrival of [Josh Schulman's Graph Computation Toolkit](https://github.com/joschu/cgt)). Long compilation times can be alleviated by moving most operations out of scan loops, however this strongly limits expressivity or complicates the code. Finally, because of the separation between the computation and the mathematical description, debugging can be hard.

(Note: [Hypergrad](https://github.com/HIPS/hypergrad/) offers gradient through control flow, but does not match the performance of Theano)

### Why not use Torch?

Torch has excellent community support and a wide variety of packages for Deep Learning, including the popular [NN](https://github.com/torch/nn) and [NN Graph](https://github.com/torch/nngraph) packages, which permit automatic differentiation of Torch Tensors. However, use of these packages requires the definition of `forward` and `backward` passes, [module / param cloning (See the Torch utilities inside Andrej Karpathy's Char-RNN code)](https://github.com/karpathy/char-rnn/blob/master/util/model_utils.lua), pre-allocation of memory when performing recurrence, and the requirement that all parameters be [concatenated when optimizing in Optim](https://github.com/torch/optim), the defacto solver for Torch. Additionally, transfering computation to the GPU demands pre-allocation of memory, which can be problematic in the case of memory-hungry tasks. Finally, running computations in parallel (Hogwild or otherwise) is tricky in Lua / Torch.

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
