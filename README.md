# Dali

[![Build Status](https://travis-ci.org/dali-ml/Dali.svg?branch=master)](https://travis-ci.org/dali-ml/Dali)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Dali is a numerical computation library that supports imperative execution of computation graphs. Code can be compiled on the fly to adapt to general program specific accelerated code using vector instructions and CUDA kernels.
Dali's automatic differentiation library allows it to differentiate arbitrary computation including RNNs, Convnets, and most mathematical expressions through control flow, while loops, recursion.

The API mimics Numpy's naming and documentation conventions.

Dali relies on several key features:

- [x] high-level API that mimics Numpy's naming and documentation conventions.
- [x] automatic differentiation with an imperative style, allowing for arbitrary differentiation
- [x] a modular set of Expressions that can be combined
- [x] JIT-able Expressions allowing for runtime code-generation, optimization, and execution
- [x] kernel fusion
- [x] automatic data placement onto GPUs (heuristic-based for now)
- [x] temporary subexpression removal
- [x] on the fly code rewriting

Several of the features above can be extended to collect runtime information or allow richer replacements, in particular:

- [ ] smart kernel fusion using timing information
- [ ] code rewriting over multiple nodes (`LSTM` -> `CuDNN LSTM`, `exp(x)/sum(exp(x))` -> `softmax(x)`, etc.)
- [ ] data placement based on observed or known movement time and device contention

(Note: Dali is currently being rewritten to incorporate this new dynamic backend)


### Code Generation

To illustrate the new behavior, consider writing a softmax naively:

```
auto exped = op::exp(a - op::max(a, {-1}, true));
auto fused_softmax = exped / op::sum(exped, {-1}, true);
```

This operation creates on-the-fly an efficient [kernel](https://gist.github.com/JonathanRaiman/8c5bd046823f66b97e2944e571e45d78#file-softmax-v2-cu).

<img src="https://raw.github.com/dali-ml/Dali/master/misc/salvador.jpg" width="25%" />

[![Jonathan Raiman, author](https://img.shields.io/badge/Author-Jonathan%20Raiman%20-blue.svg)](https://github.com/JonathanRaiman/) [![Szymon Sidor, author](https://img.shields.io/badge/Author-Szymon%20Sidor%20-blue.svg)](https://github.com/nivwusquorum)

## Installation

### Mac OSX and Fedora

```
cd build
cmake ..
make -j9
```

#### Troubleshooting NVCC compatibility with Clang

On mac the presence of an nvcc compatible clang can sometimes pose problems. You can fix these issues by [downloading Apple's Command Line Tools from Xcode 7.3](https://download.developer.apple.com/Developer_Tools/Command_Line_Tools_OS_X_10.11_for_Xcode_7.3.1/Command_Line_Tools_OS_X_10.11_for_Xcode_7.3.1.dmg), installing them, and then running:

```bash
sudo xcode-select --switch /Library/Developer/CommandLineTools
```

