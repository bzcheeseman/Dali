# Developer Guide



## Installing from source

Get **GFlags**, **HiRedis**, **Clang**, and **protobuf**, then head to the `build` folder and use `cmake` to configure and create the appropriate Makefiles.

You need the latest version of [Clang](http://llvm.org/releases/download.html) (>= 3.6.0).

##### 1. Dependency Installation

###### 1.a on Mac OSX

```bash
brew install openblas
brew link openblas --force
brew brew install llvm --with-clang
brew install cmake
brew install gflags
HOMEBREW_CC=clang HOMEBREW_CXX=clang++ brew install protobuf
brew install libev
HOMEBREW_CC=clang HOMEBREW_CXX=clang++ brew install hiredis
cmake ..
```

###### 1.b on Fedora Linux

```bash
yum install make cmake
yum install blas blas-devel
yum install openblas openblas-devel
yum install clang
yum install gflags gflags-devel
yum install sqlite-devel
yum install protobuf protobuf-devel protobuf-compiler
yum install libev libev-devel
yum install hiredis hiredis-devel
```

If during compilation `cblas.h` is not found, install the Atlas SSE fixes the problem:

```bash
yum install atlas-sse2-devel
```

##### 2. Compilation

Then use `cmake` to create the `make` targets, and run `make` to compile the code:

###### With CUDA (if available)

```bash
git submodule init
git submodule update
cd build
cmake ..
make -j 9
```

###### Without CUDA:

```bash
git submodule init
git submodule update
cd build_cpu
cmake .. -DWITH_CUDA=false
make -j 9
```

That's it. Now built examples will be stored in `build/examples`.
For instance a character prediction model using Stacked LSTMs is built under `build/examples/character_prediction`.

## Tests

To compile and run tests you need [Google Tests](https://code.google.com/p/googletest/). Download it [here](https://code.google.com/p/googletest/downloads/detail?name=gtest-1.7.0.zip).

#### 1. Compile and run tests

From the `build` (or `build_cpu`) folder do the following:

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

#### Latest Clang compiler on Mac OSX

Until Apple decides to fully embrace `thread_local` abstraction we are sadly forced to update our compilers manually (and no replacing with `__thread` is not enough...). Here are steps for updating your compiler:

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

## Packaging

Make sure that readme is consistent when releasing.

#### Source (tar.gz)
```bash
make -j9 package_source
```

#### For linux (DEB and RPM)
Must be run on linux with deb/rpm generation tools installed.

Don't forget to generate both cpu and gpu versions.

```bash
make -j9 package
```

#### For Apple (ZIP and homebrew's RB script)

Here we generate zip with file contents and rb file that uses it to install the latest dali in Homebrew.

Upload the zip file as a relase (make sure that version on github is consistent with cmake verison).
Upload the rb file.




### Future steps

* Add ImageNet, Caffe loading, broader ConvNet support (currently have `conv2d` and `conv1d`, but no pooling)
* Web interface for managing experiments (today [Dali-visualizer](https://github.com/JonathanRaiman/dali-visualizer) only shows progress and sample predictions).
* Web interface for visualizing network activity.
* Add some mathematical expressions from [Deepmind's Torch Cephes module](http://deepmind.github.io/torch-cephes/).
* Distribute training over multiple machines.
* Ensure feature parity with [**Python** extension](https://github.com/JonathanRaiman/dali-cython-stub)
* Implement multigpu support with [Fast Asynchronous Parallel SGD](http://arxiv.org/abs/1508.05711)
* Make it brew, yum/dnf and apt-get installable


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

[![Suggested theme song](https://i.ytimg.com/vi/c7BS4jbA_hw/mqdefault.jpg)](https://www.youtube.com/watch?v=c7BS4jbA_hwA)
