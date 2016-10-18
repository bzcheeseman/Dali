# Profiling Compilation

Low compilation time is important for rapid iteration. If you need to wait 10 minutes every time you make a slight change to the code, the development process becomes tiring and not very enjoyable.

If you noticed that recent changes drastically affect the compile time it might be the right time to spend some time understand the causes. Here's a simple list of steps:

### 1. Find files that take long time to compile:

```bash
cd you_build_directory
rm -rf ./*
CXX="time g++" cmake -DWITH_CUDA=FALSE ..
make -j1
```

This will cause time to be printed after compiling each file. It is important to use `-j1` to get compile time estimate, which is unaffected by other files.

### 2. Isolate a small piece of code that compiles for too long.

Now that you have a list of files that compile longer than expected, find an example piece of code that compiles longer than expected. In my case it was:

```cpp
#include "dali/array/array.h"
#include "dali/array/lazy/unary.h"

Assignable<Array> identity(const Array& x) {
    return lazy::identity(x);
}
```

Which I put in the file `dali/array/op/compile_test.cu.cpp`. Now you can quickly check how every change affect compilation time for that file by running:

```bash
touch ../dali/array/op/compile_test.cu.cpp
make -j1 dali_generated/dali/array/op/compile_test.cpp.o
```

### 3. Run a profile build in your compiler

```
cmake -DCMAKE_BUILD_TYPE="profile_compilation" ..
make -j1 dali_generated/dali/array/op/compile_test.cpp.o
```

### 4. Understand the problem and make changes

There is a (stack overflow question)[http://stackoverflow.com/questions/28919285/identify-slow-to-compile-function] that you might find helpful. In general the problem is usually too many headers or too many templates.

### 5. Profit
