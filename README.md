# Dynamatic++

Compiler for Dynamically Scheduled High-Level Synthesis

## Building the Project

The following instructions can be used to setup Dynamatic++ from source.

1. **Install dependencies required by LLVM/MLIR.** These includes working C and C++ toolchains (compiler, linker), `cmake` and `ninja` for building the project, and `git`. For example, on Ubuntu:

   ```sh
   $ sudo apt-get install git cmake ninja clang lld ccache
   ```

   `clang`, `lld`, and `ccache` are not stictly required but significantly speed up rebuilds. If you do not wish to install them, simple remove flags `DCMAKE_C_COMPILER`, `DCMAKE_CXX_COMPILER`, `DLLVM_ENABLE_LLD`, and `DLLVM_CACHE_BUILD` from the following `cmake` commands.

2. **Clone the project and its submodules.** Dynamatic++ depends on [Polygeist](https://github.com/llvm/Polygeist) (C/C++ frontend for MLIR) and [CIRCT](https://github.com/llvm/circt) (Circuit-level IR compiler and tools). Both of them depend on [LLVM/MLIR](https://github.com/llvm/llvm-project).

   ```sh
   $ git clone --recurse-submodules https://github.com/EPFL-LAP/dynamatic.git
   ```

3. **Build and test Polygeist.** First, build the version of LLVM, MLIR, and Clang that Polygeist uses.

   ```sh
   $ cd dynamatic/polygeist
   $ mkdir llvm-project/build
   $ cd llvm-project/build
   $ cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS="mlir;clang" \
       -DLLVM_TARGETS_TO_BUILD="host" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DLLVM_ENABLE_LLD=ON \
       -DLLVM_CCACHE_BUILD=ON
   $ ninja
   $ ninja check-mlir
   ```

   Then, build Polygeist itself.

   ```sh
    $ cd dynamatic/polygeist
    $ mkdir build
    $ cd build
    $ cmake -G Ninja .. \
        -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
        -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_LLD=ON
   $ ninja
   $ ninja check-polygeist-opt
   $ ninja check-cgeist
   ```

4. **Build and test CIRCT.** First, build the version of LLVM and MLIR that CIRCT uses.
   ```sh
   $ cd dynamatic/circt
   $ mkdir llvm/build
   $ cd llvm/build
   $ cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS="mlir" \
       -DLLVM_TARGETS_TO_BUILD="host" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DLLVM_ENABLE_LLD=ON \
       -DLLVM_CCACHE_BUILD=ON
   $ ninja
   $ ninja check-mlir
   ```
   Then, build CIRCT itself.
   ```sh
   $ cd dynamatic/circt
   $ mkdir build
   $ cd build
   $ cmake -G Ninja .. \
       -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
       -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
       -DCMAKE_BUILD_TYPE=Debug \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DLLVM_ENABLE_LLD=ON
   $ ninja
   $ ninja check-circt
   $ ninja check-circt-integration
   ```

The `-DCMAKE_BUILD_TYPE=Debug` flag enables debug information, which makes the whole tree compile slower, but allows you to step through code into the LLVM and MLIR frameworks. To get something that runs fast, use `-DCMAKE_BUILD_TYPE=Release` or `-DCMAKE_BUILD_TYPE=RelWithDebInfo` if you want to go fast and optionally if you want debug info to go with it.

## Getting the Latest Version of the Project

The following instructions can be used to get the latest stable version of Dynamatic++.

1. **Pull new changes from the repository and its submodules.** The following command will also automatically update all submodules recursively.
   ```sh
   $ cd dynamatic
   $ git pull --recurse-submodules
   ```
2. **Rebuild and test the project.**
   ```sh
   # First, the LLVM submodule within Polygeist
   $ cd dynamatic/polygeist/llvm-project/build
   $ ninja && ninja check-mlir
   # Then, Polygeist itself
   $ cd ../../build
   $ ninja && ninja check-polygeist-opt && ninja check-cgeist
   # Then, the LLVM submodule within CIRCT
   $ cd ../../circt/llvm/build
   $ ninja && ninja check-mlir
   # Finally, CIRCT itself
   $ cd ../../build
   $ ninja && ninja check-circt && ninja check-circt-integration
   ```
