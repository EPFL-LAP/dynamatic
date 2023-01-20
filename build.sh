#!/bin/bash

# Kill the whole script on Ctrl+C
trap "exit" INT

# Display list of possible options and exit
print_help_and_exit () {
    echo -e \
"./build.sh [options]

List of options:
    --disable-build-opt | -o    : don't use clang/lld/ccache to speed up builds
    --disable-tests | -t        : don't run tests during build
    --force-rebuild | -f        : force rebuild everything
    --release | -r              : build in \"Release\" mode (default is \"Debug\")
    --help | -h                 : display this help message
"
    exit
}

# Helper function to exit script on failed command
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo -e "\n$1"
        fi
        echo -e "\n-------------- Build failed ------------"
        exit 1
    fi
}

# Create symbolic link from the bin/ directory to an executable file built by
# the repository. The symbolic link's name is the same as the executable file.
# The path to the executable file must be passed as the first argument to this
# function and be relative to the repository's root. The function assumes that
# the bin/ directory exists and that the current working directory is the
# repository's root. 
create_symlink() {
    local src=${1}
    local dst="bin/$(basename ${1})"
    echo "${dst} -> ${src}"
    ln -f --symbolic ../${src} ${dst}
}

#### Parse arguments ####

# Loop over command line arguments and update script variables
CMAKE_FLAGS_EXT="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON" 
CMAKE_FLAGS_LLVM="$CMAKE_FLAGS_EXT -DLLVM_CCACHE_BUILD=ON" 
DISABLE_TESTS=0
FORCE_REBUILD=0
BUILD_TYPE="Debug"
for arg in "$@"; 
do
    case "$arg" in 
        "--disable-build-opt" | "-o")
            CMAKE_FLAGS_LLVM=""
            CMAKE_FLAGS_EXT=""
            ;;
        "--disable-tests" | "-t")
            DISABLE_TESTS=1
            ;;
        "--force-rebuild" | "-f")
            FORCE_REBUILD=1
            ;;
        "--release" | "-r")
            BUILD_TYPE="Release"
            ;;
        "--help" | "-h")
            print_help_and_exit
            ;;
        *)
            echo "Unknown argument \"$arg\", printing help and aborting"
            print_help_and_exit
            ;;
    esac
done

# Delete build folders if forcing rebuild
if [[ $FORCE_REBUILD -ne 0 ]]; then
    rm -rf polygeist/llvm-project/build polygeist/build # Polygeist
    rm -rf circt/llvm/build circt/build # CIRCT
    rm -rf build # Dynamatic++
fi

echo -e \
"\n----------------------------------------
---------- Building Polygeist ----------
----------------------------------------\n"

cd polygeist

echo -e "\n---- Building polygeist/llvm-project ---\n"

# Create build directory and cd to it
mkdir -p llvm-project/build && cd llvm-project/build

# Build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    ${CMAKE_FLAGS_LLVM}
exit_on_fail "Failed to cmake polygeist/llvm-project"

ninja
exit_on_fail "Failed to build polygeist/llvm-project"
if [[ DISABLE_TESTS -eq 0 ]]; then
    ninja check-mlir
    exit_on_fail "Tests for polygeist/llvm-project failed"
fi

echo -e "\n---------- Building polygeist ----------\n"

# Create build directory and cd to it
cd ../.. && mkdir -p build && cd build

# Build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
    -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    $CMAKE_FLAGS_EXT
exit_on_fail "Failed to cmake polygeist"

ninja
exit_on_fail "Failed to build polygeist"
if [[ DISABLE_TESTS -eq 0 ]]; then
    ninja check-polygeist-opt
    exit_on_fail "Tests for polygeist failed"
    ninja check-cgeist
    exit_on_fail "Tests for polygeist failed"
fi

echo -e \
"\n----------------------------------------
------------ Building CIRCT ------------
----------------------------------------\n"

echo -e "\n---------- Building circt/llvm ---------\n"

# Create build directory and cd to it
cd ../../circt && mkdir -p llvm/build && cd llvm/build

# Build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CMAKE_FLAGS_LLVM
exit_on_fail "Failed to cmake circt/llvm"

ninja
exit_on_fail "Failed to build circt-llvm"
if [[ DISABLE_TESTS -eq 0 ]]; then
    ninja check-mlir
    exit_on_fail "Tests for circt/llvm failed"
fi

echo -e "\n------------ Building circt ------------\n"

# Create build directory and cd to it
cd ../.. && mkdir -p build && cd build

cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CMAKE_FLAGS_EXT
exit_on_fail "Failed to cmake circt"

ninja
exit_on_fail "Failed to build circt"
if [[ DISABLE_TESTS -eq 0 ]]; then
    ninja check-circt
    exit_on_fail "Tests for circt failed"
    ninja check-circt-integration
    exit_on_fail "Tests for circt failed"
fi

echo -e \
"\n----------------------------------------
--------- Building Dynamatic++ ---------
----------------------------------------\n"

# Create build directory and cd to it
cd ../.. && mkdir -p build && cd build

cmake -G Ninja .. \
    -DCIRCT_DIR=$PWD/../circt/build/lib/cmake/circt \
    -DMLIR_DIR=$PWD/../circt/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../circt/llvm/build/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CMAKE_FLAGS_EXT
exit_on_fail "Failed to cmake dynamatic++"

ninja
exit_on_fail "Failed to build dynamatic++"

echo -e \
"\n----------------------------------------
------------- Building CLI -------------
----------------------------------------\n"

cd ../cli
cargo build
exit_on_fail "Failed to build CLI"

echo -e \
"\n----------------------------------------
------- Creating symbolic links --------
----------------------------------------\n"

# Create bin/ directory at the project's root
cd .. && mkdir -p bin 

# Create symbolic links to all binaries we use from subfolders
create_symlink polygeist/build/bin/cgeist
create_symlink polygeist/build/bin/polygeist-opt
create_symlink polygeist/llvm-project/build/bin/mlir-opt
create_symlink circt/build/bin/circt-opt
create_symlink cli/target/debug/cli
create_symlink build/bin/dynamatic++

echo -e "\n----------- Build successful -----------"
 