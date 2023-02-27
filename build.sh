#!/bin/bash

# Kill the whole script on Ctrl+C
trap "exit" INT

# Directory where the script is being ran from (must be directory where script
# is located!)
SCRIPT_CWD=$PWD

#### Helper functions ####

# Display list of possible options and exit
print_help_and_exit () {
    echo -e \
"./build.sh [options]

List of options:
  --disable-build-opt | -o    : don't use clang/lld/ccache to speed up builds
  --force-rebuild | -f        : force rebuild everything
  --release | -r              : build in \"Release\" mode (default is \"Debug\")
  --check | -c                : run tests during build
  --help | -h                 : display this help message
"
    exit
}

# Helper function to print large section title text
echo_section() {
    echo ""
    echo "# ===----------------------------------------------------------------------=== #"
    echo "# $1"
    echo "# ===----------------------------------------------------------------------=== #"
    echo ""
}

# Helper function to print subsection title text
echo_subsection() {
    echo ""
    echo "# ===--- $1 ---==="
    echo ""
}

# Helper function to exit script on failed command
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo -e "\n$1"
        fi
        echo_section "Build failed!"
        exit 1
    fi
}

# Helper function to create build directory and cd to it
create_build_directory() {
    cd "${SCRIPT_CWD}" && mkdir -p ${1} && cd ${1}
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
CMAKE_FLAGS_SUPER="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON" 
CMAKE_FLAGS_LLVM="$CMAKE_FLAGS_SUPER -DLLVM_CCACHE_BUILD=ON" 
ENABLE_TESTS=0
FORCE_REBUILD=0
BUILD_TYPE="Debug"
for arg in "$@"; 
do
    case "$arg" in 
        "--disable-build-opt" | "-o")
            CMAKE_FLAGS_LLVM=""
            CMAKE_FLAGS_SUPER=""
            ;;
        "--force-rebuild" | "-f")
            FORCE_REBUILD=1
            ;;
        "--release" | "-r")
            BUILD_TYPE="Release"
            ;;
        "--check" | "-c")
            ENABLE_TESTS=1
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

# Path to build directories
POLYGEIST_LLVM_BUILD_DIR="polygeist/llvm-project/build"
POLYGEIST_BUILD_DIR="polygeist/build"
CIRCT_LLVM_BUILD_DIR="circt/llvm/build"
CIRCT_BUILD_DIR="circt/build"
DYNAMATIC_BUILD_DIR="build"

# Delete build folders if forcing rebuild
if [[ $FORCE_REBUILD -ne 0 ]]; then
    rm -rf "${POLYGEIST_LLVM_BUILD_DIR}" "${POLYGEIST_BUILD_DIR}" # Polygeist
    rm -rf "${CIRCT_LLVM_BUILD_DIR}" "${CIRCT_BUILD_DIR}" # CIRCT
    rm -rf "${DYNAMATIC_BUILD_DIR}" # Dynamatic
fi

#### Build the project (submodules and superproject) ####


# Print header
echo "################################################################################"
echo "############# DYNAMATIC - DHLS COMPILER INFRASTRUCTURE - EPFL/LAP ##############"
echo "################################################################################"

echo_section "Building Polygeist"
echo_subsection "Building LLVM submodule"
create_build_directory "${POLYGEIST_LLVM_BUILD_DIR}"

# CMake
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    ${CMAKE_FLAGS_LLVM}
exit_on_fail "Failed to cmake polygeist/llvm-project"

# Build
ninja
exit_on_fail "Failed to build polygeist/llvm-project"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-mlir
    exit_on_fail "Tests for polygeist/llvm-project failed"
fi

echo_subsection "Building superproject"
create_build_directory "${POLYGEIST_BUILD_DIR}"

# CMake
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
    -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -Wno-dev \
    $CMAKE_FLAGS_SUPER
exit_on_fail "Failed to cmake polygeist"

# Build
ninja
exit_on_fail "Failed to build polygeist"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-polygeist-opt
    exit_on_fail "Tests for polygeist failed"
    ninja check-cgeist
    exit_on_fail "Tests for polygeist failed"
fi

echo_section "Building CIRCT"
echo_subsection "Building LLVM submodule"
create_build_directory "${CIRCT_LLVM_BUILD_DIR}"

# CMake
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CMAKE_FLAGS_LLVM
exit_on_fail "Failed to cmake circt/llvm"

# Build
ninja
exit_on_fail "Failed to build circt-llvm"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-mlir
    exit_on_fail "Tests for circt/llvm failed"
fi

echo_subsection "Building superproject"
create_build_directory "${CIRCT_BUILD_DIR}"

# CMake
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CMAKE_FLAGS_SUPER
exit_on_fail "Failed to cmake circt"

# Build
ninja
exit_on_fail "Failed to build circt"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-circt
    exit_on_fail "Tests for circt failed"
    ninja check-circt-integration
    exit_on_fail "Integration tests for circt failed"
fi

echo_section "Building Dynamatic"
create_build_directory "${DYNAMATIC_BUILD_DIR}"

# CMake
cmake -G Ninja .. \
    -DCIRCT_DIR=$PWD/../circt/build/lib/cmake/circt \
    -DMLIR_DIR=$PWD/../circt/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../circt/llvm/build/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CMAKE_FLAGS_SUPER
exit_on_fail "Failed to cmake dynamatic"

# Build
ninja
exit_on_fail "Failed to build dynamatic"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-dynamatic
    exit_on_fail "Tests for dynamatic failed"
fi

echo_section "Creating symbolic links"

# Create bin/ directory at the project's root
cd "${SCRIPT_CWD}" && mkdir -p bin 

# Create symbolic links to all binaries we use from subfolders
create_symlink polygeist/build/bin/cgeist
create_symlink polygeist/build/bin/polygeist-opt
create_symlink polygeist/llvm-project/build/bin/mlir-opt
create_symlink circt/build/bin/circt-opt
create_symlink build/bin/dynamatic-opt
create_symlink build/bin/dynamatic++

echo_section "Build successful!"
 