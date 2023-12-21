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
  --force-cmake | -f          : force cmake reconfiguration in each (sub)project 
  --release | -r              : build in \"Release\" mode (default is \"Debug\")
  --threads | -t              : number of concurrent threads to build on with
                                ninja (by default, ninja spawns one thread per
                                logical core on the host machine)
  --build-visualizer | -v     : build experimental visual-dataflow tool
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
    echo "# ===--- $1 ---==="
}

# Helper function to exit script on failed command
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo -e "\n$1"
        fi
        echo ""
        echo_subsection "Build failed!"
        exit 1
    fi
}

# Helper function to create build directory and cd to it
create_build_directory() {
    cd "$SCRIPT_CWD" && mkdir -p $1 && cd $1
}

# Create symbolic link from the bin/ directory to an executable file built by
# the repository. The symbolic link's name is the same as the executable file.
# The path to the executable file must be passed as the first argument to this
# function and be relative to the repository's root. The function assumes that
# the bin/ directory exists and that the current working directory is the
# repository's root. 
create_symlink() {
    local src=$1
    local dst="bin/$(basename $1)"
    echo "$dst -> $src"
    ln -f --symbolic ../$src $dst
}

# Determine whether cmake should be re-configured by looking for a
# CMakeCache.txt file in the current working directory.
should_run_cmake() {
  if [[ -f "CMakeCache.txt" && $FORCE_CMAKE -eq 0 ]]; then
    echo "CMake configuration found, will not re-configure cmake"
    echo "Run script with -f or --force-cmake flag to re-configure cmake"
    echo ""
    return 1
  fi 
  return 0
}

# Run ninja using the number of threads provided as argument, if any. Otherwise,
# let ninja pick the number of threads to use  
run_ninja() {
  if [[ $NUM_THREADS -eq 0 ]]; then
    ninja
  else
    ninja -j "$NUM_THREADS"
  fi 
}

#### Parse arguments ####

# Loop over command line arguments and update script variables
CMAKE_FLAGS_SUPER="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON" 
CMAKE_FLAGS_LLVM="$CMAKE_FLAGS_SUPER -DLLVM_CCACHE_BUILD=ON" 
ENABLE_TESTS=0
FORCE_CMAKE=0
PARSE_NUM_THREADS=0
NUM_THREADS=0
BUILD_TYPE="Debug"
BUILD_VISUALIZER="OFF"
for arg in "$@"; 
do
    if [[ $PARSE_NUM_THREADS -eq 1 ]]; then
      NUM_THREADS=$arg
      PARSE_NUM_THREADS=0
    else
      case "$arg" in 
          "--disable-build-opt" | "-o")
              CMAKE_FLAGS_LLVM=""
              CMAKE_FLAGS_SUPER=""
              ;;
          "--force-cmake" | "-f")
              FORCE_CMAKE=1
              ;;
          "--release" | "-r")
              BUILD_TYPE="Release"
              ;;
          "--threads" | "-t")
              PARSE_NUM_THREADS=1
              ;;
          "--check" | "-c")
              ENABLE_TESTS=1
              ;;
          "--build-visualizer" | "-v")
              BUILD_VISUALIZER="ON"
              ;;
          "--help" | "-h")
              print_help_and_exit
              ;;
          *)
              echo "Unknown argument \"$arg\", printing help and aborting"
              print_help_and_exit
              ;;
      esac
    fi
done

# Path to build directories
POLYGEIST_LLVM_BUILD_DIR="polygeist/llvm-project/build"
POLYGEIST_BUILD_DIR="polygeist/build"
CIRCT_LLVM_BUILD_DIR="circt/llvm/build"
CIRCT_BUILD_DIR="circt/build"
DYNAMATIC_BUILD_DIR="build"

#### Build the project (submodules and superproject) ####

# Print header
echo "################################################################################"
echo "############# DYNAMATIC - DHLS COMPILER INFRASTRUCTURE - EPFL/LAP ##############"
echo "################################################################################"

echo_section "Building Polygeist"
echo_subsection "Building LLVM submodule"
echo ""
create_build_directory "$POLYGEIST_LLVM_BUILD_DIR"

# CMake
if should_run_cmake ; then
  cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir;clang" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      $CMAKE_FLAGS_LLVM
  exit_on_fail "Failed to cmake polygeist/llvm-project"
fi

# Build
run_ninja
exit_on_fail "Failed to build polygeist/llvm-project"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-mlir
    exit_on_fail "Tests for polygeist/llvm-project failed"
fi

echo ""
echo_subsection "Building superproject"
echo ""
create_build_directory "$POLYGEIST_BUILD_DIR"

# CMake
if should_run_cmake ; then
  cmake -G Ninja .. \
      -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
      -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -Wno-dev \
      $CMAKE_FLAGS_SUPER
  exit_on_fail "Failed to cmake polygeist"
fi

# Build
run_ninja
exit_on_fail "Failed to build polygeist"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-polygeist-opt
    exit_on_fail "Tests for polygeist failed"
    ninja check-cgeist
    exit_on_fail "Tests for polygeist failed"
fi

echo_section "Building CIRCT"
echo_subsection "Building LLVM submodule"
echo ""
create_build_directory "$CIRCT_LLVM_BUILD_DIR"

# CMake
if should_run_cmake ; then
  cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      $CMAKE_FLAGS_LLVM
  exit_on_fail "Failed to cmake circt/llvm"
fi

# Build
run_ninja
exit_on_fail "Failed to build circt-llvm"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-mlir
    exit_on_fail "Tests for circt/llvm failed"
fi

echo ""
echo_subsection "Building superproject"
echo 
create_build_directory "$CIRCT_BUILD_DIR"

# CMake
if should_run_cmake ; then
  cmake -G Ninja .. \
      -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
      -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DVERILATOR_DISABLE=ON \
      -DVIVADO_DISABLE=ON \
      -DCLANG_TIDY_DISABLE=ON \
      -DSYSTEMC_DISABLE=ON \
      -DQUARTUS_DISABLE=ON \
      -DQUESTA_DISABLE=ON \
      -DYOSYS_DISABLE=ON \
      -DIVERILOG_DISABLE=ON \
      -DYOSYS_DISABLE=ON \
      -DCAPNP_DISABLE=ON \
      -DOR_TOOLS_DISABLE=ON \
      -DCAPNP_DISABLE=ON \
      $CMAKE_FLAGS_SUPER
  exit_on_fail "Failed to cmake circt"
fi

# Build
run_ninja
exit_on_fail "Failed to build circt"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-circt
    exit_on_fail "Tests for circt failed"
    ninja check-circt-integration
    exit_on_fail "Integration tests for circt failed"
fi

echo_section "Building Dynamatic"
create_build_directory "$DYNAMATIC_BUILD_DIR"

# CMake
if should_run_cmake ; then
  cmake -G Ninja .. \
      -DCIRCT_DIR=$PWD/../circt/build/lib/cmake/circt \
      -DMLIR_DIR=$PWD/../circt/llvm/build/lib/cmake/mlir \
      -DLLVM_DIR=$PWD/../circt/llvm/build/lib/cmake/llvm \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DDYNAMATIC_BUILD_VISUALIZER=$BUILD_VISUALIZER \
      $CMAKE_FLAGS_SUPER
  exit_on_fail "Failed to cmake dynamatic"
fi

# Build
run_ninja
exit_on_fail "Failed to build dynamatic"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-dynamatic
    exit_on_fail "Tests for dynamatic failed"
fi

echo_section "Creating symbolic links"

# Create bin/ directory at the project's root
cd "$SCRIPT_CWD" && mkdir -p bin 

# Create symbolic links to all binaries we use from subfolders
create_symlink polygeist/build/bin/cgeist
create_symlink polygeist/build/bin/polygeist-opt
create_symlink polygeist/llvm-project/build/bin/clang++
create_symlink build/bin/dynamatic
create_symlink build/bin/dynamatic-opt
create_symlink build/bin/export-dot
create_symlink build/bin/exp-frequency-profiler
create_symlink build/bin/handshake-simulator

# Make the scripts used by the frontend executable
chmod +x tools/dynamatic/scripts/compile.sh
chmod +x tools/dynamatic/scripts/write-hdl.sh
chmod +x tools/dynamatic/scripts/simulate.sh
chmod +x tools/dynamatic/scripts/synthesize.sh

echo ""
echo_subsection "Build successful!"
