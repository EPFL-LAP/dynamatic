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
  --build-viz | -v            : build visual-dataflow tool
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

# Prepares to build a particular part of the project by creating a "build"
# folder for it, cd-ing to it, and displaying a message to stdout.
# $1: Name of the project part
# $2: Path to build folder (relative to script's CWD) 
prepare_to_build_project() {
  local project_name=$1 
  local build_dir=$2
  cd "$SCRIPT_CWD" && mkdir -p "$build_dir" && cd "$build_dir"
  echo_section "Building $project_name ($build_dir)"
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
CMAKE_COMPILERS="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
CMAKE_EXTRA_LLVM="" 
CMAKE_EXTRA_POLYGEIST="" 
ENABLE_TESTS=0
FORCE_CMAKE=0
PARSE_NUM_THREADS=0
NUM_THREADS=0
BUILD_TYPE="Debug"
BUILD_VISUAL_DATAFLOW="OFF"
for arg in "$@"; 
do
    if [[ $PARSE_NUM_THREADS -eq 1 ]]; then
      NUM_THREADS=$arg
      PARSE_NUM_THREADS=0
    else
      case "$arg" in 
          "--disable-build-opt" | "-o")
              CMAKE_COMPILERS=""
              CMAKE_EXTRA_LLVM="-DLLVM_CCACHE_BUILD=ON -DLLVM_USE_LINKER=lld"
              CMAKE_EXTRA_POLYGEIST="-DPOLYGEIST_USE_LINKER=lld"
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
          "--build-viz" | "-v")
              BUILD_VISUAL_DATAFLOW="ON"
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

#### Build the project (submodules and superproject) ####

# Print header
echo "################################################################################"
echo "############# DYNAMATIC - DHLS COMPILER INFRASTRUCTURE - EPFL/LAP ##############"
echo "################################################################################"

prepare_to_build_project "LLVM" "polygeist/llvm-project/build"

# CMake
if should_run_cmake ; then
  cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir;clang" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      $CMAKE_COMPILERS $CMAKE_EXTRA_LLVM
  exit_on_fail "Failed to cmake polygeist/llvm-project"
fi

# Build
run_ninja
exit_on_fail "Failed to build polygeist/llvm-project"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-mlir
    exit_on_fail "Tests for polygeist/llvm-project failed"
fi

prepare_to_build_project "Polygeist" "polygeist/build"

# CMake
if should_run_cmake ; then
  cmake -G Ninja .. \
      -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
      -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      $CMAKE_COMPILERS $CMAKE_EXTRA_POLYGEIST
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

prepare_to_build_project "Dynamatic" "build"

# CMake
if should_run_cmake ; then
  cmake -G Ninja .. \
      -DMLIR_DIR=polygeist/llvm-project/build/lib/cmake/mlir \
      -DLLVM_DIR=polygeist/llvm-project/build/lib/cmake/llvm \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_EXPORT_COMPILE_COMMANDS="ON" \
      -DBUILD_VISUAL_DATAFLOW=$BUILD_VISUAL_DATAFLOW \
      $CMAKE_COMPILERS
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
create_symlink build/bin/export-vhdl
create_symlink build/bin/exp-frequency-profiler
create_symlink build/bin/handshake-simulator
create_symlink build/bin/hls-verifier

# Make the scripts used by the frontend executable
chmod +x tools/dynamatic/scripts/compile.sh
chmod +x tools/dynamatic/scripts/write-hdl.sh
chmod +x tools/dynamatic/scripts/simulate.sh
chmod +x tools/dynamatic/scripts/synthesize.sh
chmod +x tools/dynamatic/scripts/visualize.sh

echo ""
echo_subsection "Build successful!"
