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
  --release | -r                       : build in \"Release\" mode (default is \"Debug\")
  --skip-polygeist <polygeist-path>    : skip building POLYGEIST
  --visual-dataflow | -v               : build visual-dataflow's C++ library
  --export-godot | -e <godot-path>     : export the Godot project (requires engine)
  --force | -f                         : force cmake reconfiguration in each (sub)project
  --threads | -t <num-threads>         : number of concurrent threads to build on (by
                                         default, one thread per logical core on the host
                                         machine)
  --llvm-parallel-link-jobs <num-jobs> : maximum number of simultaneous link jobs when
                                         building llvm (defaults to 2)
  --disable-build-opt | -o             : don't use clang/lld/ccache to speed up builds
  --experimental-enable-xls            : enable experimental xls integration
  --enable-leq-binaries                : download binaries for elastic-miter equivalence
                                         checking
  --check | -c                         : run tests during build
  --help | -h                          : display this help message
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
    echo -e "\n# ===--- $1 ---==="
}

# Helper function to exit script on failed command
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo -e "\n$1"
        fi
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
    ln -f --symbolic $src $dst
}

# Same as create_symlink but creates the symbolic link inside the bin/generators
# subfolder.
create_generator_symlink() {
    local src=$1
    local dst="bin/generators/$(basename $1)"
    echo "$dst -> $src"
    ln -f --symbolic ../../$src $dst
}


# Determine whether cmake should be re-configured by looking for a
# CMakeCache.txt file in the current working directory.
should_run_cmake() {
  if [[ -f "CMakeCache.txt" && $FORCE_CMAKE -eq 0 ]]; then
    echo "CMake configuration found, will not re-configure cmake"
    echo "Run script with -f or --force flag to re-configure cmake"
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

CMAKE_COMPILERS="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
CMAKE_LLVM_BUILD_OPTIMIZATIONS="-DLLVM_CCACHE_BUILD=ON -DLLVM_USE_LINKER=lld"
CMAKE_POLYGEIST_BUILD_OPTIMIZATIONS="-DPOLYGEIST_USE_LINKER=lld"
CMAKE_DYNAMATIC_BUILD_OPTIMIZATIONS="-DDYNAMATIC_CCACHE_BUILD=ON -DLLVM_USE_LINKER=lld"
CMAKE_DYNAMATIC_ENABLE_XLS=""
CMAKE_DYNAMATIC_ENABLE_LEQ_BINARIES=""
ENABLE_TESTS=0
FORCE_CMAKE=0
NUM_THREADS=0
LLVM_PARALLEL_LINK_JOBS=2
BUILD_TYPE="Debug"
BUILD_VISUAL_DATAFLOW=0
GODOT_PATH=""
ENABLE_XLS_INTEGRATION=0
SKIP_POLYGEIST=0
POLYGEIST_DIR="$PWD/polygeist"

# Loop over command line arguments and update script variables
PARSE_ARG=""
for arg in "$@";
do
    if [[ $PARSE_ARG == "num-threads" ]]; then
      NUM_THREADS="$arg"
      PARSE_ARG=""
    elif [[ $PARSE_ARG == "godot-path" ]]; then
      GODOT_PATH="$arg"
      # If the path is relative, prepend .. to it since we will build the
      # project from the visual-dataflow subfolder
      if [[ $GODOT_PATH != /* ]]; then
        GODOT_PATH="../$GODOT_PATH"
      fi
      PARSE_ARG=""
    elif [[ $PARSE_ARG == "polygeist-path" ]]; then
      POLYGEIST_DIR="$arg"
      PARSE_ARG=""
    elif [[ $PARSE_ARG == "llvm-parallel-link-jobs" ]]; then
      LLVM_PARALLEL_LINK_JOBS="$arg"
      PARSE_ARG=""
    else
      case "$arg" in
          "--disable-build-opt" | "-o")
              CMAKE_COMPILERS=""
              CMAKE_LLVM_BUILD_OPTIMIZATIONS=""
              CMAKE_POLYGEIST_BUILD_OPTIMIZATIONS=""
              CMAKE_DYNAMATIC_BUILD_OPTIMIZATIONS=""
              ;;
          "--force" | "-f")
              FORCE_CMAKE=1
              ;;
          "--release" | "-r")
              BUILD_TYPE="Release"
              ;;
          "--check" | "-c")
              ENABLE_TESTS=1
              ;;
          "--visual-dataflow" | "-v")
              BUILD_VISUAL_DATAFLOW=1
              ;;
          "--threads" | "-t")
              PARSE_ARG="num-threads"
              ;;
          "--llvm-parallel-link-jobs")
              PARSE_ARG="llvm-parallel-link-jobs"
              ;;
          "--export-godot" | "-e")
              PARSE_ARG="godot-path"
              ;;
          "--skip-polygeist")
              SKIP_POLYGEIST=1
              PARSE_ARG="polygeist-path"
              ;;
          "--experimental-enable-xls")
              ENABLE_XLS_INTEGRATION=1
              CMAKE_DYNAMATIC_ENABLE_XLS="-DDYNAMATIC_ENABLE_XLS=ON"
              ;;
          "--enable-leq-binaries")
              CMAKE_DYNAMATIC_ENABLE_LEQ_BINARIES="-DDYNAMATIC_ENABLE_LEQ_BINARIES=ON"
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
if [[ $PARSE_ARG != "" ]]; then
  echo "Missing argument \"$PARSE_ARG\", printing help and aborting"
  print_help_and_exit
fi


#### Build the project (submodules, superproject, and tools) ####

# Print header
echo "################################################################################"
echo "############# DYNAMATIC - DHLS COMPILER INFRASTRUCTURE - EPFL/LAP ##############"
echo "################################################################################"

if [[ $SKIP_POLYGEIST -eq 0 ]]; then

  #### Polygeist ####

  prepare_to_build_project "LLVM" "polygeist/llvm-project/build"

  # CMake
  if should_run_cmake ; then
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="mlir;clang" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DLLVM_PARALLEL_LINK_JOBS=$LLVM_PARALLEL_LINK_JOBS \
        $CMAKE_COMPILERS $CMAKE_LLVM_BUILD_OPTIMIZATIONS
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
        $CMAKE_COMPILERS $CMAKE_POLYGEIST_BUILD_OPTIMIZATIONS
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

else
  echo "Skipping POLYGEIST/LLVM build. IMPORTANT: Verify that the path of polygeist in the script tools/dynamatic/scripts/compile.sh is the same"
  if [[ ! -d $POLYGEIST_DIR ]]; then
    echo "POLYGEIST directory not found: $POLYGEIST_DIR"
    exit 1
  fi
fi

#### XLS ####

XLS_DIR="$SCRIPT_CWD/xls"
XLS_UPSTREAM="https://github.com/ETHZ-DYNAMO/xls.git"
XLS_COMMIT="939eb43c307005caf4af75ed9b8a0dbc6c905386"

if [[ ENABLE_XLS_INTEGRATION -eq 1 || -d "$XLS_DIR" ]]; then
    echo_section "Preparing XLS"
fi

if [[ ENABLE_XLS_INTEGRATION -eq 1 ]]; then
    # If XLS does not exist, clone it down + checkout the correct commit:
    if [ ! -d "$XLS_DIR" ]; then
        echo "XLS not found. Cloning from $XLS_UPSTREAM..."
        git clone "$XLS_UPSTREAM" "$XLS_DIR"
        exit_on_fail "Failed to clone XLS"
        cd "$XLS_DIR"
        git checkout "$XLS_COMMIT"
        exit_on_fail "Failed to checkout commit $XLS_COMMIT"
    fi
fi

# If an XLS checkout exists, verify that it is on the correct commit:
if [ -d "$XLS_DIR" ]; then
    echo "XLS found, validating commit.."
    XLS_CURRENT_COMMIT=$(git --git-dir="$XLS_DIR/.git" describe --always --abbrev=0 --match "this_tag_does_not_exist")
    exit_on_fail "Failed to determine XLS commit."

    echo "Current commit:  $XLS_CURRENT_COMMIT"
    echo "Required commit: $XLS_COMMIT"

    if [[ "$XLS_CURRENT_COMMIT" != "$XLS_COMMIT" ]]; then
        echo ""
        printf "\033[91m"
        echo_subsection "!! WARNING !!"
        echo ""
        echo "WARNING: Incorrect XLS commit detected. To checkout correct commit: "
        echo "  cd $XLS_DIR"
        echo "  git checkout $XLS_COMMIT"
        echo_subsection "!! WARNING !!"
        printf "\33[0m"
    else
        echo "OK"
    fi
fi

#### Dynamatic ####

prepare_to_build_project "Dynamatic" "build"

# CMake
if should_run_cmake ; then
  cmake -G Ninja .. \
      -DMLIR_DIR="$POLYGEIST_DIR"/llvm-project/build/lib/cmake/mlir \
      -DLLVM_DIR="$POLYGEIST_DIR"/llvm-project/build/lib/cmake/llvm \
      -DCLANG_DIR="$POLYGEIST_DIR"/llvm-project/build/lib/cmake/clang \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_EXPORT_COMPILE_COMMANDS="ON" \
      $CMAKE_COMPILERS $CMAKE_DYNAMATIC_BUILD_OPTIMIZATIONS $CMAKE_DYNAMATIC_ENABLE_XLS $CMAKE_DYNAMATIC_ENABLE_LEQ_BINARIES
  exit_on_fail "Failed to cmake dynamatic"
fi

# Build
run_ninja
exit_on_fail "Failed to build dynamatic"
if [[ ENABLE_TESTS -eq 1 ]]; then
    ninja check-dynamatic
    exit_on_fail "Tests for dynamatic failed"
fi

# Build Chisel generators

echo_subsection "Building LSQ generator"

LSQ_GEN_PATH="tools/backend/lsq-generator-chisel"
LSQ_GEN_JAR="target/scala-2.13/lsq-generator.jar"
cd "$SCRIPT_CWD/$LSQ_GEN_PATH"
sbt assembly
exit_on_fail "Failed to build LSQ generator"
chmod +x $LSQ_GEN_JAR

#### visual-dataflow ####

if [[ BUILD_VISUAL_DATAFLOW -ne 0 ]]; then
  prepare_to_build_project "visual-dataflow" "visual-dataflow/build"

  # CMake
  if should_run_cmake ; then
    cmake -G Ninja .. \
        -DMLIR_DIR="$POLYGEIST_DIR"/llvm-project/build/lib/cmake/mlir \
        -DLLVM_DIR="$POLYGEIST_DIR"/llvm-project/build/lib/cmake/llvm \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_EXPORT_COMPILE_COMMANDS="ON" \
        $CMAKE_COMPILERS
    exit_on_fail "Failed to cmake visual-dataflow"
  fi

  # Build
  run_ninja
  exit_on_fail "Failed to build visual-dataflow"
fi

#### Godot ####

if [[ $GODOT_PATH != "" ]]; then
  # Go to the visualizer's subfolder and build it using godot
  cd "$SCRIPT_CWD/visual-dataflow"
  "$GODOT_PATH" --headless --export-debug "Linux/X11"
  exit_on_fail "Failed to build Godot project"

  # Erase the useless shell script generated by Godot and cd back the
  # Dynamatic's top-level folder
  rm bin/visual-dataflow.sh
  cd ..
fi

#### Symbolic links ####

echo_section "Creating symbolic links"

# Create bin/ directory at the project's root
cd "$SCRIPT_CWD" && mkdir -p bin/generators

# Create symbolic links to all binaries we use from subfolders

create_symlink "$POLYGEIST_DIR"/build/bin/cgeist
create_symlink "$POLYGEIST_DIR"/build/bin/polygeist-opt
create_symlink "$POLYGEIST_DIR"/llvm-project/build/bin/clang++
create_symlink ../build/bin/dynamatic
create_symlink ../build/bin/dynamatic-mlir-lsp-server
create_symlink ../build/bin/dynamatic-opt
create_symlink ../build/bin/elastic-miter
create_symlink ../build/bin/export-dot
create_symlink ../build/bin/export-cfg
create_symlink ../build/bin/export-rtl
create_symlink ../build/bin/exp-frequency-profiler
create_symlink ../build/bin/handshake-simulator
create_symlink ../build/bin/hls-verifier
create_symlink ../build/bin/wlf2csv
create_generator_symlink build/bin/rtl-cmpf-generator
create_generator_symlink build/bin/rtl-cmpi-generator
create_generator_symlink build/bin/rtl-text-generator
create_generator_symlink build/bin/rtl-constant-generator-verilog
create_generator_symlink build/bin/exp-sharing-wrapper-generator
create_generator_symlink "$LSQ_GEN_PATH/$LSQ_GEN_JAR"
if [[ $GODOT_PATH != "" ]]; then
  create_symlink ../visual-dataflow/bin/visual-dataflow
fi

# Make the scripts used by the frontend executable
chmod +x tools/dynamatic/scripts/compile.sh
chmod +x tools/dynamatic/scripts/write-hdl.sh
chmod +x tools/dynamatic/scripts/simulate.sh
chmod +x tools/dynamatic/scripts/synthesize.sh
chmod +x tools/dynamatic/scripts/visualize.sh

echo_subsection "Build successful!"
