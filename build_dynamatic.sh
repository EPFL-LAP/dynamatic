#!/bin/bash

# Kill the whole script on Ctrl+C
trap "exit" INT

# Directory where the script is being ran from (must be directory where script
# is located!)
SCRIPT_CWD=$PWD

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


BUILD_TYPE="Debug"
DYNAMATIC_BUILD_DIR="build"
CMAKE_FLAGS_SUPER="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON" 

echo_section "Building Dynamatic"
create_build_directory "${DYNAMATIC_BUILD_DIR}"

# CMake
echo $PWD
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
 
