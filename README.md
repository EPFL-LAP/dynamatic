# Dynamatic

Dynamatic is an academic, open-source high-level synthesis compiler that produces synchronous dynamically-scheduled circuits from C/C++ code. Dynamatic generates synthesizable RTL which currently targets Xilinx FPGAs and delivers significant performance improvements compared to state-of-the-art commercial HLS tools in specific situations (e.g., applications with irregular memory accesses or control-dominated code). The fully automated compilation flow of Dynamatic is based on MLIR. It is customizable and extensible to target different hardware platforms and easy to use with commercial tools such as Vivado (Xilinx) and Modelsim (Mentor Graphics).

## Building the Project

The following instructions can be used to setup Dynamatic from source.

1. **Install dependencies required by the project.** These include working C and C++ toolchains (compiler, linker), `cmake` and `ninja` for building the project, and `git`. For example, on apt-based Linux distributions:
    
    ```sh
    $ sudo apt-get install git cmake ninja-build clang lld ccache graphviz
    ```

    `clang`, `lld`, and `ccache` are not stictly required but significantly speed up (re)builds. If you do not wish to install them, pass the `--disable-build-opt` flag to the build script to disable their use when building.

2. **Clone the project and its submodules.** Dynamatic depends on a fork of [Polygeist](https://github.com/EPFL-LAP/Polygeist) (C/C++ frontend for MLIR), which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project).
    
    ```sh
    $ git clone --recurse-submodules git@github.com:EPFL-LAP/dynamatic.git
    $ cd dynamatic
    ```

    *Note:* The repository is set up so that Polygeist is shallow cloned by default, meaning the clone command downloads just enough of it to check out the currently specified commit. If you wish to work with the full history of this repository, you can manually unshallow it.

    ```sh
    $ cd polygeist
    $ git fetch --unshallow
    ```

3. **Build the project.** Run the build script from the directory created by the clone command (pass it the `--help` flag to see available build options).

    ```sh
    $ cd dynamatic
    $ chmod +x ./build.sh
    $ ./build.sh
    ```

    The build script creates `build` folders in the top level directory and in each submodule to run the build tasks from. All files generated during build (libraries, executable binaries, intermediate compilation files) are placed in these folders, which the repository is configured to not track. Additionally, the build script creates a `bin` folder in the top level directory that contains symbolic links to a number of executable binaries built by the superproject and subprojects that Dynamatic users may especially care about.

4. **Run the Dynamatic test-suite.** After building the project, or at any time during development, regression-tests for Dynamatic can be ran from the top level `build` folder using `ninja`.

    ```sh
    $ cd dynamatic/build
    $ ninja check-dynamatic
    ```
