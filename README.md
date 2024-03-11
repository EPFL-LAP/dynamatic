# Dynamatic

Dynamatic is an academic, open-source high-level synthesis compiler that produces synchronous dynamically-scheduled circuits from C/C++ code. Dynamatic generates synthesizable RTL which currently targets Xilinx FPGAs and delivers significant performance improvements compared to state-of-the-art commercial HLS tools in specific situations (e.g., applications with irregular memory accesses or control-dominated code). The fully automated compilation flow of Dynamatic is based on MLIR. It is customizable and extensible to target different hardware platforms and easy to use with commercial tools such as Vivado (Xilinx) and Modelsim (Mentor Graphics).

We welcome contributions and feedback from the community. If you would like to participate, please check out our [contribution guidelines](docs/GettingStarted.md#contributing) and/or join our [Zulip community server](https://dynamatic.zulipchat.com/join/kb5xdsftwz2gr76rlxqa6vz5/).

## Building the project

The following instructions can be used to setup Dynamatic from source. If you intend to modify Dynamatic's source code and/or build the interactive dataflow circuit visualizer, you can check our [advanced build instructions](docs/AdvancedBuild.md) to learn how to customize the build process to your needs.

1. **Install dependencies required by the project.** These are: working C and C++ toolchains (compiler, linker), `cmake` and `ninja` for building the project, Python (3.6 or newer), GraphViz to work with `.dot` files, Boost's regex library, and `git`. For example, on apt-based Linux distributions (requires `sudo`):
    
    ```sh
    sudo apt-get update
    sudo apt-get install clang lld ccache cmake ninja-build python3 graphviz libboost-regex-dev git 
    ```

    `clang`, `lld`, and `ccache` are not stictly required but significantly speed up (re)builds. If you do not wish to install them, call the build script with the `--disable-build-opt` flag to prevent their usage.

2. **Clone the project and its submodules.** Dynamatic depends on a fork of [Polygeist](https://github.com/EPFL-LAP/Polygeist) (C/C++ frontend for MLIR), which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project).
    
    ```sh
    # Either clone with HTTPS...
    git clone --recurse-submodules https://github.com/EPFL-LAP/dynamatic.git
    # ...or SSH
    git clone --recurse-submodules git@github.com:EPFL-LAP/dynamatic.git
    ```

    This creates a `dynamatic` folder in your current working directory. 

3. **Build the project.** Run the build script from the directory created by the clone command (check out our [advanced build instructions](docs/AdvancedBuild.md) to see how you can customize the build process and/or build the interactive dataflow visualizer).

    ```sh
    cd dynamatic
    chmod +x ./build.sh
    ./build.sh --release
    ```

    If everything went well, you should see `# ===--- Build successful! ---===` displayed at the end of the build script's output.

4. **Run the Dynamatic testsuite.** After building the project, or at any time during development, you can run Dynamatic's testsuite from the top-level `build` folder using `ninja`.

    ```sh
    # From the "dynamatic" folder created by the clone command
    cd build
    ninja check-dynamatic
    ```
