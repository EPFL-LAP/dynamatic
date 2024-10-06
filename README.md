# Dynamatic

Dynamatic is an academic, open-source high-level synthesis compiler that produces synchronous dynamically-scheduled circuits from C/C++ code. Dynamatic generates synthesizable RTL which currently targets Xilinx FPGAs and delivers significant performance improvements compared to state-of-the-art commercial HLS tools in specific situations (e.g., applications with irregular memory accesses or control-dominated code). The fully automated compilation flow of Dynamatic is based on MLIR. It is customizable and extensible to target different hardware platforms and easy to use with commercial tools such as Vivado (Xilinx) and Modelsim (Mentor Graphics).

We welcome contributions and feedback from the community. If you would like to participate, please check out our [contribution guidelines](docs/ContribGuide.md) and/or join our [Zulip community server](https://dynamatic.zulipchat.com/join/kb5xdsftwz2gr76rlxqa6vz5/).

## Setting up Dynamatic

There are currently two ways to setup and use Dynamatic locally.

1. **Build from source (recommended)**. We support building from source on Linux and on Windows (through [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). See our [simple build instructions](#building-from-source) below, which should work out-of-the-box for any apt-based or pacman-based Linux distribution. Different distributions may require cosmetic changes to the dependencies you have to install before running Dynamatic.

2. **Use the provided virtual machine**. We provide an [Ubuntu-based virtual machine](docs/VMSetup.md) (VM) that already has Dynamatic, Modelsim, and our dataflow circuit visualizer set up. This machine was originally set-up for the [*Dynamatic Reloaded* tutorial given at the FPGA'24 conference](https://www.isfpga.org/workshops-tutorials/#t7) in Monterey, California. You can use it to simply follow the tutorial (available in the [repository's documentation](docs/Tutorials/Introduction/Introduction.md)) or as a starting point to use/modify Dynamatic in general.

Also, please note the following post-install recommendations. They are necessary for running the integration tests successfully.

1.  **Gurobi Optimization** is an optional dependency, necessary for compiling with MILP-based placement algorithms. After installing Gurobi, add the following lines in your `~/.bashrc` or `~/.zshrc` (replace `<version>` with your version number of Gurobi, e.g. `1103`, and change the path to match your installation path):
    ```sh
    export GUROBI_HOME="/opt/gurobi<version>/linux64"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
    ```
    These environment variables are used by Dynamatic's cmake settings to determine how to include Gurobi headers. **Rerun the build script** after adding the environment variables in order for cmake to recognize them.

2.  Although Dynamatic can be used in conjunction with any synthesis and simulation tools, it is strongly recommended to use **Vivado** for synthesis and **ModelSim** for simulation. When installing them, make sure you use versions that are [compatible with each other](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Compatible-Third-Party-Tools). As with Gurobi in the previous step, add the following environment variables (also replace `<version>` and change the path if necessary):
    ```sh
    export XILINX_HOME="/opt/Xilinx/Vivado/<version>"
    export MODELSIM_HOME="/opt/intelFPGA/<version>/modelsim_ase"
    ```
    Again, **rerun the build script**.

3.  In order to simulate floating-point circuits, you will need to [compile floating-point AMD/Xilinx libraries](docs/FPSimulation.md).

## Using Dynamatic

To get started using Dynamatic (after setting it up), check out our [introductory tutorial](docs/Tutorials/Introduction/Introduction.md), which guides you through your first compilation of C code into a synthesizable dataflow circuit! If you want to start modifying Dynamatic and are new to MLIR or compilers in general, our [MLIR primer](docs/Tutorials/MLIRPrimer.md) and [pass creation tutorial](docs/Tutorials/CreatingPasses/CreatingPasses.md) will help you take your first steps.

For an high-level overview of the project's structure and of our contribution guidelines, see our [*Contribution Guide*](docs/ContribGuide.md).

## Building from source

The following instructions can be used to setup Dynamatic from source. If you intend to modify Dynamatic's source code and/or build the interactive dataflow circuit visualizer, you can check our [advanced build instructions](docs/AdvancedBuild.md) to learn how to customize the build process to your needs.

1. **Install dependencies required by the project.**

    Most of our dependencies are provided as standard packages on most Linux distributions. Dynamatic needs a working C/C++ toolchain (compiler, linker), `cmake` and `ninja` for building the project, Python (3.6 or newer), a recent JDK (Java Development Kit) for Scala, GraphViz to work with `.dot` files, Boost's regex library, and standard command-line tools like `git`.

    On `apt`-based Linux distributions:

    ```sh
    apt-get update
    apt-get install clang lld ccache cmake ninja-build python3 openjdk-21-jdk graphviz libboost-regex-dev git curl gzip libreadline-dev
    ```

    On `pacman`-based Linux distributions:

    ```sh
    pacman -Syu
    pacman -S clang lld ccache cmake ninja python jdk21-openjdk graphviz boost git curl gzip readline
    ```

    `clang`, `lld`, and `ccache` are not stictly required but significantly speed up (re)builds. If you do not wish to install them, in step 3 call the build script with the `--disable-build-opt` flag to prevent their usage.

    Dynamatic uses RTL generators written in Chisel---a hardware construction language embedded in the high-level programming language Scala---to produce synthesizable RTL designs. You can install Scala [using the recommended way](https://www.scala-lang.org/download/) with the following command.

    ```sh
    curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup
    ```

    Before moving on to the next step, refresh your environment variables in your current terminal to make sure that all newly installed tools are visible in your `PATH`. Alternatively, open a new terminal and proceed to cloning the project.

2. **Clone the project and its submodules.** Dynamatic depends on a fork of [Polygeist](https://github.com/EPFL-LAP/Polygeist) (C/C++ frontend for MLIR), which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project). You need to clone with the SSH link to be able to push to the repository.

    ```sh
    # Either clone with SSH... (required for pushing to the repository)
    git clone --recurse-submodules git@github.com:EPFL-LAP/dynamatic.git
    # ...or HTTPS (if you only ever intend to pull from the repository)
    git clone --recurse-submodules https://github.com/EPFL-LAP/dynamatic.git
    ```

    This creates a `dynamatic` folder in your current working directory.

3. **Build the project.** Run the build script from the directory created by the clone command (check out our [advanced build instructions](docs/AdvancedBuild.md) to see how you can customize the build process and/or build the interactive dataflow visualizer).

    ```sh
    cd dynamatic
    chmod +x ./build.sh
    ./build.sh --release
    ```

    If everything went well, you should see `===--- Build successful! ---===` displayed at the end of the build script's output.

4. **Run the Dynamatic testsuite.** After building the project, or at any time during development, you can run Dynamatic's testsuite from the top-level `build` folder using `ninja`.

    ```sh
    # From the "dynamatic" folder created by the clone command
    cd build
    ninja check-dynamatic
    ```
    You can also run the integration tests using the provided Dynamatic [script](tools/dynamatic/samples/run-all-tests.dyn).
    ```sh
    bin/dynamatic --run=/tools/dynamatic/samples/run-all-tests.dyn
    ```
