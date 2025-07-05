[Home](README.md) <span>&ensp;</span> [Usage](docs/UserGuideTopics/Usage.md)<span>&ensp;</span> [Modification](docs/UserGuideTopics/AdvancedUsage.md)<span>&ensp;</span> [Advanced-Build](docs/UserGuideTopics/AdvancedBuild.md) <span>&ensp;</span>[Examples](docs/UserGuideTopics/Examples.md) <span>&ensp;</span>[Dependencies](docs/UserGuideTopics/Dependencies.md) <span>&ensp;</span>[Development](docs/UserGuideTopics/WorkInProgress.md)
# Dynamatic
Dynamatic is an academic, open-source high-level synthesis compiler that produces synchronous dynamically-scheduled circuits from C/C++ code. Dynamatic generates synthesizable RTL which currently targets Xilinx FPGAs and delivers significant performance improvements compared to state-of-the-art commercial HLS tools in specific situations (e.g., applications with irregular memory accesses or control-dominated code). The fully automated compilation flow of Dynamatic is based on MLIR. It is customizable and extensible to target different hardware platforms and easy to use with commercial tools such as Vivado (Xilinx) and Modelsim (Mentor Graphics).

We welcome contributions and feedback from the community. If you would like to participate, please check out our [contribution guidelines](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/GettingStarted.md#contributing).

## Setting up Dynamatic

There are currently two ways to setup and use Dynamatic

**1. Build from Source (recommended)**  
We support building from source on Linux and on Windows (through [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). See our [Build instructions](#build-instructions) below. Ubuntu 24.04 LTS is officially supported; other apt-based distributions should work as well. Other distributions may also require cosmetic changes to the dependencies you have to install before running Dynamatic.

**2. Use the Provided Virtual Machine**  
We provide an [Ubuntu-based Virtual Machine](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/VMSetup.md) (VM) that already has Dynamatic, Modelsim, and our dataflow circuit visualizer set up. You can use it to simply follow the tutorial ([Using Dynamatic](docs/UserGuideTopics/usage.md)) or as a starting point to use/[modify](docs/UserGuideTopics/AdvancedUsage.md) Dynamatic in general.


### Build Instructions
The following instructions can be used to setup Dynamatic from source.  
>If you intend to modify Dynamatic's source code and/or build the interactive dataflow circuit visualizer (recommended for circuit debugging), you can check our [advanced build instructions](docs/UserGuideTopics/AdvancedBuild.md) to learn how to customize the build process to your needs.

**1. Install dependencies required by the project**  
Most of our dependencies are provided as standard packages on most Linux distributions. Dynamatic needs a working C/C++ toolchain (compiler, linker), cmake and ninja for building the project, Python (3.6 or newer), a recent JDK (Java Development Kit) for Scala, GraphViz to work with .dot files, and standard command-line tools like git.

On `apt`-based Linux distributions:
```
apt-get update
apt-get install clang lld ccache cmake ninja-build python3 openjdk-21-jdk graphviz git curl gzip libreadline-dev
```
Note that you may need super user privileges for any package installation. You can use **sudo** before entering the commands

`clang`, `lld`, and `ccache` are not strictly required but significantly speed up (re)builds. If you do not wish to install them, call the build script with the --disable-build-opt flag to prevent their usage.

Dynamatic uses RTL generators written in Chisel (a hardware construction language embedded in the high-level programming language Scala) to produce synthesizable RTL designs. You can install Scala using the recommended way with the following command:
```
curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup
```

Dynamatic utilizes Gurobi to optimize the circuit's performance. Refer to our tutorial for guidance on [how to setup the Gurobi solver](docs/UserGuideTopics/AdvancedBuild.md#1-gurobi).

>While this section helps you install the dependencies needed to get started with Dynamatic, you can find a list of dependencies used by Dynamatic in the [dependencies](docs/UserGuideTopics/Dependencies.md) section for a better understanding of how the tool works.

Finally, Dynamatic uses [Modelsim](hhttps://www.intel.com/content/www/us/en/software-kit/750666/modelsim-intel-fpgas-standard-edition-software-version-20-1-1.html) to run simulations, thus you need to install it before hand and add it to your environment variables.  

Before moving on to the next step, refresh your environment variables in your current terminal to make sure that all newly installed tools are visible in your PATH. Alternatively, open a new terminal and proceed to cloning the project.

**2. Cloning the project and its submodules**  
Dynamatic depends on a fork of [Polygeist](https://github.com/EPFL-LAP/Polygeist) (C/C++ frontend for MLIR), which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project). You need to clone with the SSH link to be able to push to the repository.
```
# Either clone with SSH... (required for pushing to the repository)
git clone --recurse-submodules git@github.com:EPFL-LAP/dynamatic.git
# ...or HTTPS (if you only ever intend to pull from the repository)
git clone --recurse-submodules https://github.com/EPFL-LAP/dynamatic.git
```
This creates a `dynamatic` folder in your current working directory.

**3. Build the Project**  
Run the build script from the directory created by the clone command (check out our [advanced build](docs/UserGuideTopics/AdvancedBuild.md) instructions to see how you can customize the build process and/or build the interactive dataflow visualizer). You may want to check the [build](docs/UserGuideTopics/AdvancedBuild.md#3-building) section of the advanced build page if you want more options for building such as multi-threaded build which is faster.
```
cd dynamatic
chmod +x ./build.sh
./build.sh --release
```

**4. Run the Dynamatic testsuite**  
After building the project, or at any time during development, you can run Dynamatic's testsuite from the top-level ```build``` folder using ```ninja```.
```
# From the "dynamatic" folder created by the clone command
cd build
ninja check-dynamatic
```
Now, you have the groundwork for [using dynamatic](docs/UserGuideTopics/Usage.md) and trying the [Advanced build](docs/UserGuideTopics/AdvancedBuild.md) options.
For information on usage and features, please check out [the user manual](docs/UserManual.md).

[Go to top of the page](#installing-dynamatic)