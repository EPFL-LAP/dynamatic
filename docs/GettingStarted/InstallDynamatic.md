# Setting up Dynamatic

There are currently two ways to setup and use Dynamatic

**1. Build From Source (Recommended)**  
We support building from source on Linux and on Windows (through [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). See our [Build instructions](#build-instructions) below. Ubuntu 24.04 LTS is officially supported; other apt-based distributions should work as well. Other distributions may also require cosmetic changes to the dependencies you have to install before running Dynamatic.

**2. Use the Provided Virtual Machine**  
We provide an [Ubuntu-based Virtual Machine](./VMSetup.md) (VM) that already has Dynamatic and our dataflow circuit visualizer set up. You can use it to simply follow the tutorial ([Using Dynamatic](../GettingStarted/Tutorials/Introduction/UsingDynamatic.md)) or as a starting point to use/[modify](../DeveloperGuide/IntroductoryMaterial/Tutorials/CreatingPasses/CreatingPassesTutorial.md) Dynamatic in general.  

### Build Instructions
The following instructions can be used to setup Dynamatic from source.  
> [!NOTE]
> If you intend to modify Dynamatic's source code and/or build the interactive dataflow circuit visualizer (recommended for circuit debugging), you can check our [advanced build instructions](../UserGuide/AdvancedBuild.md#3-building) to learn how to customize the build process to your needs.

**1. Install Dependencies Required by the Project**  
Most of our dependencies are provided as standard packages on most Linux distributions. Dynamatic needs a working C/C++ toolchain (compiler, linker), cmake and ninja for building the project, Python (3.6 or newer), a recent JDK (Java Development Kit) for Scala, GraphViz to work with .dot files, and standard command-line tools like git.
> [!NOTE]  
> You will need at least 50GB of internal storage to compile the llvm-project and 16GB+ of memory is recommended to facilitate the linking process

On `apt`-based Linux distributions:
```sh
apt-get update
apt-get install clang lld ccache cmake ninja-build python3 openjdk-21-jdk graphviz git curl gzip libreadline-dev libboost-all-dev
```
Note that you may need super user privileges for any package installation. You can use **sudo** before entering the commands

`clang`, `lld`, and `ccache` are not strictly required but significantly speed up (re)builds. If you do not wish to install them, call the build script with the --disable-build-opt flag to prevent their usage.

Dynamatic uses RTL generators written in Chisel (a hardware construction language embedded in the high-level programming language Scala) to produce synthesizable RTL designs. You can install Scala using the recommended way with the following command:
```sh
curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup
```

Dynamatic utilizes Gurobi to optimize the circuit's performance. It is optional and Dynamatic will build properly without it but is useful for more optimized results. Refer to our [Advanced Build](../UserGuide/AdvancedBuild.md) page for guidance on [how to setup the Gurobi solver](../UserGuide/AdvancedBuild.md#1-gurobi).

> [!TIP]
> While this section helps you install the dependencies needed to get started with Dynamatic, you can find a list of dependencies used by Dynamatic in the [dependencies](../UserGuide/Dependencies.md) section for a better understanding of how the tool works.

Finally, Dynamatic uses [Modelsim](https://www.intel.com/content/www/us/en/software-kit/750666/modelsim-intel-fpgas-standard-edition-software-version-20-1-1.html) or [Questa](https://www.intel.com/content/www/us/en/software-kit/849791/questa-intel-fpgas-standard-edition-software-version-24-1.html) to run simulations.  
These are optional tools which you can see how to [install](../UserGuide/AdvancedBuild.md#6-modelsimquesta-installation) in the [Advanced Build](../UserGuide/AdvancedBuild.md#6-modelsimquesta-installation) page if you intend to use the simulator.  

> [!TIP]  
> Before moving on to the next step, refresh your environment variables in your current terminal to make sure that all newly installed tools are visible in your PATH. Alternatively, open a new terminal and proceed to cloning the project.

**2. Cloning the Project and Its Submodules**  
Dynamatic depends on a fork of [Polygeist](https://github.com/EPFL-LAP/Polygeist) (C/C++ frontend for MLIR), which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project). To instruct git to clone the appropriate versions submodules used by Dynamatic, we enable the `--recurse-submodules` flag.  
```sh
git clone --recurse-submodules https://github.com/EPFL-LAP/dynamatic.git
```
This creates a `dynamatic` folder in your current working directory.

**3. Build the Project**  
Run the build script from the directory created by the clone command (see the [advanced build](../UserGuide/AdvancedBuild.md#3-building) instructions for details on how to customize the build process).
```sh
cd dynamatic
chmod +x ./build.sh
./build.sh --release
```

**4. Run the Dynamatic Testsuite**  
To confirm that you have successfully compiled Dynamatic and to test its functionality, you can run Dynamatic's testsuite from the top-level `build` folder using `ninja`.
```sh
# From the "dynamatic" folder created by the clone command
cd build
ninja check-dynamatic
```

You can now launch the Dynamatic front-end from Dynamatic's top level directory using:
```sh
./bin/dynamatic
```
With Dynamatic correctly installed, you can browse the [using dynamatic](../GettingStarted/Tutorials/Introduction/Introduction.md) tutorial to learn how to use the basic commands and features in Dynamatic to convert your C code into RTL.  
You can also explore the [Advanced build](../UserGuide/AdvancedBuild.md) options.

[Go to top of the page](#setting-up-dynamatic)
