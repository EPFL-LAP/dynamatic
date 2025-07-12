[Return to table of contents](../README.md)

# Setting up Dynamatic

There are currently two ways to setup and use Dynamatic

**1. Build From Source (Recommended)**  
We support building from source on Linux and on Windows (through [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). See our [Build instructions](#build-instructions) below. Ubuntu 24.04 LTS is officially supported; other apt-based distributions should work as well. Other distributions may also require cosmetic changes to the dependencies you have to install before running Dynamatic.

**2. Use the Provided Virtual Machine**  
We provide an [Ubuntu-based Virtual Machine](VMSetup.md) (VM) that already has Dynamatic and our dataflow circuit visualizer set up. You can use it to simply follow the tutorial ([Using Dynamatic](../GettingStarted/Tutorials/Introduction/UsingDynamatic.md)) or as a starting point to use/[modify](../DeveloperGuide/CreatingPasses/CreatingPasses.md) Dynamatic in general.  
> [!NOTE]
> You will need to install Modelsim or Questa manually to run simulations! Click [here](../UserGuide/AdvancedBuild.md#6-modelsimquesta-installation) for instructions on installing them.


### Build Instructions
The following instructions can be used to setup Dynamatic from source.  
> [!NOTE]
> If you intend to modify Dynamatic's source code and/or build the interactive dataflow circuit visualizer (recommended for circuit debugging), you can check our [advanced build instructions](../UserGuide/AdvancedBuild.md#3-building) to learn how to customize the build process to your needs.

**1. Install Dependencies Required by the Project**  
Dynamatic uses  
- [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) to manage its software dependencies (all hosted on GitHub). 
- [Polygeist](https://github.com/llvm/Polygeist), a C/C++ frontend for MLIR which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project) through a git submodule. The project is set up so that you can include LLVM/MLIR headers directly from Dynamatic code without having to specify their path through Polygeist. 
- [godot-cpp](https://github.com/godotengine/godot-cpp), the official C++ bindings for the Godot game engine which we use as the frontend to our interactive dataflow circuit visualizer.
- [Modelsim](https://www.intel.com/content/www/us/en/software-kit/750368/modelsim-intel-fpgas-standard-edition-software-version-18-1.html)/[Questa](https://www.intel.com/content/www/us/en/software-kit/849791/questa-intel-fpgas-standard-edition-software-version-24-1.html) for our simulation tool. See [installation](docs/UserGuide/AdvancedBuild.md) page on how to setup Modelsim/Questa.
- [Gurobi](https://www.gurobi.com/) to solve performance-related optimization problems. Dynamatic is still functional without Gurobi, but the resulting circuits often fail to achieve acceptable performance. See how to set up gurobi in the [advanced build section](docs/UserGuide/AdvancedBuild.md)

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

Dynamatic utilizes Gurobi to optimize the circuit's performance. Refer to our tutorial for guidance on [how to setup the Gurobi solver](../UserGuide/AdvancedBuild.md#1-gurobi).

> [!TIP]
> While this section helps you install the dependencies needed to get started with Dynamatic, you can find a list of dependencies used by Dynamatic in the [dependencies](../UserGuide/Dependencies.md) section for a better understanding of how the tool works.

Finally, Dynamatic uses [Modelsim](https://www.intel.com/content/www/us/en/software-kit/750666/modelsim-intel-fpgas-standard-edition-software-version-20-1-1.html) or [Questa](https://www.intel.com/content/www/us/en/software-kit/849791/questa-intel-fpgas-standard-edition-software-version-24-1.html) to run simulations, thus you need to [install](../UserGuide/AdvancedBuild.md#6-modelsimquesta-installation) it before hand and add it to your environment variables.  

Before moving on to the next step, refresh your environment variables in your current terminal to make sure that all newly installed tools are visible in your PATH. Alternatively, open a new terminal and proceed to cloning the project.

**2. Cloning the Project and Its Submodules**  
Dynamatic depends on a fork of [Polygeist](https://github.com/EPFL-LAP/Polygeist) (C/C++ frontend for MLIR), which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project). You need to clone with the SSH link to be able to push to the repository.
```
# Either clone with SSH... (required for pushing to the repository)
git clone --recurse-submodules git@github.com:EPFL-LAP/dynamatic.git
# ...or HTTPS (if you only ever intend to pull from the repository)
git clone --recurse-submodules https://github.com/EPFL-LAP/dynamatic.git
```
This creates a `dynamatic` folder in your current working directory.

**3. Build the Project**  
Run the build script from the directory created by the clone command (check out our [advanced build](../UserGuide/AdvancedBuild.md) instructions to see how you can customize the build process and/or build the interactive dataflow visualizer). You may want to check the [build](../UserGuide/AdvancedBuild.md#3-building) section of the advanced build page if you want more options for building such as multi-threaded build which is faster.
```
cd dynamatic
chmod +x ./build.sh
./build.sh --release
```

**4. Run the Dynamatic Testsuite**  
After building the project, or at any time during development, you can run Dynamatic's testsuite from the top-level ```build``` folder using ```ninja```.
```
# From the "dynamatic" folder created by the clone command
cd build
ninja check-dynamatic
```
Now, you have the groundwork for [using dynamatic](../GettingStarted/Tutorials/Introduction/UsingDynamatic.md) and trying the [Advanced build](../UserGuide/AdvancedBuild.md) options.
For information on commands and features, please check out [command reference](../UserGuide/CommandReference.md).

[Go to top of the page](#setting-up-dynamatic)