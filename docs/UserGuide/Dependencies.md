# Dependencies
Dynamatic uses a number of libraries and tools to implement its full functionality. This document provides a list of these dependencies with some information on them.  

## Libraries
### Git Submodules  
Dynamatic uses [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) to manage its software dependencies (all hosted on GitHub). We depend on [Polygeist](https://github.com/llvm/Polygeist), a C/C++ frontend for MLIR which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project) through a git submodule. The project is set up so that you can include LLVM/MLIR headers directly from Dynamatic code without having to specify their path through Polygeist. We also depend on [godot-cpp](https://github.com/godotengine/godot-cpp), the official C++ bindings for the Godot game engine which we use as the frontend to our interactive dataflow circuit visualizer. See the [git submodules guide](./SubModulesGuide.md) for a summary on how to work with submodules in this project.

### Polygeist

[Polygeist](https://github.com/llvm/Polygeist) is a C/C++ frontend for MLIR including polyhedral optimizations and parallel optimizations features. Polygeist is thus responsible for the first step of our compilation process, that is taking source code written in C/C++ into the MLIR ecosystem. In particular, we care that our entry point to MLIR is at a very high semantic level, namely, at a level where polyhedral analysis is possible. The latter allows us to easily identify dependencies between memory accesses in source programs in a very accurate manner, which is key to optimizing the allocation of memory interfaces and resources in our elastic circuits down the line. Polygeist is able to emit MLIR code in the [*Affine*](https://mlir.llvm.org/docs/Dialects/Affine/) dialect, which is perfectly suited for this kind of analysis. 

### CMake & Ninja
These constitute the primary build system for Dynamatic. They are used to build Dynamatic core, Polygeist, and LLVM/MLIR. You can have more details on [CMake](https://cmake.org/cmake/help/latest/index.html) and [Ninja](https://ninja-build.org/manual.html) by checking their official documentations.

### Boost.Regex
[Boost.Regex](https://www.boost.org/doc/libs/1_88_0/libs/regex/doc/html/index.html) is used to resolve Dynamatic regex expressions.

## Scripting & Tools
### Python (≥ 3.6)
Used in build systems, scripting, testing. See official [documentation](https://docs.python.org/3/)

### Graphviz (dot)
Generates visual representations of dataflow circuits (i.e., .dot). See official [documentation](https://graphviz.org/documentation/)

### JDK (Java Development Kit)
Required to run Scala/Chisel compilation. See official [documentation](https://docs.oracle.com/en/java/javase/17/).

## Tools
Dynamatic uses some third party tools to implement smart buffer placement, simulation, and interactive dataflow circuit visualization. Below is a list of the tools:

### Optimization & Scheduling: Gurobi
[Gurobi](https://www.gurobi.com/) solves MILP (Mixed-Integer Linear Programming) problems used during buffer placement and optimization. Dynamatic is still functional without Gurobi, but the resulting circuits often fail to achieve acceptable performance. See how to set up gurobi in the [advanced build section](../UserGuide/AdvancedBuild.md)

### Simulation Tool: ModelSim/Questa
Dynamatic uses [ModelSim](https://www.intel.com/content/www/us/en/software-kit/750368/modelsim-intel-fpgas-standard-edition-software-version-18-1.html)/[Questa](https://www.intel.com/content/www/us/en/software-kit/849791/questa-intel-fpgas-standard-edition-software-version-24-1.html) to perform simulations. See [installation](../UserGuide/AdvancedBuild.md#6-modelsimquesta-installation) page on how to setup ModelSim/Questa.

### Graphical Tools: Godot
[godot-cpp](https://github.com/godotengine/godot-cpp), the official C++ bindings for the Godot game engine which we use as the frontend to our interactive dataflow circuit visualizer.

### Utility/Development Tools
#### `clang`, `lld`, `ccache`
These are optional compiler/linker improvements to speed up builds. See their official documentations for details.

### Git
Dynamatic uses [git]() for project and submodule version control

### Standard UNIX Toolchain: `curl`, `gzip`, etc.
These are used for the various build scripts in the Dynamatic project.
