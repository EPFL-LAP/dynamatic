# Advanced Build Instructions

This document contains advanced build instructions targeted at users who would like to modify Dynamatic's source code and/or use the interactive dataflow circuit visualizer. For basic setup instructions, see the [top-level README file](../README.md#building-the-project).

> [!NOTE]
> In the instructions below, we assume that you have already cloned Dynamatic and its submodules and that the project is rooted in a folder called `dynamatic`. Whenever provided shell commands contain `cd dynamatic`, it refers to this directory created during cloning. Adjust paths as needed depending on your current working directory.  

## Cloning

The repository is set up so that Polygeist and LLVM are shallow cloned by default, meaning the clone command downloads just enough of them to check out currently specified commits. If you wish to work with the full history of these repositories, you can manually unshallow them after cloning.

For Polygeist:

```sh
cd dynamatic/polygeist
git fetch --unshallow
```

For LLVM:

```sh
cd dynamatic/polygeist/llvm-project
git fetch --unshallow
```

## Building

This section provides some insights into our custom build script, [build.sh](../build.sh), located in the repository's top-level folder. The script recognizes a number of flags and arguments that allow you to customize the build process to your needs. The `--help` flag makes the script print the entire list of available flags/arguments and exit.

> [!WARNING]
> The script should always be ran from Dynamatic's top-level folder. 

### General behavior

The build script successively builds all parts of the project using *CMake* and *Ninja*. In order, it builds
1. LLVM (with *MLIR* and *clang* as additional tools),
2. Polygeist (our C/C++ frontend for MLIR),
3. Dynamatic, and
4. (optionally) the interactive dataflow circuit visualizer ([see instructions below](#interactive-dataflow-circuit-visualizer)).

It creates `build` folders in the top level directory and in each submodule to run the build tasks from. All files generated during build (libraries, executable binaries, intermediate compilation files) are placed in these folders, which the repository is configured to not track. Additionally, the build script creates a `bin` folder in the top-level directory that contains symbolic links to a number of executable binaries built by the superproject and subprojects that Dynamatic users may especially care about.

### *Debug* or *Release* mode

The build script builds the entire project in *Debug* mode by default, which enables assertions in the code and gives you access to runtime debug information that is very useful when working on Dynamatic's code. However, *Debug* mode increases build time and (especially) build size (the project takes around 60GB once fully built). If you do not care for runtime debug information and/or want Dynamatic to have a smaller footprint on your disk, you can instead build Dynamatic in *Release* mode by using the `--release` flag when running the build script.

```sh
# Build Dynamatic in Debug mode
./build.sh
```

```sh
# Build Dynamatic in Release mode
./build.sh --release
```

### Multi-threaded builds

By default, *Ninja* builds the project by concurrently using at most one thread per logical core on your machine. This can put a lot of strain on your system's CPU and RAM, preventing you from using other applications smoothly or making you run out of RAM (especially during linking of LLVM/MLIR). You can customize the maximum number of concurrent threads that are used to build the project using the `--threads` argument.

```sh
# Build using at most one thread per logical core on your machine
./build.sh
```

```sh
# Build using at most 4 concurrent threads
./build.sh --threads 4
```

### Forcing CMake re-configuration

To reduce the build script's execution time when re-building the project regularly (which happens during active development), the script does not try to fully reconfigure each submodule or the superproject using *CMake* if it sees that a *CMake* cache is already present on your filesystem for each part. This can cause problems if you suddenly decide to change build flags that affect the CMake configuration (e.g., when going from a *Debug* build to a *Release* build) as the CMake configuration will not take into account the new configuration. Whenever that happens (or whenever in doubt), provide the `--force` flag to force the build script to re-configure each part of the project using CMake. 

```sh
# Force re-configuration of every submodule and the superproject
./build.sh --force
```

> [!NOTE]
> If the *CMake* configuration of each submodule and of the superproject has not changed since the last build script's invocation and the `--force` flag is provided, the script will just take around half a minute more to run than normal but will not fully re-build everything. Therefore it is safe and not too inconvenient to specify the `--force` flag on every invocation of the script. 

### Interactive dataflow circuit visualizer

The repository contains an optionally built tool that allows to visualize the dataflow circuits produced by Dynamatic and interact with them as they are simulated on test inputs. This is a very useful tool for debugging and for better understanding dataflow circuits in general. It is built on top of the open-source [Godot game engine](https://godotengine.org/) and of its [C++ bindings](https://github.com/godotengine/godot-cpp), the latter of which Dynamatic depends on as a submodule rooted at `visual-dataflow/godot-cpp` (relative to Dynamatic's top-level folder). To build and/or modify this tool (which is only supported on Linux at this point), one must therefore download the Godot engine (a single executable file) from the Internet manually.

> [!WARNING]
> Note that Godot's C++ bindings only work for a specific major/minor version of the engine. This version is specified in the `branch` field of the submodule's declaration in [`.gitmodules`](../.gitmodules). The version of the engine you download must therefore match the bindinds currently tracked by Dynamatic. [You can download any version of Godot from the official archive](https://godotengine.org/download/archive/).

Due to these extra dependencies, building this tool is opt-in, meaning that by default it is not built along the rest of Dynamatic. This also means that the `CMakeLists.txt` file in `visual-dataflow/` is meant to be configured independently from the one located one folder above it i.e., at the project's root. As a consequence, intermediate build files for the tool are dumped into the `visual-dataflow/build/` folder instead of the top-level `build/` folder.  

Building an executable binary for the interactive dataflow circuit visualizer is a two-step process, one which is automated and one which still requires some manual work detailed below.
1. First, one must build the C++ shared library that the Godot project uses to get access to Dynamatic's API. The `--visual-dataflow` build script flag performs this task automatically.
  
    ```sh
    # Build the C++ library needed by the dataflow visualizer along the rest of Dynamatic 
    ./build.sh --visual-dataflow
    ```

    At this point, it becomes possible to open the Godot project in the Godot editor and modify/run it from there.

2. Second, one must export the Godot project as an executable binary to be able to run it from outside the editor. In addition to having downloaded the Godot engine, at the moment this also requires that the project has been exported manually once from the Godot editor. The Godot documentation details the process [here](https://docs.godotengine.org/en/stable/tutorials/export/exporting_projects.html#export-menu), which you only need to follow up to and including the part where it asks you to download export templates using the graphical interface. Once they are downloaded for your specific export target, you are now able to automatically build the tool by using the `--export-godot` build script argument and specifying the path to the Godot engine executable you downloaded.

    ```sh
    # Export the Godot project as an executable binary
    # Here it is a good idea to also provide the --visual-dataflow flag to ensure
    # that the C++ library needed by the dataflow visualizer is up-to-date 
    ./build.sh --visual-dataflow --export-godot /path/to/godot-engine
    ```

    The tool's binary is generated at `visual-dataflow/bin/visual-dataflow` and sym-linked at `bin/visual-dataflow` for convenience. Whenever you make a modification to the C++ library or to the Godot project itself, you can simply re-run the above command to recompile everything and re-generate the executable binary for the tool.
