# Advanced Build Instructions
### Table of contents
1. [Gurobi](#1-gurobi)
2. [Cloning](#2-cloning)
3. [Building](#3-building)
4. [Interactive Visualizer](#4-interactive-dataflow-circuit-visualizer)
5. [Enabling XLS Integration](#5-enabling-the-xls-integration)
6. [Modelsim/Questa sim installation](#6-modelsimquesta-installation)

> [!NOTE]
> This document contains advanced build instructions targeted at users who would like to modify Dynamatic's build process and/or use the interactive dataflow circuit visualizer. For basic setup instructions, see the [installation](../GettingStarted/InstallDynamatic.md) page.

## 1. Gurobi

#### Why Do We Need Gurobi?

Currently, Dynamatic relies on [Gurobi](https://www.gurobi.com/) to solve performance-related optimization problems ([MILP](https://en.wikipedia.org/wiki/Integer_programming)). Dynamatic is still functional without Gurobi, but the resulting circuits often fail to achieve acceptable performance.

#### Download Gurobi

Gurobi is available for Linux [here](https://www.gurobi.com/downloads/gurobi-software/) (log in required). The resulting downloaded file will be gurobiXX.X.X_linux64.tar.gz

#### Obtain a License

Free academic licenses for Gurobi are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

#### Installation

To install Gurobi, first extract your downloaded file to your desired installation directory. We recommend to place this in `/opt/`, e.g. `/opt/gurobiXXXX/linux64/` (with XXXX as the downloaded version). If extraction fails, try with sudo.

Use the following command to pass your obtained license to Gurobi, which it stores in  `~/gurobi.lic`


```sh
# Replace x's with obtained license
/opt/gurobiXXXX/linux64/bin/grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx 
```
> [!NOTE]  
> If you chose a web library (WLS license), copy the gurobi.lic file provided to your home directory rather than running the command above

#### Configuring Your Environment

In addition to adding Gurobi to your path, Dynamatic's CMake requires the GUROBI_HOME environment variable to find headers and libraries. These lines can be added to your shell initiation script, e.g. ~/.bashrc or ~/.zshrc, or used with any other environment setup method.

```sh
# Replace "gurobiXXXX" with the correct version
export GUROBI_HOME="/opt/gurobiXXXX/linux64"
export PATH="${GUROBI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:$LD_LIBRARY_PATH"
```

Once Gurobi is set up, you can change the buffer placement algorithm using the `--buffer-algorithm` compile flag and setting the value to either `fpga20` or `fpl22`. See [Using Dynamatic](../GettingStarted/Tutorials/Introduction/UsingDynamatic.md#compile-flags) page for details on how to use Dynamatic and modify the compile flags.

## 2. Cloning

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

## 3. Building

This section provides some insights into our custom build script, **build.sh**, located in the repository's top-level folder. The script recognizes a number of flags and arguments that allow you to customize the build process to your needs. The --help flag makes the script print the entire list of available flags/arguments and exit.
> [!NOTE]  
> The script should always be ran from Dynamatic's top-level folder.  

### General Behavior

The build script successively builds all parts of the project using CMake and Ninja. In order, it builds

  1. LLVM (with MLIR and clang as additional tools),
  2. Polygeist (our C/C++ frontend for MLIR),
  3. Dynamatic, and
  4. (optionally) the interactive dataflow circuit visualizer ([see instructions below](#4-interactive-dataflow-circuit-visualizer)).

It creates build folders in the top level directory and in each submodule to run the build tasks from. All files generated during build (libraries, executable binaries, intermediate compilation files) are placed in these folders, which the repository is configured to not track. Additionally, the build script creates a **`bin`** folder in the top-level directory that contains symbolic links to a number of executable binaries built by the superproject and subprojects that Dynamatic users may especially care about.

### Debug or Release Mode

The build script builds the entire project in Debug mode by default, which enables assertions in the code and gives you access to runtime debug information that is very useful when working on Dynamatic's code. However, Debug mode increases build time and (especially) build size (the project takes around 60GB once fully built). If you do not care for runtime debug information and/or want Dynamatic to have a smaller footprint on your disk, you can instead build Dynamatic in Release mode by using the `--release` flag when running the build script.

```sh
# Build Dynamatic in Debug mode
./build.sh
```

```sh
# Build Dynamatic in Release mode
./build.sh --release
```

### Multi-Threaded Builds

By default, Ninja builds the project by concurrently using at most one thread per logical core on your machine. This can put a lot of strain on your system's CPU and RAM, preventing you from using other applications smoothly. You can customize the maximum number of concurrent threads that are used to build the project using the --threads argument.

```sh
# Build using at most one thread per logical core on your machine
./build.sh
```

```sh
# Build using at most 4 concurrent threads
./build.sh --threads 4
```

It is also common to run out of RAM especially during linking of LLVM/MLIR. If this is a problem, consider limiting the maximum number of parallel LLVM link jobs to one per 15GB of available RAM, using the --llvm-parallel-link-jobs flag:

```sh
# Perform at most 1 concurrent LLVM link jobs
./build.sh --llvm-parallel-link-jobs 1
```

> [!NOTE]
> This flag defaults to a value of 2

### Forcing CMake Re-Configuration

To reduce the build script's execution time when re-building the project regularly (which happens during active development), the script does not try to fully reconfigure each submodule or the superproject using CMake if it sees that a CMake cache is already present on your filesystem for each part. This can cause problems if you suddenly decide to change build flags that affect the CMake configuration (e.g., when going from a Debug build to a Release build) as the CMake configuration will not take into account the new configuration. Whenever that happens (or whenever in doubt), provide the `--force` flag to force the build script to re-configure each part of the project using CMake.

```sh
# Force re-configuration of every submodule and the superproject
./build.sh --force
```

> [!TIP]
> If the CMake configuration of each submodule and of the superproject has not changed since the last build script's invocation and the --force flag is provided, the script will just take around half a minute more to run than normal but will not fully re-build everything. Therefore it is safe and not too inconvenient to specify the `--force` flag on every invocation of the script.

## 4. Interactive Dataflow Circuit Visualizer

The repository contains an optionally built tool that allows you to visualize the dataflow circuits produced by Dynamatic and interact with them as they are simulated on test inputs. This is a very useful tool for debugging and for better understanding dataflow circuits in general. It is built on top of the open-source [Godot game engine](https://godotengine.org/) and of its [C++ bindings](https://github.com/godotengine/godot-cpp), the latter of which Dynamatic depends on as a submodule rooted at visual-dataflow/godot-cpp (relative to Dynamatic's top-level folder). To build and/or modify this tool (which is only supported on Linux at this point), one must therefore download the Godot engine (a single executable file) from the Internet manually.
> [!NOTE]
> Godot's C++ bindings only work for a specific major/minor version of the engine. This version is specified in the branch field of the submodule's declaration in `.gitmodules`. The version of the engine you download must therefore match the bindings currently tracked by Dynamatic. You can [download any version of Godot from the official archive](https://godotengine.org/download/archive/).
>
Due to these extra dependencies, building this tool is opt-in, meaning that

- by default it is not built along the rest of Dynamatic.
- the CMakeLists.txt file in visual-dataflow/ is meant to be configured independently from the one located one folder above it i.e., at the project's root. As a consequence, intermediate build files for the tool are dumped into the `visual-dataflow/build/` folder instead of the top-level `build/` folder.

Building an executable binary for the interactive dataflow circuit visualizer is a two-step process, one which is automated and one which still requires some manual work detailed below.

1. Build the C++ shared library that the Godot project uses to get access to Dynamatic's API. The `--visual-dataflow` build script flag performs this task automatically.
```
# Build the C++ library needed by the dataflow visualizer along the rest of Dynamatic 
./build.sh --visual-dataflow
```

At this point, it becomes possible to open the Godot project (in the `/dynamatic/visual-dataflow` directory) in the Godot editor and modify/run it from there. Run your downloaded Godot file and open the project in the visual data-flow directory.

2. export the Godot project as an executable binary to be able to run it from outside the editor. In addition to having downloaded the Godot engine, at the moment this also requires that the project has been exported manually once from the Godot editor. The Godot documentation details the process [here](https://docs.godotengine.org/en/stable/tutorials/export/exporting_projects.html#export-menu), which you only need to follow up to and including the part where it asks you to download export templates using the graphical interface. Once they are downloaded for your specific export target, you are now able to automatically build the tool by using the `--export-godot` build script argument and specifying the path to the Godot engine executable you downloaded.

**Quick Steps From Godot Tutorial**
1. [Download Godot](https://godotengine.org/download/archive/)
2. Build Dynamatic with `--visual-dataflow` flag
3. Run Godot (from the directory to which it was downloaded)
4. Click `Editor` in the top navigation bar and select `Manage Export Templates`
5. Click `Online` button, download and Install Export Templates
6. Click `Project` button at top left of editor and select Export
7. Click the `Export PCK/ZIP...` enter a name for your export and validate it

For more details, visit [official godot engine website](https://docs.godotengine.org/en/stable/tutorials/export/exporting_projects.html#export-menu).

Finally, run the command below to export the Godot project as an executable binary that will be accessed by Dynamatic
```
# Export the Godot project as an executable binary
# Here it is a good idea to also provide the --visual-dataflow flag to ensure
# that the C++ library needed by the dataflow visualizer is up-to-date 
./build.sh --visual-dataflow --export-godot /path/to/godot-engine
```
The tool's binary is generated at `visual-dataflow/bin/visual-dataflow` and sym-linked at `bin/visual-dataflow` for convenience. 
Now, you can visualize the dataflow graphs for your compiled programs with Godot. See [how to use Dynamatic](../GettingStarted/Tutorials/Introduction/UsingDynamatic.md) for more details.
> [!NOTE]  
> Whenever you make a modification to the C++ library or to the Godot project itself, you can simply re-run the above command to recompile everything and re-generate the executable binary for the tool.

## 5. Enabling the XLS Integration
The experimental integration with the XLS HLS tool (see [here](../DeveloperGuide/Xls/XlsIntegration.md) for more information) can be enabled by providing the `--experimental-enable-xls` flag to build.sh.
> [!NOTE]
> `--experimental-enable-xls`, just like any other cmake-related flags, will only be applied if `./build.sh` configures CMake, which it, by default, will not do if a build folder (with a `CMakeCache.txt`) exists. To enable xls if you already have a local build, you can either force a reconfigure of all projects by providing the `--force` flag, or delete the Dynamatic's `CMakeCache.txt` to only force a reconfigure (and costly rebuild) of Dynamatic:
>
```sh
./build.sh --force --experimental-enable-xls
# OR
rm build/CMakeCache.txt
./build.sh --experimental-enable-xls
```
Once enabled, you do not need to provide `./build.sh` with `--experimental-enable-xls` to re-build.

## 6. Modelsim/Questa Installation
Dynamatic uses [Modelsim](hhttps://www.intel.com/content/www/us/en/software-kit/750666/modelsim-intel-fpgas-standard-edition-software-version-20-1-1.html) (has 32 bit dependencies) or [Questa](https://www.intel.com/content/www/us/en/software-kit/849791/questa-intel-fpgas-standard-edition-software-version-24-1.html) (64 bit simulator) to run simulations, thus you need to install it before hand. [Download](https://www.intel.com/content/www/us/en/software-kit/750666/modelsim-intel-fpgas-standard-edition-software-version-20-1-1.html) Modelsim or Questa, install it (in a directory with no special access permissions) and add it to path for Dynamatic to be able to run it. Add the following lines to the `.bashrc` file in your home directory to add modelsim to path variables.  
> [!NOTE]
> Ensure you write the full path
```sh
export MODELSIM_HOME=/path/to/modelsim  # path will look like /home/username/intelFPGA/20.1/modelsim_ase
export PATH="$MODELSIM_HOME/bin:$PATH"  # (adjust the path accordingly)
```
or
```sh
export MODELSIM_HOME=/path/to/questa    # path will look like home/username/altera/24.1std/questa_fse/
export PATH="$MODELSIM_HOME/bin:$PATH"
```

In any terminal, `source` .bashrc file and run the `vsim` command to verify that modelsim was added to path properly and runs.
```sh
source ~/.bashrc
vsim
```
If you encounter any issue related to `libXext` (if you installed Modelsim) you may need to install a few more libraries to enable the 32 bit architecture which supports packages needed by Modelsim:
```sh
sudo dpkg -add-architecture i386
sudo apt update
sudo apt install libxext6:i386 libxft2:i386 libxrender1:i386
```

If you are using Questa, running `vsim` will give you an error relating to the absence of a license.
To obtain a license (free or paid):
- Create an account on Intel's [Self Servicing License Center](https://www.intel.com/content/www/us/en/docs/programmable/683472/22-1/and-software-license.html) page. The page has detailed instructions on how to obtain a license.
- Request for a license. You will receive an authorization email with instructions on setting up a fixed or floating license (a fixed license suffices). This could take some minutes or up to a few hours.
- Download the license file and add it to path as shown below
```sh
#Questa license set up
export LM_LICENSE_FILE=/path/to/license/file     # looks like this "home/username/.../LR-240645_License.dat:$LM_LICENSE_FILE"
export MGLS_LICENSE_FILE=/path/to/license/file   # looks like this "/home/beta-tester/Downloads/LR-240645_License.dat"
export SALT_LICENSE_SERVER=/path/to/license/file # looks like this "/home/beta-tester/Downloads/LR-240645_License.dat"
```
> [!NOTE]  
> You may need only one of the three lines above based on the version of Questa you are using. Refer to the release notes for the version you have installed. Having the three lines poses no issue nonetheless.
