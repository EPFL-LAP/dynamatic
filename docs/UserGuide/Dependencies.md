[Documentation Table of Contents](../README.md)

# Dependencies

Dynamatic uses [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) to manage its software dependencies (all hosted on GitHub). We depend on [Polygeist](https://github.com/llvm/Polygeist), a C/C++ frontend for MLIR which itself depends on [LLVM/MLIR](https://github.com/llvm/llvm-project) through a git submodule. The project is set up so that you can include LLVM/MLIR headers directly from Dynamatic code without having to specify their path through Polygeist. We also depend on [godot-cpp](https://github.com/godotengine/godot-cpp), the official C++ bindings for the Godot game engine which we use as the frontend to our interactive dataflow circuit visualizer. Finally, we inherit two MLIR dialects (*Handshake* and *HW*) from the [CIRCT](https://github.com/EPFL-LAP/circt) project (details [below](#circt)).

### Polygeist

[Polygeist](https://github.com/llvm/Polygeist) is a C/C++ frontend for MLIR including polyhedral optimizations and parallel optimizations features. Polygeist is thus responsible for the first step of our compilation process, that is taking source code written in C/C++ into the MLIR ecosystem. In particular, we care that our entry point to MLIR is at a very high semantic level, namely, at a level where polyhedral analysis is possible. The latter allows us to easily identify dependencies between memory accesses in source programs in a very accurate manner, which is key to optimizing the allocation of memory interfaces and resources in our elastic circuits down the line. Polygeist is able to emit MLIR code in the [*Affine*](https://mlir.llvm.org/docs/Dialects/Affine/) dialect, which is perfectly suited for this kind of analysis. 

### Working With Submodules

Having a project with submodules means that you have to pay attention to a couple additional things when pulling/pushing code to the project to maintain it in sync with the submodules. If you are unfamiliar with submodules, you can learn more about how to work with them [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Below is a very short and incomplete description of how our submodules are managed by our repository as well as a few pointers on how to perform simple git-related tasks in this context.

Along the history of Dynamatic's (in this context, called the *superproject*) directory structure and file contents, the repository stores the commit hash of a specific commit for each submodule's repository to identify the version of each *subproject* that the superproject currently depends on. These commit hashes are added and commited the same way as any other modification to the repository, and can thus evolve as development moves forward, allowing us to use more recent version of our submodules as they are pushed to their respective repositories. Here are a few concrete things you need to keep in mind while using the repository that may differ from your usual submodule-free workflow. 
- Clone the repository with `git clone --recurse-submodules git@github.com:EPFL-LAP/dynamatic.git` to instruct git to also pull and check out the version of the submodules referenced in the latest commit of Dynamatic's `main` branch.
- When pulling the latest commit(s), use `git pull --recurse-submodules` from the top level repository to also update the checked out commit from submodules in case the superproject changed the subprojects commits it is tracking.
- To commit changes made to files within Polygeist from the superproject (which is possible thanks to the fact that we use a fork of Polygeist), you first need to commit these changes to the Polygeist fork, and then update the Polygeist commit tracked by the superproject. More precisely,
  1. `cd` to the `polygeist` subdirectory,
  2. `git add` your changes and `git commit` them to the Polygeist fork,
  3. `cd` back to the top level directory,
  4. `git add polygeist` to tell the superproject to track your new Polygeist commit and `git commit` to Dynamatic.
  
  If you want to push these changes to remote, note that you will need to `git push` **twice**, once from the `polygeist` subdirectory (the Polygeist commit) and once from the top level directory (the Dynamatic commit). 