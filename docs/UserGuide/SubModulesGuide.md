## Working With Submodules

Having a project with submodules means that you have to pay attention to a couple additional things when pulling/pushing code to the project to maintain it in sync with the submodules. If you are unfamiliar with submodules, you can learn more about how to work with them [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Below is a very short and incomplete description of how our submodules are managed by our repository as well as a few pointers on how to perform simple git-related tasks in this context.

Along the history of Dynamatic's (in this context, called the *superproject*) directory structure and file contents, the repository stores the commit hash of a specific commit for each submodule's repository to identify the version of each *subproject* that the superproject currently depends on. These commit hashes are added and commited the same way as any other modification to the repository, and can thus evolve as development moves forward, allowing us to use more recent version of our submodules as they are pushed to their respective repositories. Here are a few concrete things you need to keep in mind while using the repository that may differ from the submodule-free workflow. 
- Clone the repository with `git clone --recurse-submodules git@github.com:EPFL-LAP/dynamatic.git` to instruct git to also pull and check out the version of the submodules referenced in the latest commit of Dynamatic's `main` branch.
- When pulling the latest commit(s), use `git pull --recurse-submodules` from the top level repository to also update the checked out commit from submodules in case the superproject changed the subprojects commits it is tracking.
- To commit changes made to files within Polygeist from the superproject (which is possible thanks to the fact that we use a fork of Polygeist), you first need to commit these changes to the Polygeist fork, and then update the Polygeist commit tracked by the superproject. More precisely,
  1. `cd` to the `polygeist` subdirectory,
  2. `git add` your changes and `git commit` them to the Polygeist fork,
  3. `cd` back to the top level directory,
  4. `git add polygeist` to tell the superproject to track your new Polygeist commit and `git commit` to Dynamatic.
  
  If you want to push these changes to remote, note that you will need to `git push` **twice**, once from the `polygeist` subdirectory (the Polygeist commit) and once from the top level directory (the Dynamatic commit). 