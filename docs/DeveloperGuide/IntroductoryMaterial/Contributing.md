# Contributing 

Dynamatic welcomes contributions from the open-source community and from students as part of academic projects. We generally follow the LLVM and MLIR community practices, and currently use [GitHub issues and pull requests](#github-issues--pull-requests) to handle bug reports/design proposals and code contributions, respectively. Here are some high-level guidelines (inspired by CIRCT's guidelines):
- Please use `clang-format` in the LLVM style to format the code (see [`.clang-format`](https://github.com/EPFL-LAP/dynamatic/tree/main/tutorials/Introduction)). There are good plugins for common editors like VSCode ([cpptool](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools&ssr=false) or [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)) that can be set up to format each file on save, or you can run them manually. This makes code easier to read and understand, and more uniform throughout the codebase.
- Please pay attention to warnings from `clang-tidy` (see [`.clang-tidy`](https://github.com/EPFL-LAP/dynamatic/blob/main/.clang-tidy)). Not all necessarily need to be acted upon, but in the majority of cases, they help in identifying code-smells. 
- Please follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html).
- Please practice [*incremental development*](https://llvm.org/docs/DeveloperPolicy.html#incremental-development), preferring to send a small series of incremental patches rather than large patches. There are other policies in the LLVM Developer Policy document that are worth skimming.
- Please create an issue if you run into a bug or problem with Dynamatic.
- Please create a PR to get a code review. For reviewers, it is good to look at the primary author of the code you are touching to make sure they are at least CC'd on the PR.

## Relevant Documentation

You may find the following documentation useful when contributing to Dynamatic:
- [Advanced Build Instructions](../../UserGuide/AdvancedBuild.md)
<!-- - [Testing](../../DeveloperGuide/Testing.md) -->
- [Development Tools](../DevelopmentTools.md)

## GitHub Issues & Pull requests

The project uses GitHub [issues](https://github.com/features/issues) and [pull requests (PRs)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) to handle contributions from the community. If you are unfamiliar with those, here are some guidelines on how to use them productively:
- Use meaningful titles and descriptions for issues and PRs you create. Titles should be short yet specific and descriptions should give a good sense of what you are bringing forward, be it a bug report or code contribution.  
- If you intend to contribute a large chunk of code to the project, it may be a good idea to first open a GitHub issue to describe the high-level design of your contribution there and leave it up for discussion. This can only increase the likelihood of your work eventually being merged, as the community will have had a chance to discuss the design before you propose your implementation in a PR (e.g., if the contribution is deemed to large, the community may advise to split it up in several incremental patches). This is especially advisable to first-time contributors to open-source projects and/or compiler development beginners.
- Use "Squash and Merge" in PRs when they are approved - we don't need the intra-change history in the repository history.

## Experimental Work

One of Dynamatic's priority is to keep the repository's `main` branch stable at all times, with a high code quality throughout the project. At the same time, as an academic project we also receive regular code contributions from students with widely different backgrounds and field expertises. These contributions are often part of research-oriented academic projects, and are thus very "experimental" in nature. They will generally result in code that doesn't quite match the standard of quality (less tested, reliable, interoperable) that we expect in the repository. Yet, we still want to keep track of these efforts on the `main` branch to make them visible to and usable by the community, and encourage future contributions to the more experimental parts of the codebase.

To achieve these dual and slightly conflicting goals, Dynamatic supports *experimental* contributions to the repository. These will still have to go through a PR but will be merged more easily (i.e., with *slightly* less regards to code quality) compared to *non-experimental* contributions. We offer this possibility as a way to push for the integration of research work inside the project, with the ultimate goal of having these contributions graduate to full *non-experimental* work. Obviously, we strongly encourage developers to make their submitted code contributions as clean and reliable as possible regardless of whether they are classified as experimental. It can only increase their chance of acceptance.

To clearly separate them from the rest, all **experimental** contributions should exist within the `experimental` directory which is located at the top level of the repository. The latter's internal structure is identical to the one at the top level with an `include` folder for all headers, a `lib` folder for pass implementations, etc. All public code entities defined within experimental work should live under the `dynamatic::experimental` C++ namespace for clear separation with *non-experimental* publicly defined entities. 
