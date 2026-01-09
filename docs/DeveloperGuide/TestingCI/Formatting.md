# Formatting Checker

This document describes the code style and format checks implemented as part of the CI workflow. These checks are based on [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and the [git-clang-format script](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-format/git-clang-format) for C/C++ and [autopep8](https://github.com/hhatto/autopep8) for Python. 

## Introduction

It is important for any large code base that the style and formatting of the code is consistent and adheres to some standard. This makes the code readable and visually pleasing.

Checking code style manually is tedious and prone to mistakes; it is very hard to review hundreds of lines of code for formatting mistakes. For this reason, this task should be automated. Specifically, clang-format is the tool used for automatically checking C/C++ code formatting and autopep8 is used for checking Python code formatting.

Note that both clang-format and autopep8 checks used in the GitHub Actions workflow are incremental, in the sense that they only check files that have been modified in your branch compared to main. This is because reformatting the entire codebase is impractical; this way we can ensure that the code style will eventually be consistent and correct without causing too many headaches.

## clang-format

The CI workflow uses clang-format 20, and it is recommended that you use the same version locally. This is because version differences between formatters can cause issues (e.g. some version may prefer `ctor() {}`, while another may prefer `ctor(){}`) You can also always see `.github/workflows/ci.yml` and double check that you have the correct version that is used there.

The style used by clang-format is determined by the `.clang-format` file in the project root directory.

Some ways you can use clang-format yourself are listed below.

1. Just check formatting and list the style changes that would be applied without modifying the file:
    ```
    $ clang-format --dry-run filename.cpp
    ```

2. Fix formatting in the file (i.e. make clang-format modify it):
    ```
    $ clang-format -i filename.cpp
    ```

3. Run it on multiple files:
    ```
    $ clang-format --dry-run filename1.cpp filename2.cpp filename3.cpp
    ```

Listing modified files by hand when running clang-format can be a tedious job. git-clang-format makes this job much easier.

1. To check files that were modified compared to main:
    ```
    $ git clang-format --diff main
    ```
    This is the equivalent of running `clang-format --dry-run` on all modified files.

2. To immediately fix the formatting in those files (recommended because you want to avoid a workflow failure):
    ```
    $ git clang-format main
    ```

3. To modify files that have unstaged changes, add the option `--force`.

**It is highly recommended** that you run clang-format yourself prior to opening a PR/pushing to a PR in order to avoid unnecessary CI failures. You can also use various VS Code extensions that format your code automatically when you save it.

## autopep8

The CI currently uses autopep8 2.3.2, but if you are unsure, again you should check the version in `.github/workflows/ci.yml`. As with clang-format, this is because version differences can be problematic.

autopep8 is quite similar to clang-format. Below are some things to know about it.

1. To see changes that would be applied:
    ```
    $ autopep8 --diff file.py
    ```

2. To automatically apply the changes to the file in place:
    ```
    $ autopep8 --in-place file.py
    ```

3. It is recommended to use the flag `--max-line-length 200`. This is because a lot of Python files in this project contain multiline strings with VHDL/Verilog, which may be affected by autopep8 making line breaks, which in turn would make the HDL code harder to read.
4. Unfortunately, there is no equivalent to git-clang-format for autopep8, so to fix formatting in files that were modified compared to main, you have to use[^1]:
    ```
    $ autopep8 --max-line-length 200 --in-place $(git ls-files '*.py' --modified)
    ```

The simplest way to keep your Python scripts well-formatted is to use VS Code formatters that run when you save files. Again, **it is highly recommended** to format properly and avoid unnecessary CI failures.


[^1]: Note that the above command will fail if no files have been modified, or if some of the modified files are deleted files. Hence, the entire code of the check in `.github/workflows/ci.yml` is:
    ```bash
    FILES=$(comm -23 <(git ls-files '*.py' --modified | sort) <(git ls-files '*.py' --deleted | sort))
    STATUS=$?
    if [[ $FILES ]]; then
      exec 3>&1
      OUTPUT=$(autopep8 --max-line-length 200 --diff $FILES 3>&- | tee /dev/fd/3)
      exec 3>&-
      if [[ $OUTPUT ]]; then
        STATUS=1
      else
        STATUS=0
      fi
    else
      STATUS=0
    fi
    echo "autopep8 exited with $STATUS"
    ```