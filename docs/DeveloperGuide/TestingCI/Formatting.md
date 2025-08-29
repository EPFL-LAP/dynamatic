# Formatting Checker

This document describes the code style and format checks implemented as part of the CI workflow. These checks are based on [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and the [git-clang-format script](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-format/git-clang-format) for C/C++ and [autopep8](https://github.com/hhatto/autopep8) for Python. 

Note that both clang-format and autopep8 checks are incremental, in the sense that they only check files that have been modified in your branch compared to main. This is because reformatting the entire codebase is impractical; this way we can ensure that the code style will eventually be consistent and correct without causing too many headaches.

## clang-format

Version differences between formatters can cause problems. The CI workflow uses clang-format 20, and it is recommended that you use the same version locally. You can also always see `.github/workflows/ci.yml` and double check that you have the correct version that is used there.

The style used by clang-format is determined by the `.clang-format` file in the project root directory.

To try it yourself, use:
```
$ clang-format --dry-run filename.cpp
```
The option `--dry-run` tells clang-format not to modify the file(s), but just to print the style modifications that it considers necessary. If you want it to modify the file(s) in place, use `-i` instead.

Listing modified files by hand when running clang-format can be a tedious job. git-clang-format makes this job much easier. Just use:
```
$ git clang-format --diff main
```
This will print the formatting changes that should be done to all files that are modified compared to main. If you want these changes to be immediately applied (which usually is the case, because you want to avoid a CI failure), use:
```
$ git clang-format main
```
This will not modify any files that have unstaged changes. If you want that as well, add the option `--force`.

**It is highly recommended** that you run clang-format yourself prior to opening a PR/pushing to a PR in order to avoid unnecessary CI failures. You can also use various VS Code extensions that format your code automatically when you save it.

## autopep8

As with clang-format, version differences can be problematic. The CI currently uses autopep8 2.3.2, but if you are unsure, again you should check the version in `.github/workflows/ci.yml`. 

autopep8 is quite similar to clang-format. Use
```
$ autopep8 --diff file.py
```
to see the changes that would be applied, or 
```
$ autopep8 --in-place file.py
```
to automatically apply the changes to the file in place.

Note that due to the fact that a lot of Python files in this project contain multiline strings with VHDL/Verilog, we use the flag `--max-line-length 200`, to avoid modifying them, which would make the HDL code harder to read.

Unfortunately, there is no equivalent to git-clang-format for autopep8, so to format files that were modified compared to main, you have to use:
```
$ autopep8 --max-line-length 200 --diff $(git ls-files '*.py' --modified)
```
Of course, you can replace `--diff` with `--in-place` according to what you need. Unfortunately, the above command will fail if no files have been modified, or if some of the modified files are deleted files. Hence, the entire code of the check in `ci.yml` is:
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

The simplest way to keep your Python scripts well-formatted is to use VS Code formatters that run when you save files. Again, **it is highly recommended** to format properly and avoid unnecessary CI failures.