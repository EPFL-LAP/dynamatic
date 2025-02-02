# Dynamatic integration tests

## Introduction

Integration tests are used to ensure proper functioning of all parts of Dynamatic. Unlike unit tests, which check the correctness of a single piece of the final software product (e.g. a single function/method), the purpose of integration tests is to make sure that Dynamatic as a whole yields correct outputs. In other words, that a C source file is compiled to a valid HDL description of a circuit that functions correctly (according to the original source file).

Hence, each test case in Dynamatic's integration tests consists of:
- a C header file with a declaration of the kernel function to be compiled
- a C source file with the definition of said kernel function and a `main` entry point that calls the kernel using the `CALL_KERNEL` macro, defined in `dynamatic/Integration.h`

Integration tests are performed in the following manner:
1. The C source is compiled using Dynamatic
2. The result is written to a HDL file
3. The HDL description is simulated using ModelSim
4. The test passes iff the simulation ends without errors

This process is represented by the following Dynamatic `.dyn` script:
```
set-src integration-test/test/test.c
compile
write-hdl
simulate
```

Considering the fact that ideally, it should be possible to run the tests with various compiler flags (e.g. use a specific buffering algorithm) and to run a variety of tests instead of just one, a Python script is used to automate this process of running Dynamatic commands.

## Python integration script

`run_integration.py` is a Python script whose purpose is to allow the integration tests to be run in a simple and user-friendly way. The command syntax for calling it is as follows:
```
python run_integration.py [-h] [-l LIST] [-i IGNORE] [-t TIMEOUT]
```

- `-h/--help` - Displays the usage message.
- `-l/--list` - Give a path to a list of tests to be run. The tests should be listed only by their name (i.e. `gcd`, and not `integration-test/gcd/gcd.c`) and each in its own line. If not given, all tests within the `integration-test` directory are run.
- `-i/--ignore` - Give a path to a list of tests to be skipped. They should be listed in the same way as described above in the case of `-l`. Note that this has priority over the `-l` list, i.e. if a test is listed both in this list and in the list given by `-l`, it will be skipped.
- `-t/--timeout` - Set a timeout (in seconds) for the duration of each integration test. If not given, a default of 500 seconds is used.
- **Note: This script must be run from the folder it is located in.**

The file `ignored_tests.txt` in the main branch has a list of tests that are known to fail for various reasons, so it is recommended to use it when testing to avoid unnecessary running. Please note that the integration tests take quite a long time to run, so it is also recommended to use a custom list of tests to run to speed up testing when debugging.

## GitHub Actions workflow

In order to verify the compliance of contributions, an Actions workflow is set up for the purpose of integration testing. The workflow runs all integration tests (except the ones in `ignored_tests.txt`) when a pull request to the main branch is made. This does not apply to PRs marked as drafts, so if your PR is a draft, make sure to mark it as such to avoid running the workflow unnecessarily. Also, external contributors must receive approval to run the workflow on their PR due to security concerns. Please note that having the workflow successfully complete (i.e. pass all integration tests) is a necessary condition for having your PR approved.