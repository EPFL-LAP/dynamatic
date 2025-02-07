# Dynamatic integration tests

## Introduction

Integration tests are used to ensure proper functioning of all parts of Dynamatic. Unlike unit tests, which check the correctness of a single piece of the final software product (e.g. a single function/method), the purpose of integration tests is to make sure that Dynamatic as a whole yields correct outputs. In other words, that a C source file is compiled to a valid HDL description of a circuit that functions correctly (according to the original source file).

Hence, each test case in Dynamatic's integration tests consists of:
- a C header file with a declaration of the kernel function to be compiled
- a C source file with the definition of said kernel function and a `main` entry point that calls the kernel using the `CALL_KERNEL` macro, defined in `include/dynamatic/Integration.h`

Integration tests are performed in the following manner:
1. The C source is compiled using Dynamatic
2. The result is written to a HDL file
3. The HDL description is simulated using ModelSim
4. The test passes if the outputs/memory states match what would be obtained by regular execution of the C source code

This process is represented by the following Dynamatic `.dyn` script:
```
set-src integration-test/test/test.c
compile
write-hdl
simulate
```

Considering the fact that ideally, it should be possible to run the tests with various compiler flags (e.g. use a specific buffering algorithm) and to run a variety of tests instead of just one, a Python script is used to automate this process of running Dynamatic commands.

## Test folder structure

Integration tests are located in the `integration-test` folder within Dynamatic's root directory. By default, each test case is located in its own subfolder, for example:
```
📂 integration-test
├── 📂 kernel_3mm_float
│   ├── 📄 kernel_3mm_float.h
│   ├── 📄 kernel_3mm_float.c
├── 📂 gcd
│   ├── 📄 gcd.h
│   ├── 📄 gcd.c
...
```

However, it is also possible to place test cases within another subfolder, for example:
```
📂 integration-test
├── 📂 memory
│   ├── 📂 test_memory_1
│   │   ├── 📄 test_memory_1.h
│   │   ├── 📄 test_memory_1.c
│   ├── 📂 test_memory_2
│   │   ├── 📄 test_memory_2.h
│   │   ├── 📄 test_memory_2.c
│   ├── 📂 test_memory_3
│   │   ├── 📄 test_memory_3.h
│   │   ├── 📄 test_memory_3.c
...
```
This is useful if there is a set of tests that are similar in some way and should hence be grouped together. Note that there should never exist two tests whose `.c` files have the same name, due to the way the integration script works (see below).

Please adhere to one of the schemes above when adding new integration tests in order to keep the folder structure clean for everyone. 

## Python integration script

`run_integration.py` is a Python script whose purpose is to allow the integration tests to be run in a simple and user-friendly way. The command syntax for calling it is as follows:
```
python run_integration.py [-h] [-l LIST] [-i IGNORE] [-t TIMEOUT]
```

- `-h/--help` - Displays the usage message.
- `-l/--list` - Give a path to a list of tests to be run. The tests should be listed only by their name (*see below how tests are identified*), each in its own line. If not given, all tests within the `integration-test` directory are run.
- `-i/--ignore` - Give a path to a list of tests to be skipped. They should be listed in the same way as in the case of `-l`. Note that this has priority over the `-l` list, i.e. if a test is listed both in this list and in the list given by `-l`, it will be skipped.
- `-t/--timeout` - Set a timeout (in seconds) for the duration of each integration test. If not given, a default of 500 seconds is used.

The script identifies tests by their `.c` file and considers each test's name to be the name of the `.c` file without the extension (i.e. `gcd` if the source is named `gcd.c`). It will find all tests in the `integration-test` folder by searching for `.c` source files and then filter them according to the files given by `-l` or `-i`. For example, running 
```sh
python run_integration.py --ignore ignore.txt --list list.txt
``` 
where `ignore.txt` contains
```
gcd
float_basic
```
and `list.txt` contains
```
float_basic
kernel_3mm_float
binary_search
gcd
```
the script will run tests whose source files are named `kernel_3mm_float.c` and `binary_search.c`, regardless of where they are located in the `integration-test` directory.

The file `ignored_tests.txt` in the main branch has a list of tests that are known to fail for various reasons, so it is recommended to use it when testing to avoid unnecessary running. Please note that the integration tests take quite a long time to run, so it is also recommended to use a custom list of tests to run to speed up testing when debugging.

**Note:** It is possible to modify the `.dyn` file that is run by modifying the `SCRIPT_CONTENT` variable in `run_integration.py`. This is useful if you require, for example, running the integration tests with specific Dynamatic flags. This is just temporary, as in the future, a more robust way of running tests with various options will be added.