# Integration Tests

This document describes the setup used for running integration tests in Dynamatic, based on [GoogleTest](https://google.github.io/googletest/), [Ninja](https://ninja-build.org/)
and [CMake's ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html). For a more general overview of the purpose of integration testing, see [Introduction](Introduction.md).

## Introduction

In order to avoid confusion, we introduce the following terminology.

A **benchmark** is a piece of .c code, commonly called a *kernel*, which is written in order to be compiled by Dynamatic into a HDL representation of a dataflow circuit. Confusingly, benchmarks are located in the `integration-test` folder. We also say that benchmarks are *testing resources*, since they are files that are loaded and used when running a test.

An **integration test** is a piece of code which runs a Dynamatic in order to compile, convert into HDL and simulate a certain benchmark, with some specific parameters. To clarify the difference, `integration-test/fir/fir.c` is a benchmark, while a C++ function which runs Dynamatic `set-src <benchmark>`, `compile`, `write-hdl` and `simulate` commands is an integration test. Note that integration tests differ by parameters used; a test that runs `compile --buffer-algorithm on-merges` is not the same as a test that runs `compile --buffer-algorithm fpga20`, even if they use the same benchmark.

As mentioned above, the basic integration tests run the following steps using Dynamatic's frontend shell (this is referred to as the "basic" flow):
```
> set-dynamatic-path path/to/dynamatic/root
> set-src path/to/benchmark
> set-clock-period 5
> compile --buffer-algorithm fpga20
> write-hdl --hdl vhdl
> simulate
```
If any of the steps report a failure, the test fails. For more information about the commands and the Dynamatic frontend shell, see [Command Reference](../../UserGuide/CommandReference.md).

Dynamatic uses GoogleTest as the testing framework for writing integration tests. When Dynamatic is built, CMake registers all of the GoogleTest tests, which allows them to be run using CMake's ctest utility:
```
$ cd build/tools/integration
$ ctest --parallel 8 --output-on-failure 
```
To simplify this, a custom target is defined in `tools/integration/CMakeLists.txt`, so that the tests can easily be run using ninja:
```
$ ninja -C build run-all-integration-tests
```

## Code structure

All code related to integration testing is located in `tools/integration`.
- `util.cpp` and `util.h` contain helper functions for running integration tests. 
- `TEST_SUITE.cpp` contains the actual tests. For more details, [see below](#googletest-basics).
- `generate_perf_report.py` is a script which reads the results of testing and generates a performance (in terms of cycle count) report in .md format.
- `run_integration.py`, `run_spec_integration.py` and `ignored_tests.txt` are leftovers from the old testing setup which didn't rely on GoogleTest. Even though they are unused, they are left here as reference if any problems arise with the current setup. `run_integration.py` was used to run all benchmarks with the basic flow, except the ones listed in `ignored_tests.txt`. `run_spec_integration.py` was used to run benchmarks with a custom flow which used the speculation feature. It was replaced with the new setup because GoogleTest is a standard framework that people might be familiar with already, is very well documented and writing different tests with it is simpler than extending a custom script.

## GoogleTest basics

This is a crash course on GoogleTest features that are important for Dynamatic. Also, refer to the [GoogleTest documentation](https://google.github.io/googletest/), since these concepts can be confusing. 

GoogleTest allows us to write tests in a function-like form. For this, we use the `TEST` macro. Here is a very elementary example of a test:
```c++
int sum(int a, int b) {
  return a + b;
}

TEST(SumTests, positive_numbers) {
  int x = 5, y = 4;
  int expected = x + y;
  int received = sum(x, y);

  EXPECT_EQ(received, expected);
}
```
The code inside the test is run by GoogleTest like a function. `EXPECT_EQ` is a macro assertion which ensures that the two arguments are equal. If not, the test will fail.

Now, let's demonstrate how this would look like in the case of Dynamatic. Assume there is a helper function
```c++
int runIntegrationTest(std::string benchmark);
```
which runs the basic flow described in the introduction and returns 0 if it was successful and 1 otherwise. Then, a test would look like:
```c++
TEST(BasicTests, binary_search) {
  int exitCode = runIntegrationTest("binary_search");
  
  EXPECT_EQ(exitCode, 0);
}
```
Because of the macro, the test will fail if the exit code is non-zero.

However, by doing this we only run one benchmark with the basic flow. If we want to run multiple benchmarks, we would have something like this:
```c++
TEST(BasicTests, binary_search) {
  int exitCode = runIntegrationTest("binary_search");
  
  EXPECT_EQ(exitCode, 0);
}

TEST(BasicTests, fir) {
  int exitCode = runIntegrationTest("fir");
  
  EXPECT_EQ(exitCode, 0);
}

TEST(BasicTests, kernel_3mm_float) {
  int exitCode = runIntegrationTest("kernel_3mm_float");
  
  EXPECT_EQ(exitCode, 0);
}

// etc.
```
and so on for all Dynamatic benchmarks. It is immediately clear that such an approach is very impractical, since there are a lot of benchmarks, which means a lot of duplicate code. This is where GoogleTest's parameterized testing helps us. A parameterized test is a test which accepts a parameter, much like a function or a template in programming. First, we define a testing fixture for the parameterized test. This is necessary to define the type of the parameter.
```c++
class BasicTests : public testing::TestWithParam<std::string> {};
```

Then, we create a parameterized test with the given fixture. As specified in the fixture, the test takes a parameter of type `std::string`. The parameter's value can be retrieved using `GetParam()`.
```c++
TEST_P(BasicTests, basic) {
  std::string name = GetParam();
  int exitCode = runIntegrationTest(name);

  EXPECT_EQ(exitCode, 0);
}
```

Finally, we must specify the concrete parameters that will be used to run the parameterized test. We use the macro `INSTANTIATE_TEST_SUITE_P(instantiationName, fixtureName, params)` for this purpose. Since the parameter list contains all benchmarks, we appropriately name the instantiation `AllBenchmarks`.
```c++
INSTANTIATE_TEST_SUITE_P(
    AllBenchmarks, BasicTests,
    testing::Values("binary_search", "fir", "kernel_3mm_float", /* ... */)
);
```

This way, we have used only a few lines of code to achieve the same outcome as if we were to duplicate the code as in the prior example.

Note that GoogleTest uses the following naming scheme for parameterized tests:
```
InstantiationName/FixtureName.TestCaseName/ParamValue
```
For example:
```
AllBenchmarks/BasicTests.vhdl/binary_search
AllBenchmarks/BasicTests.verilog/fir
```
This will be important in the following section.

## Running tests

GoogleTest is compatible with CMake's ctest utility for running tests, and so everything is set up in the corresponding CMakeLists.txt such that CMake registers all GoogleTest tests.

After building Dynamatic, the test binaries are located in `build/tools/integration`. After `cd`-ing to that path, we can use ctest to run all tests:
```
$ ctest
Test project /home/user/dynamatic/build/tools/integration
      Start  1: MiscBenchmarks/BasicFixture.basic/single_loop  # GetParam() = "single_loop"
...
```
The tests will be run one by one and we can see the outcome of each test. 

An obvious problem is that this is slow; only one test is being run at a time. To solve this, use the option `--parallel <num-workers>` to run several tests in parallel. For example, `ctest --parallel 8` will run 8 tests at a time. Note that the output will be a bit messier though. 

It is advisable to explicitly set the timeout for each test. This can be achieved using the option `--timeout <seconds>`. For tests using the naive on-merges buffer algorithm, 500 seconds is more than enough, but when using the FPGA'20 algorithm, it is recommended to use a larger number such as 1500 seconds.

We would also like to see the output in case something goes wrong, because ctest doesn't do that by default. To do this, add the `--output-on-failure` flag.

Hence, running all integration tests would look like:
```
$ cd build/tools/integration
$ ctest --parallel 8 --timeout 500 --output-on-failure
```
Typing this is a bit annoying, isn't it? To mitigate this, there is a custom CMake target added in CMakeLists.txt:
```cmake
add_custom_target(
  run-all-integration-tests
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel 8 --timeout 500 --output-on-failure
  DEPENDS dynamatic-opt integration
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tools/integration
  COMMENT "Run integration tests."
  VERBATIM
  USES_TERMINAL
)
```
Now, we can (from the Dynamatic root directory) simply type:
```
$ ninja -C build run-all-integration-tests
```

What if we don't want to run all tests? This is where the naming scheme mentioned in the previous section comes into play. We can use `-R <regex>` to only run tests whose names match the given regular expression. For example:
```
$ ctest -R vhdl
```
would only run tests whose names contain the string `vhdl`. We can also exclude tests using `-E <regex>`. This command
```
$ ctest -E NoCI
```
would run all test *except* the ones whose names contain the string `NoCI`. In fact, this is used by the target `run-ci-integration-tests`, which will skip any tests that are marked with `NoCI`.

## A basic test case explained

Let's take the `BasicFixture` and `basic` test case in order to explain how a typical test looks like:
```c++
TEST_P(BasicFixture, basic) {
  std::string name = GetParam();
  int simTime = -1;

  EXPECT_EQ(runIntegrationTest(name, simTime), 0);

  RecordProperty("cycles", std::to_string(simTime));
}
```

First, we call `GetParam` to retrieve the test's parameter, in this case the name of the benchmark.

`runIntegrationTest` is a helper function defined in `util.cpp` that abstracts away the details of calling Dynamatic. It returns an exit code (0 if success) by value, and the simulation time (in cycles) by reference. 

`EXPECT_EQ` is an assertion that fails the test if the first argument isn't equal to the second one.

The simulation time is then recorded using GoogleTest's `RecordProperty`, which can log arbitrary key-value properties and save them in .xml files under `build/tools/integration/results`. This is important because these values are used by `generate_perf_report.py` to generate the performance report.

This test is parameterized, so when running, many invocations with various parameters will be made:
```
MiscBenchmarks/BasicFixture.basic/atax_float
MiscBenchmarks/BasicFixture.basic/binary_search
MiscBenchmarks/BasicFixture.basic/fir
MiscBenchmarks/BasicFixture.basic/kernel_3mm_float

etc.
```

## Adding tests and benchmarks

Currently, there are 3 groups of benchmarks:
- Memory benchmarks, located in `integration-test/memory/`,
- Sharing benchmarks, located in `integration-test/sharing/`,
- Miscellaneous benchmarks, located in `integration-test/`.

Test suite instantiations follow these benchmark groups, since each instantiation defines which benchmarks the test suite will use.

### Adding a benchmark

*Example use case: You created some benchmarks that don't have array accesses to evaluate timing/area optimization, and you want them to be run with the basic Dynamatic flow.*

Add the corresponding folder and .c kernel inside of it, and then add its name to the test suite instantiation. For example, if we want to add `new_kernel` to miscellaneous benchmarks, we would:
1. Create the folder `integration-test/new_kernel`.
2. Write the code of the kernel to `integration-test/new_kernel/new_kernel.c`.
3. In `tools/integration/TEST_SUITE.cpp`, find the instantiation `MiscBenchmarks` and inside `testing::Values` add the name `"new_kernel"`.

### Adding a new test case for an existing fixture

*Example use case: You want the same set of benchmarks to be run with some different parameters, e.g. with Verilog instead of VHDL or with FPL'22 instead of FPGA'20 buffering.*

Add a new macro invocation:
```c++
TEST_P(ExistingFixtureName, newTestCaseName) {
  // code
}
```
Because of the fact that the fixture already exists (and is instantiated), the new test case will be run with all parameters that the instantiation contains. An example of this is if we would like to add tests using the Verilog backend to the existing `BasicFixture` (which is instantiated with miscellaneous benchmarks):
```c++
TEST_P(BasicFixture, verilog) {
  std::string name = GetParam();
  int simTime = -1;

  EXPECT_EQ(runIntegrationTest(name, simTime, std::nullopt, true), 0);

  RecordProperty("cycles", std::to_string(simTime));
}
```

### Adding a new fixture

*Example use case: You want to have a new group of benchmarks, different than the three mentioned at the beginning. Maybe you have a set of benchmarks only for speculation.*

First we define the fixture at the top of `TEST_SUITE.cpp`:
```c++
class NewFixture : public testing::TestWithParam<std::string> {};
```
Then we create some test cases for it as stated earlier. Finally, we instantiate it:
```c++
INSTANTIATE_TEST_SUITE_P(SomeBenchmarks, NewFixture, testing::Values(...));
```

### Ignoring tests in the Actions workflow

*Example use case: Your feature is merged into main along with some tests, but is incomplete and the tests take a long time, so you want the workflow to skip them.*

If you want the actions workflow to ignore a test case for whatever reason, you should name it `testCaseName_NoCI`. For example:
```c++
TEST_P(SpecFixture, spec_NoCI) {
  // ...
}
```

## Custom integration tests

Sometimes, there is a need to test Dynamatic in a way that the frontend shell (`bin/dynamatic`) doesn't support out-of-the-box. One of these examples is the speculation feature. In this case, you will need to implement everything that `bin/dynamatic` does, but yourself (with your modifications of course). You can see an example of this in the function `runSpecIntegrationTest` in `util.cpp`.