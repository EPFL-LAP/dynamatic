# Integration Tests

This document describes the setup used for running integration tests in Dynamatic, based on [GoogleTest](https://google.github.io/googletest/), [Ninja](https://ninja-build.org/)
and [CMake's ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html).

## Introduction

In order to avoid confusion, we introduce the following terminology.

A **benchmark** is a piece of .c code, commonly called a *kernel*, which is written in order to be compiled by Dynamatic into a HDL representation of a dataflow circuit. Confusingly, benchmarks are located in the `integration-test` folder. We also say that benchmarks are *testing resources*, since they are files that are loaded and used when running a test.

An **integration test** is a piece of code which runs a Dynamatic in order to compile, convert into HDL and simulate a certain benchmark, with some specific parameters. To clarify the difference, `integration-test/fir/fir.c` is a benchmark, while a C++ function which runs Dynamatic `set-src <benchmark>`, `compile`, `write-hdl` and `simulate` commands is an integration test. Note that integration tests differ by parameters used; a test that runs `compile --buffer-algorithm on-merges` is not the same as a test that runs `compile --buffer-algorithm fpga20`, even if they use the same benchmark.

As mentioned above, integration tests usually run the following steps using Dynamatic's frontend shell:
```
> set-src path/to/benchmark
> compile
> write-hdl
> simulate
```
If any of the steps report a failure, the test fails.

All code related to integration testing is located in `tools/integration`.
- `util.cpp` and `util.h` contain helper functions for running integration tests. 
- `TEST_SUITE.cpp` contains the actual tests. For more details, [see below](#googletest-basics).
- `generate_perf_report.py` is a script which reads the results of testing and generates a performance (in terms of cycle count) report in .md format.
- `run_integration.py`, `run_spec_integration.py` and `ignored_tests.txt` are leftovers from the old testing setup which didn't rely on GoogleTest. Even though they are unused, they are left here as reference if any problems arise with the current setup.

## GoogleTest basics

This is a crash course on GoogleTest features that are important for Dynamatic. For more information, see the [GoogleTest documentation](https://google.github.io/googletest/). 

In theory, the simplest way we could write integration tests for Dynamatic would be to make a test case for each combination of parameters (such as `--buffer-algorithm` and other flags) and benchmarks. Note that the code in the examples is simplified with some details removed for the sake of clarity.
```c++
TEST(BasicTests, binary_search_vhdl) {
  EXPECT_EQ(runIntegrationTest("binary_search", "vhdl"));
}

TEST(BasicTests, binary_search_verilog) {
  EXPECT_EQ(runIntegrationTest("binary_search", "verilog"));
}

TEST(BasicTests, fir_vhdl) {
  EXPECT_EQ(runIntegrationTest("fir", "vhdl"));
}

TEST(BasicTests, fir_verilog) {
  EXPECT_EQ(runIntegrationTest("fir", "verilog"));
}

...
```
It is immediately clear that such an approach is highly impractical. For this reason, we take advantage of GoogleTest's parameterized testing feature. This allows us to run the same integration test code for many different parameters, in our case benchmarks. For example:
```c++
TEST_P(BasicTests, vhdl) {
  std::string name = GetParam();
  EXPECT_EQ(runIntegrationTest(name, "vhdl"));
}

TEST_P(BasicTests, verilog) {
  std::string name = GetParam();
  EXPECT_EQ(runIntegrationTest(name, "verilog"));
}

INSTANTIATE_TEST_SUITE_P(
    AllBenchmarks, BasicTests,
    testing::Values("single_loop", "atax", "atax_float", /* ... */)
);
```
In this example, `BasicTests` is the name of something called a *test fixture*, which is a group of tests which should be run in the same way. This fixture is parameterized, with the parameter being the benchmark name, and has two test cases, `vhdl` and `verilog`. Finally, we instantiate the test suite. The instantiation's parameters are `single_loop`, `atax`, `atax_float` etc. (i.e. all Dynamatic benchmarks), and the instantiation is appropriately named `AllBenchmarks`. By instantiating, we achieved the same as in the first example, but without having to write it all manually.

GoogleTest uses the following naming scheme for parameterized tests:
```
InstantiationName/FixtureName.TestCaseName/ParamValue
```
For example:
```
AllBenchmarks/BasicTests.vhdl/binary_search
AllBenchmarks/BasicTests.verilog/fir
```

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

**If you want to add a benchmark**, add the corresponding folder and .c kernel inside of it, and then add its name to the test suite instantiation. For example, if we want to add `new_kernel` to miscellaneous benchmarks, we would:
1. Create the folder `integration-test/new_kernel`.
2. Write the code of the kernel to `integration-test/new_kernel/new_kernel.c`.
3. In `tools/integration/TEST_SUITE.cpp`, find the instantiation `MiscBenchmarks` and inside `testing::Values` add the name `"new_kernel"`.

**If you want to add a new test case for an existing fixture**, add a new macro invocation:
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

**If we want to add a new fixture** (for a new group of benchmarks), first we define the fixture at the top of `TEST_SUITE.cpp`:
```c++
class NewFixture : public testing::TestWithParam<std::string> {};
```
Then we create some test cases for it as stated earlier. Finally, we instantiate it:
```c++
INSTANTIATE_TEST_SUITE_P(SomeBenchmarks, NewFixture, testing::Values(...));
```

**If you want a test to be ignored by the CI** for whatever reason, you should name it `testCaseName_NoCI`.

## Custom integration tests

Sometimes, there is a need to test Dynamatic in a way that the frontend shell (`bin/dynamatic`) doesn't support out-of-the-box. One of these examples is the speculation feature. In this case, you will need to implement everything that `bin/dynamatic` does, but yourself (with your modifications of course). You can see an example of this in the function `runSpecIntegrationTest` in `util.cpp`.