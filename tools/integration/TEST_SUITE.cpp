//===- TEST_SUITE.cpp - Basic integration test suite -----------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a basic set of parameterized integration tests
// that run Dynamatic without any special flags/settings.
//
//===----------------------------------------------------------------------===//
#include "util.h"

#include <gtest/gtest.h>

/// Base class for Dynamatic unit tests
/// provides utilities
class BaseFixture : public testing::TestWithParam<std::string> {
public:
  /// \brief: This is called to log the number of cycles in the console.
  void logPerformance(unsigned cycles) const {
    const std::string &benchmarkName(GetParam());
    auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string fixtureName(info->test_suite_name());
    std::cout << "[INFO] Benchmark " << fixtureName << "/" << benchmarkName
              << " latency: " << cycles << " cycles" << std::endl;
  }

protected:
  /// \brief: This is a callback function on the startup of the test.
  void SetUp() override {
    const std::string &benchmarkName(GetParam());
    auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string fixtureName(info->test_suite_name());
    std::cout << "[INFO] Running " << fixtureName << "/" << benchmarkName
              << std::endl;
  }
};

class BasicFixture : public BaseFixture {};
// Use CBC MILP solver to test a subset of MiscBenchmarks (CBC is slower than
// Gurobi)
class CBCSolverFixture : public BaseFixture {};
// Use FPL22 placement algorithm on a small subset of MiscBenchmarks
class FPL22Fixture : public BaseFixture {};
class MemoryFixture : public BaseFixture {};
class SharingFixture : public BaseFixture {};
class SharingUnitTestFixture : public BaseFixture {};
class SpecFixture : public BaseFixture {};

class RigidificationFixture : public BaseFixture {};
class FtdWithSimpleBuffersFixture : public BaseFixture {};

TEST_P(BasicFixture, basic) {
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = true,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

TEST_P(CBCSolverFixture, basic) {
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = true,
      .useSharing = false,
      .milpSolver = "cbc",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

TEST_P(FtdWithSimpleBuffersFixture, basic) {
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .milpSolver = "cbc",
      .bufferAlgorithm = "on-merges",
      .useFtd = true,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

#if 0
TEST_P(FPL22Fixture, basic) {
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpl22",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
}
#endif

//
// This is an example test case which uses the Verilog backend.
// It is currently disabled because a lot of benchmarks still
// don't work properly with Verilog, so running it would create
// a lot of errors, preventing the CI from running normally.
//
// TEST_P(BasicFixture, verilog) {
//   std::string name = GetParam();
//   int simTime = -1;

//   EXPECT_EQ(runIntegrationTest(name, simTime, std::nullopt, true), 0);

//   RecordProperty("cycles", std::to_string(simTime));
// }

TEST_P(MemoryFixture, basic) {
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" / "memory",
      .testVerilog = true,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

/// This testing fixture runs the test with and without sharing. It checks
/// whenever the sharing option is enabled, the pass can run without any
/// interruption and does not penalize the latency.
TEST_P(SharingUnitTestFixture, basic) {
  IntegrationTestData configWithSharing{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" / "sharing",
      .testVerilog = false,
      .useSharing = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(configWithSharing), 0);

  IntegrationTestData configWithoutSharing{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" / "sharing",
      .testVerilog = false,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(configWithoutSharing), 0);

  // Check if sharing brings under 5% latency increase
  EXPECT_EQ(configWithoutSharing.simTime * 1.05 > configWithSharing.simTime,
            true);

  RecordProperty("cycles", std::to_string(configWithSharing.simTime));
  logPerformance(configWithSharing.simTime);
}

/// This testing fixture runs the test with and without sharing. It checks
/// whenever the sharing option is enabled, the pass can run without any
/// interruption and does not penalize the latency.
TEST_P(SharingFixture, sharing_NoCI) {
  IntegrationTestData configWithSharing{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" ,
      .testVerilog = false,
      .useSharing = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(configWithSharing), 0);

  IntegrationTestData configWithoutSharing{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" ,
      .testVerilog = false,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(configWithoutSharing), 0);

  // Check if sharing brings under 5% latency increase
  EXPECT_EQ(configWithoutSharing.simTime * 1.05 > configWithSharing.simTime,
            true);

  RecordProperty("cycles", std::to_string(configWithSharing.simTime));
  logPerformance(configWithSharing.simTime);
}

TEST_P(SpecFixture, spec) {
  const std::string &name = GetParam();
  int simTime = -1;

  EXPECT_EQ(runSpecIntegrationTest(name, simTime), true);

  RecordProperty("cycles", std::to_string(simTime));
  logPerformance(simTime);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    MiscBenchmarks, BasicFixture,
    testing::Values(
      "single_loop",
      "atax",
      "atax_float",
      "bicg",
      "bicg_float",
      "binary_search",
      "covariance",
      "factorial",
      "fir",
      "float_basic",
      "gaussian",
      "gcd",
      "gemm",
      "gemm_float",
      "gemver",
      "gemver_float",
      "gesummv_float",
      "get_tanh",
      "gsum",
      "gsumif",
      "histogram",
      "if_loop_1",
      "if_loop_2",
      "if_loop_3",
      "if_loop_add",
      "if_loop_mul",
      "iir",
      "image_resize",
      "insertion_sort",
      "iterative_division",
      "iterative_sqrt",
      "jacobi_1d_imper",
      "kernel_2mm",
      "kernel_2mm_float",
      "kernel_3mm",
      "kernel_3mm_float",
      "kmp",
      "loop_array",
      "lu",
      "matching",
      "matching_2",
      "matrix",
      "matrix_power",
      "matvec",
      "mul_example",
      "mvt_float",
      "pivot",
      "polyn_mult",
      "simple_example_1",
      "sobel",
      "spmv",
      "stencil_2d",
      "sumi3_mem",
      "symm_float",
      "syr2k_float",
      "test_stdint",
      "threshold",
      "triangular",
      "vector_rescale",
      "video_filter",
      "while_loop_1",
      "while_loop_3",
      "test_loop_free",
      "test_bitint"
      ),
      [](const auto &info) { return info.param; });

// Smoke test: Using the CBC MILP solver to optimize some simple benchmarks
INSTANTIATE_TEST_SUITE_P(
    Tiny, CBCSolverFixture,
    testing::Values(
      "fir",
      "histogram",
      "if_loop_add",
      "if_loop_mul",
      "iir",
      "matvec"
      ),
      [](const auto &info) { return info.param; });

// Smoke test: Using the fast token delivery algorithm to compile the circuit.
INSTANTIATE_TEST_SUITE_P(
    Tiny, FtdWithSimpleBuffersFixture,
    testing::Values(
      // "matvec" // matvec does not work currently
      "fir",
      "if_loop_add",
      "if_loop_mul",
      "iir",
      "matvec",
      "kernel_2mm",
      "kernel_3mm",
      "kmp"
      ),
      [](const auto &info) { return info.param; });

#if 0
// Smoke test: Using the FPL22 placement algorithm to optimize some simple benchmarks
INSTANTIATE_TEST_SUITE_P(
    Tiny, FPL22Fixture,
    testing::Values(
      "fir",
      "histogram",
      "if_loop_add",
      "if_loop_mul", // Cannot break one combinational loop
      "iir",
      "matvec"
      ),
      [](const auto &info) { return info.param; });
#endif 

INSTANTIATE_TEST_SUITE_P(
    MemoryBenchmarks, MemoryFixture,
    testing::Values(
      "test_flatten_array",
      "test_memory_1",
      "test_memory_2",
      "test_memory_3",
      "test_memory_4",
      "test_memory_5",
      "test_memory_6",
      "test_memory_7",
      "test_memory_8",
      "test_memory_9",
      "test_memory_10",
      "test_memory_11",
      "test_memory_12",
      "test_memory_13",
      "test_memory_14",
      "test_memory_15",
      "test_memory_16",
      "test_memory_17",
      "test_memory_18",
      "test_smallbound",
      "test_internal_array",
      "test_constant_array"
    ),
    [](const auto &info) { return "memory_" + info.param; });

INSTANTIATE_TEST_SUITE_P(SharingUnitTests, SharingUnitTestFixture,
    testing::Values(
      "share_test_1",
      "share_test_2"),
      [](const auto &info) {
        return "sharing_" + info.param;
      });

INSTANTIATE_TEST_SUITE_P(SharingBenchmarks, SharingFixture,
    testing::Values(
      "atax_float",
      "bicg_float",
      "gsum",
      "gsumif",
      "gemm_float",
      "mvt_float",
      "syr2k_float",
      "kernel_3mm_float",
      "kernel_2mm_float",
      "gesummv_float"
      ),
    [](const auto &info) {
    return "sharing_" + info.param;
    });

INSTANTIATE_TEST_SUITE_P(SpecBenchmarks, SpecFixture,
    testing::Values(
      "single_loop",
      "fixed",
      "if_convert",
      "loop_path",
      "nested_loop",
      "sparse",
      "subdiag",
      "subdiag_fast"
      ),
    [](const auto &info) { return "spec_" + info.param; });

// Smoke test: Using the CBC MILP solver to optimize some simple benchmarks
// clang-format on

#ifdef DYNAMATIC_ENABLE_LEQ_BINARIES

TEST_P(RigidificationFixture, basic) {
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .useRigidification = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(Tiny, RigidificationFixture,
   testing::Values(
     "fir",
     "iir",
     "matvec"
     ),
   [](const auto &info) { return info.param; });
// clang-format on

#endif // DYNAMATIC_ENABLE_LEQ_BINARIES
