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

class BasicFixture : public testing::TestWithParam<std::string> {};
class MemoryFixture : public testing::TestWithParam<std::string> {};
class SharingFixture : public testing::TestWithParam<std::string> {};
class SharingUnitTestFixture : public testing::TestWithParam<std::string> {};
class SpecFixture : public testing::TestWithParam<std::string> {};

TEST_P(BasicFixture, basic) {
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .useVerilog = false,
      .useSharing = false,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
}

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
      .useVerilog = false,
      .useSharing = false,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
}

#if 0
TEST_P(SharingUnitTestFixture, basic) {
  IntegrationTestData configWithSharing{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" / "sharing",
      .useVerilog = false,
      .useSharing = true,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(configWithSharing), 0);

  IntegrationTestData configWithoutSharing{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" / "sharing",
      .useVerilog = false,
      .useSharing = false,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(configWithoutSharing), 0);

  // Check if sharing brings under 5% latency increase
  EXPECT_EQ(configWithoutSharing.simTime * 1.05 > configWithSharing.simTime,
            true);

  RecordProperty("cycles", std::to_string(configWithSharing.simTime));
}
#endif

TEST_P(SharingFixture, basic) {
  IntegrationTestData configWithSharing{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" ,
      .useVerilog = false,
      .useSharing = true,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(configWithSharing), 0);

  IntegrationTestData configWithoutSharing{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" ,
      .useVerilog = false,
      .useSharing = false,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(configWithoutSharing), 0);

  // Check if sharing brings under 5% latency increase
  EXPECT_EQ(configWithoutSharing.simTime * 1.05 > configWithSharing.simTime,
            true);

  RecordProperty("cycles", std::to_string(configWithSharing.simTime));
}

TEST_P(SpecFixture, spec_NoCI) {
  const std::string &name = GetParam();
  int simTime = -1;

  EXPECT_EQ(runSpecIntegrationTest(name, simTime), true);

  RecordProperty("cycles", std::to_string(simTime));
}

INSTANTIATE_TEST_SUITE_P(
    MiscBenchmarks, BasicFixture,
    testing::Values("single_loop", "atax", "atax_float", "bicg", "bicg_float",
                    "binary_search", "factorial", "fir", "float_basic",
                    "gaussian", "gcd", "gemm", "gemm_float", "gemver",
                    "gemver_float", "gesummv_float", "get_tanh", "gsum",
                    "gsumif", "histogram", "if_loop_1", "if_loop_2",
                    "if_loop_3", "if_loop_add", "if_loop_mul", "iir",
                    "image_resize", "insertion_sort", "iterative_division",
                    "iterative_sqrt", "jacobi_1d_imper", "kernel_2mm",
                    "kernel_2mm_float", "kernel_3mm", "kernel_3mm_float", "kmp",
                    "loop_array", "lu", "matching", "matching_2", "matrix",
                    "matrix_power", "matvec", "mul_example", "mvt_float",
                    "pivot", "polyn_mult", "simple_example_1", "sobel", "spmv",
                    "stencil_2d", "sumi3_mem", "symm_float", "syr2k_float",
                    "test_stdint", "threshold", "triangular", "vector_rescale",
                    "video_filter", "while_loop_1", "while_loop_3"),
    [](const auto &info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(
    MemoryBenchmarks, MemoryFixture,
    testing::Values("test_flatten_array", "test_memory_1", "test_memory_2",
                    "test_memory_3", "test_memory_4", "test_memory_5",
                    "test_memory_6", "test_memory_7", "test_memory_8",
                    "test_memory_9", "test_memory_10", "test_memory_11",
                    "test_memory_12", "test_memory_13", "test_memory_14",
                    "test_memory_15", "test_memory_16", "test_memory_17",
                    "test_memory_18", "test_smallbound"),
    [](const auto &info) { return "memory_" + info.param; });

INSTANTIATE_TEST_SUITE_P(SharingUnitTests, SharingUnitTestFixture,
                         testing::Values("share_test_1", "share_test_2"),
                         [](const auto &info) {
                           return "sharing_" + info.param;
                         });

#if 0
INSTANTIATE_TEST_SUITE_P(SharingBenchmarks, SharingFixture,
                         testing::Values("gsum", "gsumif", "kernel_3mm_float",
                                         "kernel_2mm_float", "gesummv_float"),
                         [](const auto &info) {
                           return "sharing_" + info.param;
                         });
#endif

INSTANTIATE_TEST_SUITE_P(SpecBenchmarks, SpecFixture,
                         testing::Values("single_loop", "fixed", "if_convert",
                                         "loop_path", "nested_loop", "sparse",
                                         "subdiag", "subdiag_fast"),
                         [](const auto &info) { return "spec_" + info.param; });
