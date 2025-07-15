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
class GroupFixture
    : public testing::TestWithParam<std::tuple<fs::path, std::string>> {};
class SpecFixture : public testing::TestWithParam<std::string> {};

TEST_P(BasicFixture, basic) {
  std::string name = GetParam();
  int simTime = -1;

  EXPECT_EQ(runIntegrationTest(name, simTime), 0);

  RecordProperty("cycles", std::to_string(simTime));
}

TEST_P(GroupFixture, basic) {
  fs::path root = std::get<0>(GetParam());
  std::string name = std::get<1>(GetParam());
  int simTime = -1;

  EXPECT_EQ(runIntegrationTest(name, simTime, root), 0);

  RecordProperty("cycles", std::to_string(simTime));
}

TEST_P(SpecFixture, spec) {
  std::string name = GetParam();
  int simTime = -1;

  EXPECT_EQ(runSpecIntegrationTest(name, simTime), true);

  RecordProperty("cycles", std::to_string(simTime));
}

INSTANTIATE_TEST_SUITE_P(
    IndividualTests, BasicFixture,
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
                    "video_filter", "while_loop_1", "while_loop_3"));

fs::path memoryTestRoot =
    fs::path(DYNAMATIC_ROOT) / "integration-test" / "memory";
fs::path sharingTestRoot =
    fs::path(DYNAMATIC_ROOT) / "integration-test" / "sharing";

INSTANTIATE_TEST_SUITE_P(
    Memory, GroupFixture,
    testing::Combine(
        testing::Values(memoryTestRoot),
        testing::Values("test_flatten_array", "test_memory_1", "test_memory_2",
                        "test_memory_3", "test_memory_4", "test_memory_5",
                        "test_memory_6", "test_memory_7", "test_memory_8",
                        "test_memory_9", "test_memory_10", "test_memory_11",
                        "test_memory_12", "test_memory_13", "test_memory_14",
                        "test_memory_15", "test_memory_16", "test_memory_17",
                        "test_memory_18", "test_smallbound")),
    [](const auto &info) { return "memory_" + std::get<1>(info.param); });

INSTANTIATE_TEST_SUITE_P(
    Sharing, GroupFixture,
    testing::Combine(testing::Values(sharingTestRoot),
                     testing::Values("share_test_1", "share_test_2")),
    [](const auto &info) { return "sharing_" + std::get<1>(info.param); });

INSTANTIATE_TEST_SUITE_P(SpecSingleLoop_NoCI, SpecFixture,
                         testing::Values("single_loop", "fixed", "if_convert",
                                         "loop_path", "nested_loop",
                                         "single_loop", "sparse", "subdiag",
                                         "subdiag_fast"));