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
// This set contains various memory-related integration tests.
//
//===----------------------------------------------------------------------===//
#include "../util.h"

#include <gtest/gtest.h>

class BasicMemIntegrationFixture : public testing::TestWithParam<fs::path> { };

TEST_P(BasicMemIntegrationFixture, basicMemNoFlags) {
  fs::path testPath = GetParam();
  int simTime = -1;
  
  EXPECT_EQ(runIntegrationTest(testPath, simTime), 0);

  RecordProperty("cycles", std::to_string(simTime));
}

INSTANTIATE_TEST_SUITE_P(
  BasicMemIntegration,
  BasicMemIntegrationFixture,
  testing::ValuesIn(
    findTests(
      fs::path(DYNAMATIC_ROOT) / "integration-test" / "memory"
    )
  )
);