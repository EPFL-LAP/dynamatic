#include "util.h"

#include <gtest/gtest.h>

class BasicIntegrationFixture : public testing::TestWithParam<fs::path> { };

TEST_P(BasicIntegrationFixture, basicNoFlags) {
  fs::path testPath = GetParam();
  EXPECT_EQ(runIntegrationTest(testPath), 0);
}

INSTANTIATE_TEST_SUITE_P(
  BasicIntegration,
  BasicIntegrationFixture,
  testing::ValuesIn(
    findTests(
      fs::path(DYNAMATIC_ROOT) / fs::path("integration-test")
    )
  )
);