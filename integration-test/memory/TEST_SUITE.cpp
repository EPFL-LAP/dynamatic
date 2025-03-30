#include "../util.h"

#include <gtest/gtest.h>

class BasicMemIntegrationFixture : public testing::TestWithParam<fs::path> { };

TEST_P(BasicMemIntegrationFixture, basicMemNoFlags) {
  fs::path testPath = GetParam();
  EXPECT_EQ(runIntegrationTest(testPath), 0);
}

INSTANTIATE_TEST_SUITE_P(
  BasicMemIntegration,
  BasicMemIntegrationFixture,
  testing::ValuesIn(
    findTests(
      fs::path(DYNAMATIC_ROOT) / fs::path("integration-test") / fs::path("memory")
    )
  )
);