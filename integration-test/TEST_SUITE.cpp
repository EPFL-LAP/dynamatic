#include "util.h"

#include <gtest/gtest.h>

class BasicIntegrationFixture : public testing::TestWithParam<fs::path> { };

TEST_P(BasicIntegrationFixture, basicNoFlags) {
  fs::path testPath = GetParam();
  int simTime = -1;

  EXPECT_EQ(runIntegrationTest(testPath, simTime), 0);

  RecordProperty("cycles", std::to_string(simTime));
}

INSTANTIATE_TEST_SUITE_P(
  BasicIntegration,
  BasicIntegrationFixture,
  testing::ValuesIn(
    findTests(
      fs::path(DYNAMATIC_ROOT) / "integration-test"
    )
  )
);