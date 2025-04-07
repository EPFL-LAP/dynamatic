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