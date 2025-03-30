#include "TEST_SUITE.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <fstream>

class BasicIntegrationFixture : public testing::TestWithParam<std::string> { };

int runIntegrationTest(const std::string& name) {
  std::string tmpFilename = "tmp_" + name + ".dyn";
  std::ofstream script_file(tmpFilename);
  if (!script_file.is_open()) {
    std::cout << "Failed to create .dyn script file" << std::endl;
    return -1;
  } 

  script_file << "set-dynamatic-path " << DYNAMATIC_ROOT << std::endl
    << "set-src " << DYNAMATIC_ROOT << "/integration-test/" 
    << name << "/" << name << ".c" << std::endl
    << "compile" << std::endl
    << "write-hdl" << std::endl
    << "simulate" << std::endl
    << "exit" << std::endl;
  
  script_file.close();

  std::string cmd = DYNAMATIC_ROOT;
  cmd += "/bin/dynamatic --exit-on-failure --run ";
  cmd += tmpFilename;
  return system(cmd.c_str());
}

TEST_P(BasicIntegrationFixture, basicNoFlags) {
  std::string testName = GetParam();
  EXPECT_EQ(runIntegrationTest(testName), 0);
}

INSTANTIATE_TEST_SUITE_P(
  BasicIntegration,
  BasicIntegrationFixture,
  testing::Values(
    "gcd",
    "binary_search",
    "kernel_3mm_float"
  )
);