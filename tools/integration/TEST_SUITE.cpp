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

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Parser for custom flags that are not googletest
//
// Remarks:
// - This does not use the LLVM command line parser infrastructure because the
// googletest flags are seen as undefined flags.
// - This does not use abseil command line parser to avoid adding extra
// dependencies.
namespace {
// Make it a global variable to avoid the need to route this flag all the way to
// the integration test config
bool clVerboseOutDir = false;
void parseClOptions(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--verbose-outdir", 0) == 0) {
      // Global variable
      clVerboseOutDir = true;
    }

    // Print out flags before gtest's flags
    if (arg.rfind("--help") == 0 || arg.rfind("-h") == 0 ||
        arg.rfind("--gtest_help") == 0) {
      std::cout << "---------------------------------------------\n";
      std::cout << "---------------------------------------------\n";
      std::cout << "---- Dynamatic's integration test driver ----\n";
      std::cout << "---------------------------------------------\n";
      std::cout << "---------------------------------------------\n";
      std::cout << "Custom options:\n";
      std::cout << "-h|--help            print this page\n";
      std::cout << "--verbose-outdir     instead of \"out\", explicitly print "
                   "the option into the output directory\n";
      std::cout << "----------------------------------------------\n\n";
    }
  }
}
} // namespace

struct IntegrationTestData {
  // Configurations
  std::string name;
  fs::path benchmarkPath;
  bool testVerilog;
  bool testVHDL = true; // default to true
  // Use resource sharing to reduce the functional unit usage.
  bool useSharing = false;
  // Use model checking to remove redundant logic.
  bool useRigidification = false;
  bool verifyInvariants = false;
  // Enable speculation, using the speculate pragma
  bool useSpeculation = false;
  std::string milpSolver = "gurobi";
  std::string bufferAlgorithm = "fpga20";
  unsigned clockPeriod = 5;

  // Results
  int simTime;

  // This func. generate a prefix string according to the configuration.
  // For example, for the default configuration, we generate:
  // out-hdl:vhdl-milpSolver:gurobi-bufferAlgorithm:fpga20-cp:5
  //
  // For example, if we just enable sharing
  // out-hdl:vhdl-sharing:on-milpSolver:gurobi-bufferAlgorithm:fpga20-cp:5
  std::string getVerboseOutputDirName() {
    std::vector<std::string> symbols{"out"};

    auto stringifyBoolean = [](bool b) {
      return std::string(b ? "on" : "off");
    };

    if (useSharing)
      symbols.emplace_back("sharing:" + stringifyBoolean(this->useSharing));

    if (this->useRigidification)
      symbols.emplace_back("rigidification:" +
                           stringifyBoolean(this->useRigidification));

    if (this->verifyInvariants)
      symbols.emplace_back("verifyInvariants:" +
                           stringifyBoolean(this->verifyInvariants));

    if (this->useSpeculation)
      symbols.emplace_back("useSpeculation:" +
                           stringifyBoolean(this->useSpeculation));

    symbols.emplace_back("milpSolver:" + this->milpSolver);
    symbols.emplace_back("bufferAlgorithm:" + this->bufferAlgorithm);
    symbols.emplace_back("cp:" + std::to_string(this->clockPeriod));

    // Generating the output file name: Interleaving the fields with "-".
    std::stringstream ss;
    bool started = false;
    for (const auto &symbol : symbols) {
      if (started)
        ss << "-";
      started = true;
      ss << symbol;
    }
    return ss.str();
  }
};

namespace {

int getSimulationTime(const fs::path &logFile) {
  std::ifstream file(logFile);
  if (!file.is_open()) {
    std::cout << "[WARNING] Failed to open " << logFile << std::endl;
    return -1;
  }

  std::vector<std::string> lines;
  std::string line;

  // Read all lines into a vector
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  std::regex pattern("Simulation done! Latency = (\\d+) cycles");
  std::smatch match;

  // Search lines in reverse order
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
    if (std::regex_search(*it, match, pattern)) {
      return std::stoi(match[1]);
    }
  }

  std::cout << "[WARNING] Log file does not contain simulation time!"
            << std::endl;
  return -1;
}

int runIntegrationTest(IntegrationTestData &config) {
  fs::path cSourcePath =
      config.benchmarkPath / config.name / (config.name + ".c");

  std::string tmpFilename = "tmp_" + config.name + ".dyn";
  std::ofstream scriptFile(tmpFilename);
  if (!scriptFile.is_open()) {
    std::cout << "[ERROR] Failed to create .dyn script file" << std::endl;
    return -1;
  }

  std::string outputDirName;
  if (clVerboseOutDir)
    outputDirName = config.getVerboseOutputDirName();
  else
    outputDirName = "out";

  scriptFile << "set-dynamatic-path " << DYNAMATIC_ROOT << std::endl
             << "set-src " << cSourcePath.string() << std::endl
             << "set-clock-period " << config.clockPeriod << std::endl
             << "set-output-dir " << outputDirName << std::endl;

  // clang-format off
  scriptFile << "compile"
             << " --buffer-algorithm " << config.bufferAlgorithm
             << (config.useSharing ? " --sharing" : "")
             << (config.useRigidification ? " --rigidification" : "")
             << (config.useSpeculation ? " --speculation" : "")
             << " --milp-solver " << config.milpSolver << std::endl;
  // clang-format on

  // Assert testVHDL or testVerilog is true
  if (!config.testVHDL && !config.testVerilog) {
    std::cout << "[ERROR] Either testVHDL or testVerilog must be true"
              << std::endl;
    return -1;
  }

  if (config.verifyInvariants) {
    scriptFile << "verify-invariants" << std::endl;
  }

  // Verify Verilog works correctly
  if (config.testVerilog) {
    scriptFile << "write-hdl --hdl verilog" << std::endl
               << "simulate" << std::endl;
  }
  // Verify VHDL works correctly
  if (config.testVHDL) {
    // By default, the report containing the simulation time is re-written
    // during the second simulation (i.e., the VHDL simulation).
    scriptFile << "write-hdl --hdl vhdl" << std::endl
               << "simulate" << std::endl;
  }
  scriptFile << "exit" << std::endl;

  scriptFile.close();

  fs::path dynamaticPath = fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic";
  fs::path dynamaticOutPath =
      cSourcePath.parent_path() / outputDirName / "dynamatic_out.txt";
  fs::path dynamaticErrPath =
      cSourcePath.parent_path() / outputDirName / "dynamatic_err.txt";
  if (!fs::exists(dynamaticOutPath.parent_path())) {
    fs::create_directories(dynamaticOutPath.parent_path());
  }

  std::string cmd = dynamaticPath.string() + " --exit-on-failure --run ";
  cmd += tmpFilename;
  cmd += " 1> ";
  cmd += dynamaticOutPath;
  cmd += " 2> ";
  cmd += dynamaticErrPath;

  int status = system(cmd.c_str());
  if (status == 0) {
    fs::path logFilePath =
        cSourcePath.parent_path() / outputDirName / "sim" / "report.txt";
    config.simTime = getSimulationTime(logFilePath);
  }

  return status;
}

} // namespace

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
#ifdef DYNAMATIC_ENABLE_CBC
class CBCSolverFixture : public BaseFixture {};
#endif // DYNAMATIC_ENABLE_CBC
// Use FPL22 placement algorithm on a small subset of MiscBenchmarks
class FPL22Fixture : public BaseFixture {};
class MemoryFixture : public BaseFixture {};
class SharingFixture : public BaseFixture {};
class SharingUnitTestFixture : public BaseFixture {};
class SpecFixture : public BaseFixture {};

class RigidificationFixture : public BaseFixture {};
class VerifyInvariantsFixture : public BaseFixture {};

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

#ifdef DYNAMATIC_ENABLE_CBC
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
#endif // DYNAMATIC_ENABLE_CBC

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
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .useSpeculation = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .clockPeriod = 7,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
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
      "test_bitint",
      "test_int16",
      "test_double",
      "unused_arg",
      "test_bool_array",
      "test_divui",
      "test_fneg"
      ),
      [](const auto &info) { return info.param; });

#ifdef DYNAMATIC_ENABLE_CBC
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
#endif // DYNAMATIC_ENABLE_CBC

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

INSTANTIATE_TEST_SUITE_P(Tiny, VerifyInvariantsFixture,
                         testing::Values("fir", "iir", "matvec"),
                         [](const auto &info) { return info.param; });

TEST_P(VerifyInvariantsFixture, basic) {
  IntegrationTestData config{
      // clang-format off
      .name = GetParam(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .useRigidification = false,
      .verifyInvariants = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "on-merges",
      .simTime = -1
      // clang-format on
  };

  EXPECT_EQ(runIntegrationTest(config), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

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

int main(int argc, char **argv) {
  parseClOptions(argc, argv);
  // https://google.github.io/googletest/primer.html#writing-the-main-function
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
