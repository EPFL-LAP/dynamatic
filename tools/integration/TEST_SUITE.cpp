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
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

/// The aggregated measurements of one II monitor, computed from the
/// per-iteration "II_INSTRUMENT: ..." lines the monitor prints during
/// simulation (one per loop iteration, as the slowest header mux takes it
/// in).
struct IISummary {
  /// The loop's nesting depth and the deepest depth in its nest.
  int depth;
  int maxDepth;
  /// The achieved II: the median of the intervals between consecutive
  /// iterations within an activation (the "iter>0" lines; see the II
  /// monitor's entity comment). The median is the typical interval, robust
  /// to warmup and other outliers (and thereby to much of the lower-bound
  /// bias the monitor documents for loops paced behind memory queues). -1
  /// when no activation ran two or more iterations.
  double medianII;
  /// The total number of iterations the loop ran.
  int iterations;
  /// How often the loop was entered from outside.
  int activations;
};

std::vector<IISummary> parseIISummaries(const fs::path &logFile) {
  std::vector<IISummary> summaries;
  std::vector<std::string> loops;
  std::vector<std::vector<long>> intervals;
  std::ifstream file(logFile);
  if (!file.is_open()) {
    std::cout << "[WARNING] Failed to open " << logFile << std::endl;
    return summaries;
  }

  const std::string prefix = "II_INSTRUMENT: ";
  std::string line;
  while (std::getline(file, line)) {
    size_t pos = line.find(prefix);
    if (pos == std::string::npos)
      continue;
    nlohmann::json report =
        nlohmann::json::parse(line.substr(pos + prefix.size()));

    std::string loop = report["loop"];
    auto it = std::find(loops.begin(), loops.end(), loop);
    if (it == loops.end()) {
      loops.push_back(loop);
      summaries.push_back({report["depth"].get<int>(),
                           report["max_depth"].get<int>(), -1.0, 0, 0});
      intervals.emplace_back();
      it = std::prev(loops.end());
    }
    size_t idx = it - loops.begin();
    ++summaries[idx].iterations;
    if (report["iter"].get<int>() == 0) {
      // An activation's first iteration; its interval spans the gap between
      // two activations and belongs to neither.
      ++summaries[idx].activations;
    } else {
      intervals[idx].push_back(report["interval"].get<long>());
    }
  }
  for (size_t idx = 0; idx < summaries.size(); ++idx) {
    std::vector<long> &loopIntervals = intervals[idx];
    if (loopIntervals.empty())
      continue;
    std::sort(loopIntervals.begin(), loopIntervals.end());
    size_t mid = loopIntervals.size() / 2;
    summaries[idx].medianII =
        loopIntervals.size() % 2 != 0
            ? double(loopIntervals[mid])
            : (loopIntervals[mid - 1] + loopIntervals[mid]) / 2.0;
  }
  return summaries;
}
} // namespace

struct IntegrationTest {
  // Configurations
  std::string name;
  // Use to deduplicate the generate output file folder
  std::string testName;
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
  bool useDuplication = false;
  std::string milpSolver = "gurobi";
  std::string bufferAlgorithm = "fpga20";
  unsigned clockPeriod = 5;
  // Insert an II monitor per loop of the circuit.
  bool instrumentII = false;

  // Results
  int simTime;
  int run();

  /// Path of the simulation log, which also holds the II monitors' reports.
  fs::path simReportPath() const {
    return benchmarkPath / name / ("out_" + testName) / "sim" / "report.txt";
  }
};

int IntegrationTest::run() {

  fs::path cSourcePath = this->benchmarkPath / this->name / (this->name + ".c");

  assert(this->testName.size() > 0);

  std::string tmpFilename = "tmp_" + this->name + "_" + this->testName + ".dyn";

  std::ofstream scriptFile(tmpFilename);

  if (!scriptFile.is_open()) {
    std::cout << "[ERROR] Failed to create .dyn script file" << std::endl;
    return -1;
  }

  std::string outputDirName;

  outputDirName = "out_" + this->testName;

  scriptFile << "set-dynamatic-path " << DYNAMATIC_ROOT << std::endl
             << "set-src " << cSourcePath.string() << std::endl
             << "set-clock-period " << this->clockPeriod << std::endl
             << "set-output-dir " << outputDirName << std::endl;

  // clang-format off
  scriptFile << "compile"
             << " --buffer-algorithm " << this->bufferAlgorithm
             << (this->useSharing ? " --sharing" : "")
             << (this->useRigidification ? " --rigidification" : "")
             << (this->useSpeculation ? " --speculation" : "")
             << (this->useDuplication ? " --enable-duplication" : "")
             << (this->instrumentII ? " --instrument-ii" : "")
             << " --milp-solver " << this->milpSolver << std::endl;
  // clang-format on

  // Assert testVHDL or testVerilog is true
  if (!this->testVHDL && !this->testVerilog) {
    std::cout << "[ERROR] Either testVHDL or testVerilog must be true"
              << std::endl;
    return -1;
  }

  if (this->verifyInvariants) {
    scriptFile << "verify-invariants" << std::endl;
  }

  // Verify Verilog works correctly
  if (this->testVerilog) {
    scriptFile << "write-hdl --hdl verilog" << std::endl
               << "simulate" << std::endl;
  }
  // Verify VHDL works correctly
  if (this->testVHDL) {
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
    this->simTime = getSimulationTime(simReportPath());
  }

  return status;
}

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

  // Use the fixture name as the suffix of the outdir when we set
  // `--verbose-outdir`
  std::string getVerboseOutdirSuffix() const {
    auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    const auto *fixtureName = info->test_suite_name();
    return std::regex_replace(fixtureName, std::regex("/"), "_");
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
class DuplicationFixture : public BaseFixture {};

class RigidificationFixture : public BaseFixture {};
class VerifyInvariantsFixture : public BaseFixture {};

TEST_P(BasicFixture, basic) {
  IntegrationTest config{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = true,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1,
      // clang-format on
  };
  EXPECT_EQ(config.run(), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

#ifdef DYNAMATIC_ENABLE_CBC
TEST_P(CBCSolverFixture, basic) {
  IntegrationTest config{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = true,
      .useSharing = false,
      .milpSolver = "cbc",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(config.run(), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}
#endif // DYNAMATIC_ENABLE_CBC

#if 0
TEST_P(FPL22Fixture, basic) {
  IntegrationTest config{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpl22",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(config.run(), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
}
#endif

TEST_P(MemoryFixture, basic) {
  IntegrationTest config{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" / "memory",
      .testVerilog = true,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(config.run(), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

/// This testing fixture runs the test with and without sharing. It checks
/// whenever the sharing option is enabled, the pass can run without any
/// interruption and does not penalize the latency.
TEST_P(SharingUnitTestFixture, basic) {
  IntegrationTest configWithSharing{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" / "sharing",
      .testVerilog = false,
      .useSharing = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(configWithSharing.run(), 0);

  IntegrationTest configWithoutSharing{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" / "sharing",
      .testVerilog = false,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(configWithoutSharing.run(), 0);

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
  IntegrationTest configWithSharing{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" ,
      .testVerilog = false,
      .useSharing = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(configWithSharing.run(), 0);

  IntegrationTest configWithoutSharing{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test" ,
      .testVerilog = false,
      .useSharing = false,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(configWithoutSharing.run(), 0);

  // Check if sharing brings under 5% latency increase
  EXPECT_EQ(configWithoutSharing.simTime * 1.05 > configWithSharing.simTime,
            true);

  RecordProperty("cycles", std::to_string(configWithSharing.simTime));
  logPerformance(configWithSharing.simTime);
}

TEST_P(SpecFixture, spec) {
  IntegrationTest config{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .useSpeculation = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpl22",
      .clockPeriod = 20,
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(config.run(), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

TEST_P(DuplicationFixture, basic) {
  IntegrationTest config{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .useDuplication = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(config.run(), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

/// Compiles benchmarks with '--instrument-ii' and checks the per-loop
/// measurements aggregated from the II monitors' per-iteration reports.
class IIMonitorFixture : public testing::Test {
protected:
  /// Compiles and simulates 'name' with II instrumentation and returns the
  /// per-loop aggregates of the reported iterations.
  std::vector<IISummary> runInstrumented(const std::string &name) {
    IntegrationTest config{
        .name = name,
        .testName = "IIMonitorTests",
        .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
        .testVerilog = false,
        .instrumentII = true,
    };
    EXPECT_EQ(config.run(), 0);
    return parseIISummaries(config.simReportPath());
  }
};

// A single loop with a statically known iteration count, activated once.
TEST_F(IIMonitorFixture, singleLoop) {
  // 'fir' runs its only loop once, for exactly 1000 iterations.
  std::vector<IISummary> summaries = runInstrumented("fir");
  ASSERT_EQ(summaries.size(), 1u);
  const IISummary &summary = summaries.front();
  EXPECT_EQ(summary.depth, 1);
  EXPECT_EQ(summary.maxDepth, 1);
  EXPECT_EQ(summary.iterations, 1000);
  EXPECT_EQ(summary.activations, 1);
  // The loop pipelines perfectly.
  EXPECT_EQ(summary.medianII, 1.0);
}

// A loop nest: both loops report, tagged with their nesting depth; the inner
// one accumulates its iterations across one activation per outer iteration.
TEST_F(IIMonitorFixture, nestedLoops) {
  // 'matvec' iterates a 100x100 nest once.
  std::vector<IISummary> summaries = runInstrumented("matvec");
  ASSERT_EQ(summaries.size(), 2u);

  for (const IISummary &summary : summaries)
    EXPECT_EQ(summary.maxDepth, 2);
  auto isOuter = [](const IISummary &summary) { return summary.depth == 1; };
  auto outerIt = std::find_if(summaries.begin(), summaries.end(), isOuter);
  auto innerIt = std::find_if_not(summaries.begin(), summaries.end(), isOuter);
  ASSERT_NE(outerIt, summaries.end());
  ASSERT_NE(innerIt, summaries.end());

  EXPECT_EQ(outerIt->iterations, 100);
  EXPECT_EQ(outerIt->activations, 1);

  EXPECT_EQ(innerIt->depth, 2);
  EXPECT_EQ(innerIt->iterations, 100 * 100);
  EXPECT_EQ(innerIt->activations, 100);
  EXPECT_EQ(innerIt->medianII, 1.0);
  // An outer iteration contains a full inner activation.
  EXPECT_GT(outerIt->medianII, innerIt->medianII);
}

// A while loop whose iteration count is only decided by the data at runtime.
TEST_F(IIMonitorFixture, dataDependentTripCount) {
  // 'while_loop_1' scans until a[i] + b[i] >= 1000, which its input data
  // arranges to first happen at i == 900: 901 iterations.
  std::vector<IISummary> summaries = runInstrumented("while_loop_1");
  ASSERT_EQ(summaries.size(), 1u);
  const IISummary &summary = summaries.front();
  EXPECT_EQ(summary.iterations, 901);
  EXPECT_EQ(summary.activations, 1);
  // The comparison feeding the exit decision is on the loop's recurrence.
  EXPECT_EQ(summary.medianII, 4.0);
}

// A program without loops gets no monitors and must still simulate cleanly.
TEST_F(IIMonitorFixture, loopFree) {
  std::vector<IISummary> summaries = runInstrumented("test_loop_free");
  EXPECT_TRUE(summaries.empty());
}

// The monitor's boundary conditions: a loop that never runs prints nothing
// (its iterations are the events being reported), a single iteration yields
// no measurable II, and two iterations are the shortest activation that
// does.
TEST_F(IIMonitorFixture, iterationCountEdgeCases) {
  std::vector<IISummary> summaries = runInstrumented("ii_edge_cases");
  ASSERT_EQ(summaries.size(), 2u);
  std::vector<int> iterations;
  for (const IISummary &summary : summaries) {
    EXPECT_EQ(summary.depth, 1);
    EXPECT_EQ(summary.maxDepth, 1);
    EXPECT_EQ(summary.activations, 1);
    iterations.push_back(summary.iterations);
    if (summary.iterations < 2) {
      // A single iteration taken in: no interval to see.
      EXPECT_EQ(summary.medianII, -1.0);
    } else {
      EXPECT_GE(summary.medianII, 1.0);
    }
  }
  std::sort(iterations.begin(), iterations.end());
  EXPECT_EQ(iterations, (std::vector<int>{1, 2}));
}

// A frequently re-entered inner loop whose recurrence takes many cycles: the
// intervals bridging its activations are shorter than its II, so they must
// not leak into the measurement and drag it down.
TEST_F(IIMonitorFixture, nestedHighLatency) {
  // 'ii_nested_float' re-enters its inner loop 20 times for only 5 iterations
  // each, accumulating floats: a multi-cycle addf recurrence.
  std::vector<IISummary> summaries = runInstrumented("ii_nested_float");
  ASSERT_EQ(summaries.size(), 2u);

  auto isOuter = [](const IISummary &summary) { return summary.depth == 1; };
  auto outerIt = std::find_if(summaries.begin(), summaries.end(), isOuter);
  auto innerIt = std::find_if_not(summaries.begin(), summaries.end(), isOuter);
  ASSERT_NE(outerIt, summaries.end());
  ASSERT_NE(innerIt, summaries.end());

  EXPECT_EQ(innerIt->iterations, 20 * 5);
  EXPECT_EQ(innerIt->activations, 20);
  // The addf takes 6 cycles: a typical interval below that would mean
  // intervals bridging two activations leaked into the measurement; one
  // above it, that the loop did not actually pace on the addf.
  EXPECT_EQ(innerIt->medianII, 6.0);

  EXPECT_EQ(outerIt->iterations, 20);
  EXPECT_EQ(outerIt->activations, 1);
  // An outer iteration contains a full inner activation.
  EXPECT_GT(outerIt->medianII, innerIt->medianII);
}

// The opposite extreme: an inner loop that pipelines perfectly. Its measured
// II must come out as exactly 1, so any overshoot in the measurement shows.
TEST_F(IIMonitorFixture, nestedIIOne) {
  // 'ii_nested_int' re-enters its inner loop 8 times for 30 iterations each,
  // accumulating integers: a combinational recurrence.
  std::vector<IISummary> summaries = runInstrumented("ii_nested_int");
  ASSERT_EQ(summaries.size(), 2u);

  auto isOuter = [](const IISummary &summary) { return summary.depth == 1; };
  auto outerIt = std::find_if(summaries.begin(), summaries.end(), isOuter);
  auto innerIt = std::find_if_not(summaries.begin(), summaries.end(), isOuter);
  ASSERT_NE(outerIt, summaries.end());
  ASSERT_NE(innerIt, summaries.end());

  EXPECT_EQ(innerIt->iterations, 8 * 30);
  EXPECT_EQ(innerIt->activations, 8);
  // A perfectly pipelined loop initiates every cycle; any overshoot in the
  // measurement shows here.
  EXPECT_EQ(innerIt->medianII, 1.0);

  EXPECT_EQ(outerIt->iterations, 8);
  EXPECT_EQ(outerIt->activations, 1);
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
      "subdiag_fast",
      "newton_raphson",
      "backtrack"
      ),
    [](const auto &info) { return "spec_" + info.param; });

INSTANTIATE_TEST_SUITE_P(DuplicationBenchmarks, DuplicationFixture,
    testing::Values(
      "divergent_paths",
      "nested_conditionals_1",
      "nested_conditionals_2",
      "prediction",
      "sparse",
      "wrap_if"
    ),
    [](const auto &info) { return "dup_" + info.param; });

// Smoke test: Using the CBC MILP solver to optimize some simple benchmarks
// clang-format on

#ifdef DYNAMATIC_ENABLE_LEQ_BINARIES

INSTANTIATE_TEST_SUITE_P(Tiny, VerifyInvariantsFixture,
                         testing::Values("fir", "iir", "matvec"),
                         [](const auto &info) { return info.param; });

TEST_P(VerifyInvariantsFixture, basic) {
  IntegrationTest config{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
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

  EXPECT_EQ(config.run(), 0);
  RecordProperty("cycles", std::to_string(config.simTime));
  logPerformance(config.simTime);
}

TEST_P(RigidificationFixture, basic) {
  IntegrationTest config{
      // clang-format off
      .name = GetParam(),
      .testName = getVerboseOutdirSuffix(),
      .benchmarkPath = fs::path(DYNAMATIC_ROOT) / "integration-test",
      .testVerilog = false,
      .useSharing = false,
      .useRigidification = true,
      .milpSolver = "gurobi",
      .bufferAlgorithm = "fpga20",
      .simTime = -1
      // clang-format on
  };
  EXPECT_EQ(config.run(), 0);
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
