#ifndef DYNAMATIC_HLS_FUZZER_STATISTICS_IIREPORT
#define DYNAMATIC_HLS_FUZZER_STATISTICS_IIREPORT

#include "Histogram.h"

#include "llvm/ADT/SmallVector.h"

#include <filesystem>
#include <optional>
#include <string>

namespace dynamatic {

/// Everything one II monitor reported about its loop over a simulation,
/// gathered from the per-iteration lines it prints (see 'parseIIReport').
struct LoopIIReport {
  /// Hierarchical instance path of the monitor, which identifies the loop
  /// within the program.
  std::string loop;
  /// Nesting depth of the loop (1 for a top-level loop) and the deepest depth
  /// reachable in its nest. The loop is innermost exactly when the two are
  /// equal.
  unsigned depth = 0;
  unsigned maxDepth = 0;

  /// Intervals, in cycles, between an iteration and the one before it within
  /// the same activation.
  Histogram<unsigned> intervals;

  /// Sequential history of number of iterations.
  /// Every fresh activation adds a new entry which contains the number of
  /// iterations that activation lasted.
  llvm::SmallVector<unsigned> iterationsPerActivation;

  /// Whether the loop is an innermost one, i.e. has no loop nested inside it.
  bool isInnermost() const { return depth == maxDepth; }

  /// The loop's depth measured from the innermost loop of its nest outwards: 0
  /// for an innermost loop, 1 for a loop directly enclosing one, and so on.
  unsigned depthFromInnermost() const { return maxDepth - depth; }

  /// The loop's achieved median II over all intervals.
  /// Empty if the loop never ran more than one iteration per activation.
  std::optional<double> getMedianII() const;
};

/// Parses the II instrumentation's reports out of the simulation log in
/// 'outputDir' (the output directory dynamatic was run with, compiled with
/// '--instrument-ii') and returns one entry per loop. Returns an empty vector
/// if the log is missing or holds no report.
///
/// Every monitor prints one line per iteration of its loop, as the loop takes
/// that iteration in:
///   'II_INSTRUMENT: {"loop": <path>, "depth": <d>, "max_depth": <m>,
///                    "iter": <i>, "interval": <n>}'
/// where 'iter' is the iteration's index within its activation (so 'iter == 0'
/// opens a fresh activation) and 'interval' the number of cycles since the
/// previous iteration was taken in ('null' on the very first line of the run).
/// The monitor deliberately leaves how these are aggregated to its consumers;
/// this is where the fuzzer's statistics decide on it.
llvm::SmallVector<LoopIIReport>
parseIIReport(const std::filesystem::path &outputDir);

} // namespace dynamatic
#endif
