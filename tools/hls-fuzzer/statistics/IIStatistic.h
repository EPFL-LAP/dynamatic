#ifndef DYNAMATIC_HLS_FUZZER_STATISTICS_IISTATISTIC
#define DYNAMATIC_HLS_FUZZER_STATISTICS_IISTATISTIC

#include "Histogram.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <map>

namespace dynamatic {

/// Statistic gathering the initiation interval (II) of loops in generated
/// programs as measured by the II instrumentation. Requires the flow to have
/// been compiled with '--instrument-ii'.
class IIStatistic {
public:
  /// Records an additional sample from the simulation. 'outputDir' should be
  /// the output directory used by dynamatic.
  void update(const std::filesystem::path &outputDir);

  /// Merges the samples gathered by 'rhs' into this statistic.
  void merge(const IIStatistic &rhs);

  void print(llvm::raw_ostream &os) const;

  constexpr static llvm::StringRef CATEGORY = "II";

private:
  /// Histogram of each measured II across all loop activation windows that had
  /// a measurable II (i.e. more than one iteration).
  Histogram<double> iiCounts;

  /// Per-depth II histograms, keyed by the loop's depth measured from the
  /// innermost loop of its nest outwards: 0 for an innermost loop, 1 for a loop
  /// directly enclosing an innermost loop, and so on. This groups loops that
  /// sit at the same distance from the innermost loop regardless of the total
  /// nesting depth of their nest.
  std::map<int, Histogram<double>> iiCountsByDepth;

  /// Histogram of each loop's measured iteration count across all activation
  /// windows. Includes single-iteration windows, for which no II can be
  /// measured.
  Histogram<unsigned> iterationCounts;
};

} // namespace dynamatic
#endif
