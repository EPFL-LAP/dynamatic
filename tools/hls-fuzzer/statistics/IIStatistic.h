#ifndef DYNAMATIC_HLS_FUZZER_STATISTICS_IISTATISTIC
#define DYNAMATIC_HLS_FUZZER_STATISTICS_IISTATISTIC

#include "Histogram.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
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

  /// Returns an object holding the II histogram, its per-depth breakdown and
  /// the iteration count histogram.
  llvm::json::Value toJSON() const;

  /// Replaces the samples with the ones described by 'value', which must have
  /// been created by 'toJSON'.
  bool fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  constexpr static llvm::StringRef CATEGORY = "II";

private:
  /// Histogram of the achieved II of every loop that had a measurable one, i.e.
  /// that ran at least one activation of more than one iteration. A loop
  /// contributes a single sample however often it was activated: the number of
  /// activations is a property of the loop nest around it, not of the loop's
  /// own II, so counting each activation separately would weight a program's
  /// loops by their enclosing trip counts.
  Histogram<double> iiCounts;

  /// Per-depth II histograms, keyed by the loop's depth measured from the
  /// innermost loop of its nest outwards: 0 for an innermost loop, 1 for a loop
  /// directly enclosing an innermost loop, and so on. This groups loops that
  /// sit at the same distance from the innermost loop regardless of the total
  /// nesting depth of their nest.
  std::map<int, Histogram<double>> iiCountsByDepth;

  /// Histogram of how many iterations each activation of each loop ran.
  /// Includes single-iteration activations, which contribute no II.
  Histogram<unsigned> iterationCounts;
};

} // namespace dynamatic
#endif
