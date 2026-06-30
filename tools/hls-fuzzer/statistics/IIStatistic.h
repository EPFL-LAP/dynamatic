#ifndef DYNAMATIC_HLS_FUZZER_STATISTICS_IISTATISTIC
#define DYNAMATIC_HLS_FUZZER_STATISTICS_IISTATISTIC

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
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
  /// Histogram mapping each measured II to the number of loops that exhibited
  /// it. Ordered so that the median can be computed directly.
  std::map<double, std::size_t> iiCounts;

  /// Number of loops that have been sampled.
  std::size_t numLoops = 0;
};

} // namespace dynamatic
#endif
