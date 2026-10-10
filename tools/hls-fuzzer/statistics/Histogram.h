#ifndef DYNAMATIC_HLS_FUZZER_STATISTICS_HISTOGRAM
#define DYNAMATIC_HLS_FUZZER_STATISTICS_HISTOGRAM

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>

namespace dynamatic {

/// A frequency histogram mapping observed values of type 'T' to the number of
/// times they were observed. Values are kept ordered so the median and any
/// rendering come out sorted. Formatting is intentionally left to callers (it
/// differs per statistic); iterate the histogram with 'begin()'/'end()' to
/// print it.
template <typename T>
class Histogram {
public:
  /// Records 'count' additional observations of 'value'.
  void add(T value, std::size_t count = 1) { counts[value] += count; }

  /// Merges the observations of 'rhs' into this histogram.
  void merge(const Histogram &rhs) {
    for (const auto &[value, count] : rhs.counts)
      counts[value] += count;
  }

  /// Total number of observations across all values.
  std::size_t total() const {
    std::size_t sum = 0;
    for (const auto &[value, count] : counts)
      sum += count;
    return sum;
  }

  bool empty() const { return counts.empty(); }

  /// The median of all observations. For an even number of observations the
  /// mean of the two central values is returned; an empty histogram has median
  /// 0.
  double median() const {
    std::size_t n = total();
    if (n == 0)
      return 0.0;

    double median = 0.0;
    std::size_t seen = 0;
    std::size_t lowerHalf = n / 2;
    for (const auto &[value, count] : counts) {
      // The median sits at index 'lowerHalf' (or, for an even count, between
      // 'lowerHalf - 1' and 'lowerHalf'). Capture both samples as they are
      // crossed.
      if (n % 2 == 0 && seen <= lowerHalf - 1 && seen + count > lowerHalf - 1)
        median += static_cast<double>(value) / 2.0;
      if (seen <= lowerHalf && seen + count > lowerHalf)
        median += n % 2 == 0 ? static_cast<double>(value) / 2.0
                             : static_cast<double>(value);
      seen += count;
    }
    return median;
  }

  /// Ordered iteration over '{value, count}' pairs, for rendering.
  auto begin() const { return counts.begin(); }
  auto end() const { return counts.end(); }

private:
  std::map<T, std::size_t> counts;
};

/// The type a histogram value of type 'T' is parsed as before being narrowed
/// back to 'T'. 'llvm::json' only parses the widest integer and floating-point
/// types, so a 'Histogram<unsigned>' has to go through 'uint64_t'.
template <typename T>
using HistogramParseType = std::conditional_t<
    std::is_floating_point_v<T>, double,
    std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>>;

/// Serializes 'histogram' as an array of '{"value": ..., "count": ...}'
/// objects, ordered by value.
///
/// Counts are written raw rather than as a share of the total, so that a
/// histogram read back by 'fromJSON' is indistinguishable from the original and
/// can still be merged into another one.
template <typename T>
llvm::json::Value toJSON(const Histogram<T> &histogram) {
  llvm::json::Array entries;
  for (const auto &[value, count] : histogram)
    entries.push_back(llvm::json::Object{
        {"value", value},
        {"count", static_cast<uint64_t>(count)},
    });
  return entries;
}

/// Parses a histogram written by 'toJSON', replacing the contents of
/// 'histogram'. Returns false if 'value' is not a valid representation,
/// reporting why to 'path'.
template <typename T>
bool fromJSON(const llvm::json::Value &value, Histogram<T> &histogram,
              llvm::json::Path path) {
  const llvm::json::Array *entries = value.getAsArray();
  if (!entries) {
    path.report("expected array");
    return false;
  }

  histogram = Histogram<T>();
  for (const auto &[index, entry] : llvm::enumerate(*entries)) {
    HistogramParseType<T> entryValue;
    uint64_t count;
    llvm::json::ObjectMapper mapper(entry, path.index(index));
    if (!mapper || !mapper.map("value", entryValue) ||
        !mapper.map("count", count))
      return false;

    histogram.add(static_cast<T>(entryValue), static_cast<std::size_t>(count));
  }
  return true;
}

} // namespace dynamatic
#endif
