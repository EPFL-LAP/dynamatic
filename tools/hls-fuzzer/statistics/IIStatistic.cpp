#include "IIStatistic.h"
#include "IIReport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"

#include <optional>

using namespace dynamatic;

/// Prints the '  II=<x>: <n> occurrence(s)' lines of an II histogram followed
/// by its median, each line prefixed with 'indent'.
static void printIIHistogram(llvm::raw_ostream &os, llvm::StringRef indent,
                             const Histogram<double> &hist) {
  for (const auto &[ii, count] : hist)
    os << indent << "II=" << llvm::format("%.2f", ii) << ": " << count
       << " occurrence(s)\n";
  os << indent << "median loop II: " << llvm::format("%.2f", hist.median())
     << '\n';
}

void IIStatistic::update(const std::filesystem::path &outputDir) {
  // The II instrumentation reports one line per loop iteration; 'parseIIReport'
  // gathers those into one entry per loop, from which the loop's achieved II
  // and the length of each of its activations follow.
  for (const LoopIIReport &report : parseIIReport(outputDir)) {
    // Every activation contributes to the iteration histogram, including
    // single-iteration ones (which have no interval and thus no II).
    for (unsigned iterations : report.iterationsPerActivation)
      iterationCounts.add(iterations);

    // A loop no activation of which ran two or more iterations has no interval
    // to measure an II from, and only contributes to the histogram above.
    std::optional<double> ii = report.getMedianII();
    if (!ii)
      continue;

    iiCounts.add(*ii);
    iiCountsByDepth[static_cast<int>(report.depthFromInnermost())].add(*ii);
  }
}

void IIStatistic::merge(const IIStatistic &rhs) {
  iiCounts.merge(rhs.iiCounts);
  for (const auto &[depth, hist] : rhs.iiCountsByDepth)
    iiCountsByDepth[depth].merge(hist);
  iterationCounts.merge(rhs.iterationCounts);
}

void IIStatistic::print(llvm::raw_ostream &os) const {
  // Overall II histogram (across all loops, regardless of depth).
  os << CATEGORY << " (gathered over " << iiCounts.total() << " loops):\n";
  printIIHistogram(os, "  ", iiCounts);

  // Same histogram, split by the loop's depth from the innermost loop.
  os << "  by depth from innermost:\n";
  for (const auto &[depth, hist] : iiCountsByDepth) {
    os << "    depth " << depth << (depth == 0 ? " (innermost)" : "")
       << " (gathered over " << hist.total() << " loops):\n";
    printIIHistogram(os, "      ", hist);
  }

  // Iteration count histogram (across all activations, including
  // single-iteration ones without a measurable II).
  os << "  iteration count (gathered over " << iterationCounts.total()
     << " activations):\n";
  for (const auto &[iters, count] : iterationCounts)
    os << "    iterations=" << iters << ": " << count << " occurrence(s)\n";
  os << "  median iterations: "
     << llvm::format("%.2f", iterationCounts.median()) << '\n';
}

llvm::json::Value IIStatistic::toJSON() const {
  // An array of '{depth, ii}' objects rather than an object keyed by depth, as
  // JSON object keys can only be strings.
  llvm::json::Array byDepth;
  for (const auto &[depth, hist] : iiCountsByDepth)
    byDepth.push_back(llvm::json::Object{
        {"depth", depth},
        {"ii", hist},
    });

  return llvm::json::Object{
      {"ii", iiCounts},
      {"iiByDepth", std::move(byDepth)},
      {"iterations", iterationCounts},
  };
}

bool IIStatistic::fromJSON(const llvm::json::Value &value,
                           llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map("ii", iiCounts) ||
      !mapper.map("iterations", iterationCounts))
    return false;

  const llvm::json::Array *byDepth = nullptr;
  llvm::json::Path byDepthPath = path.field("iiByDepth");
  if (const llvm::json::Object *object = value.getAsObject())
    byDepth = object->getArray("iiByDepth");
  if (!byDepth) {
    byDepthPath.report("expected array");
    return false;
  }

  iiCountsByDepth.clear();
  for (const auto &[index, entry] : llvm::enumerate(*byDepth)) {
    int depth;
    Histogram<double> hist;
    llvm::json::ObjectMapper entryMapper(entry, byDepthPath.index(index));
    if (!entryMapper || !entryMapper.map("depth", depth) ||
        !entryMapper.map("ii", hist))
      return false;

    iiCountsByDepth[depth].merge(hist);
  }
  return true;
}
