#include "IIStatistic.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

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
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile((outputDir / "sim" / "report.txt").string());
  if (!buffer)
    return;

  // The 'ii_monitor' reports each measured loop activation window with a line
  // of the form:
  //   'II_INSTRUMENT: loop=<path> depth=<d>/<m> II=<float|n/a>
  //   iterations=<int>'
  // where 'II' is the average II measured over the window (or 'n/a' when the
  // window ran a single iteration), 'depth' is the loop's nesting depth 'd' and
  // the deepest depth 'm' reachable in its nest (equal to 'd' for an innermost
  // loop), and 'iterations' the number of iterations observed.
  for (llvm::line_iterator it(**buffer, /*SkipBlanks=*/true); !it.is_at_eof();
       ++it) {
    llvm::StringRef ref = *it;
    if (!ref.contains("II_INSTRUMENT:"))
      continue;

    // Returns the whitespace-delimited value following 'key=' (with 'key='
    // included in the argument), or nullopt if 'key=' does not appear.
    auto field = [&](llvm::StringRef key) -> std::optional<llvm::StringRef> {
      size_t pos = ref.find(key);
      if (pos == llvm::StringRef::npos)
        return std::nullopt;
      return ref.drop_front(pos + key.size()).take_until([](char c) {
        return c == ' ';
      });
    };

    // Reduce 'depth=<d>/<m>' to a depth measured from the innermost loop
    // outwards (0 == innermost). Both a 1-of-2 and a 2-of-3 loop map to 1.
    std::optional<int> depthFromInnermost;
    if (std::optional<llvm::StringRef> depthField = field("depth=")) {
      auto [depthStr, maxStr] = depthField->split('/');
      unsigned depth = 0, maxDepth = 0;
      if (!depthStr.getAsInteger(10, depth) &&
          !maxStr.getAsInteger(10, maxDepth) && maxDepth >= depth)
        depthFromInnermost = static_cast<int>(maxDepth - depth);
    }

    // Every window contributes to the iteration histogram, including
    // single-iteration ones (which report 'II=n/a').
    if (std::optional<llvm::StringRef> itersField = field("iterations=")) {
      unsigned iters = 0;
      if (!itersField->getAsInteger(10, iters))
        iterationCounts.add(iters);
    }

    // 'II=n/a' fails to parse as a double and is skipped here; such windows
    // only contribute to the iteration histogram above.
    if (std::optional<llvm::StringRef> iiField = field("II=")) {
      double ii = 0.0;
      if (!iiField->getAsDouble(ii)) {
        iiCounts.add(ii);
        if (depthFromInnermost)
          iiCountsByDepth[*depthFromInnermost].add(ii);
      }
    }
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

  // Iteration count histogram (across all windows, including single-iteration
  // ones without a measurable II).
  os << "  iteration count (gathered over " << iterationCounts.total()
     << " loops):\n";
  for (const auto &[iters, count] : iterationCounts)
    os << "    iterations=" << iters << ": " << count << " occurrence(s)\n";
  os << "  median iterations: "
     << llvm::format("%.2f", iterationCounts.median()) << '\n';
}
