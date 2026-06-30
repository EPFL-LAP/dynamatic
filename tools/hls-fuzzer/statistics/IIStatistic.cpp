#include "IIStatistic.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace dynamatic;

void IIStatistic::update(const std::filesystem::path &outputDir) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile((outputDir / "sim" / "report.txt").string());
  if (!buffer)
    return;

  // The 'ii_monitor' reports each measured loop with a line of the form
  // 'II_INSTRUMENT: II=<float> iterations=<int>', where 'II' is the average II
  // measured over one activation window of an innermost loop.
  constexpr llvm::StringLiteral iiPrefix = "II=";

  for (llvm::line_iterator it(**buffer, /*SkipBlanks=*/true); !it.is_at_eof();
       ++it) {
    llvm::StringRef ref = *it;
    if (!ref.contains("II_INSTRUMENT:"))
      continue;

    size_t iiPos = ref.find(iiPrefix);
    if (iiPos == llvm::StringRef::npos)
      continue;

    llvm::StringRef iiField = ref.drop_front(iiPos + iiPrefix.size());
    iiField = iiField.take_until([](char c) { return c == ' '; });
    double ii = 0.0;
    if (iiField.getAsDouble(ii))
      continue;

    iiCounts[ii]++;
    numLoops++;
  }
}

void IIStatistic::merge(const IIStatistic &rhs) {
  for (const auto &[ii, count] : rhs.iiCounts)
    iiCounts[ii] += count;
  numLoops += rhs.numLoops;
}

void IIStatistic::print(llvm::raw_ostream &os) const {
  os << CATEGORY << " (gathered over " << numLoops << " loops):\n";

  // Walk the ordered histogram to compute the median while also reporting the
  // number of occurrences of each measured II.
  double median = 0.0;
  std::size_t seen = 0;
  std::size_t lowerHalf = numLoops / 2;
  for (const auto &[ii, count] : iiCounts) {
    os << "  II=" << llvm::format("%.2f", ii) << ": " << count
       << " occurrence(s)\n";

    // The median sits at index 'lowerHalf' (or, for an even count, between
    // 'lowerHalf - 1' and 'lowerHalf'). Capture both samples as they are
    // crossed.
    if (numLoops % 2 == 0 && seen <= lowerHalf - 1 &&
        seen + count > lowerHalf - 1)
      median += ii / 2.0;
    if (seen <= lowerHalf && seen + count > lowerHalf)
      median += numLoops % 2 == 0 ? ii / 2.0 : ii;

    seen += count;
  }

  os << "  median loop II: " << llvm::format("%.2f", median) << '\n';
}
