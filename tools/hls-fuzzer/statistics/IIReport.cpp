#include "IIReport.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace dynamatic;

std::optional<double> LoopIIReport::getMedianII() const {
  if (intervals.empty())
    return std::nullopt;
  return intervals.median();
}

llvm::SmallVector<LoopIIReport>
dynamatic::parseIIReport(const std::filesystem::path &outputDir) {
  llvm::SmallVector<LoopIIReport> reports;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile((outputDir / "sim" / "report.txt").string());
  if (!buffer)
    return reports;

  // Index into 'reports', so that a loop's lines -- which the simulator
  // interleaves with every other loop's -- all land in the same entry while the
  // loops keep the order they were first seen in.
  llvm::StringMap<unsigned> indexByLoop;
  for (llvm::line_iterator it(**buffer, /*SkipBlanks=*/true); !it.is_at_eof();
       ++it) {
    llvm::StringRef ref = *it;
    size_t pos = ref.find("II_INSTRUMENT:");
    if (pos == llvm::StringRef::npos)
      continue;

    // The simulator surrounds the report with a severity prefix and, depending
    // on the backend, a trailing time stamp; the object itself is what lies
    // between the outermost braces of the line.
    ref = ref.drop_front(pos);
    size_t objStart = ref.find('{'), objEnd = ref.rfind('}');
    if (objStart == llvm::StringRef::npos || objEnd == llvm::StringRef::npos ||
        objEnd < objStart)
      continue;

    llvm::Expected<llvm::json::Value> value =
        llvm::json::parse(ref.slice(objStart, objEnd + 1));
    if (!value) {
      llvm::consumeError(value.takeError());
      continue;
    }
    const llvm::json::Object *report = value->getAsObject();
    if (!report)
      continue;

    std::optional<llvm::StringRef> loop = report->getString("loop");
    std::optional<int64_t> depth = report->getInteger("depth");
    std::optional<int64_t> maxDepth = report->getInteger("max_depth");
    std::optional<int64_t> iter = report->getInteger("iter");
    if (!loop || !depth || !maxDepth || !iter)
      continue;

    auto [entry, inserted] = indexByLoop.try_emplace(*loop, reports.size());
    if (inserted) {
      LoopIIReport &newReport = reports.emplace_back();
      newReport.loop = loop->str();
      newReport.depth = static_cast<unsigned>(*depth);
      newReport.maxDepth = static_cast<unsigned>(*maxDepth);
    }
    LoopIIReport &loopReport = reports[entry->second];

    if (*iter == 0) {
      // A fresh activation of the loop. Its interval, if any, spans the gap
      // since the previous activation and is dropped.
      loopReport.iterationsPerActivation.push_back(1);
      continue;
    }
    if (loopReport.iterationsPerActivation.empty())
      // An iteration of an activation the simulation did not record the start
      // of, which cannot happen but would corrupt the iteration counts.
      continue;

    ++loopReport.iterationsPerActivation.back();
    // 'null' on the very first line of the whole run, which has no previous
    // iteration to measure against.
    if (std::optional<int64_t> interval = report->getInteger("interval"))
      loopReport.intervals.add(static_cast<unsigned>(*interval));
  }
  return reports;
}
