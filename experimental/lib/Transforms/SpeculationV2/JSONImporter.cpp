#include "JSONImporter.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <fstream>

using namespace llvm::sys;
using namespace mlir;

FailureOr<SpecSpecification> readFromJSON(const std::string &jsonPath) {
  // Open the speculation file
  std::ifstream inputFile(jsonPath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open kernel information file for speculation\n";
    return failure();
  }

  // Read the JSON content from the file and into a string
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line))
    jsonString += line;

  // Try to parse the string as a JSON
  llvm::Expected<llvm::json::Value> value = llvm::json::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse kernel information file for speculation\n";
    return failure();
  }

  llvm::json::Object *jsonObject = value->getAsObject();
  if (!jsonObject) {
    llvm::errs() << "Expected a JSON object in the kernel information file for "
                    "speculation\n";
    return failure();
  }

  SmallVector<unsigned> loopBBsVec;

  std::optional<int64_t> headBB = jsonObject->getInteger("spec-head-bb");
  if (headBB) {
    loopBBsVec.push_back(static_cast<unsigned>(headBB.value()));
  }

  std::optional<int64_t> tailBB = jsonObject->getInteger("spec-tail-bb");
  if (tailBB) {
    loopBBsVec.push_back(static_cast<unsigned>(tailBB.value()));
  }

  llvm::json::Array *loopBBs = jsonObject->getArray("spec-loop-bbs");
  if (loopBBs) {
    for (const auto &loopBB : *loopBBs) {
      std::optional<int64_t> loopBBInt = loopBB.getAsInteger();
      if (!loopBBInt) {
        llvm::errs() << "Expected 'spec-loop-bbs' to contain integers\n";
        return failure();
      }
      loopBBsVec.push_back(static_cast<unsigned>(loopBBInt.value()));
    }
  }

  return SpecSpecification{loopBBsVec};
}
