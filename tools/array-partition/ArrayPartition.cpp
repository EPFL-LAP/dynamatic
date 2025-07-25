#include "polly/DependenceInfo.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include <stdexcept>
#include <stdlib.h>
#include <utility>

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/MemoryDependency.h"

using namespace llvm;
using namespace polly;

namespace {

/// \brief: an LLVM pass that combines polyhedral and alias analysis to compute
/// a set of dependency edges from the LLVM IR. It further uses dataflow
/// analysis to eliminate dependency edges enforced by the dataflow.
struct ArrayPartition : PassInfoMixin<ArrayPartition> {

  unsigned memCount = 0;

  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);
};

PreservedAnalyses ArrayPartition::run(Function &f,
                                      FunctionAnalysisManager &fam) {

  llvm::LLVMContext &ctx = f.getContext();

  auto &regionInfoAnalysis = fam.getResult<RegionInfoAnalysis>(f);

  auto &scopInfoAnalysis = fam.getResult<ScopInfoAnalysis>(f);

  auto &loopAnalysis = fam.getResult<LoopAnalysis>(f);
  llvm::errs() << "Hello!\n";

  return PreservedAnalyses::all();
}

} // end anonymous namespace

// Register the pass for opt-style loading
// Important note: you need to enable shared libarary in LLVM to load pass
// plugin:
// https://stackoverflow.com/questions/51474188/using-shared-object-so-by-command-opt-in-llvm
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ArrayPartition", LLVM_VERSION_STRING,
          [](PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](StringRef name, FunctionPassManager &fpm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "array-partition") {
                    fpm.addPass(ArrayPartition());
                    return true;
                  }
                  return false;
                });
          }};
}
