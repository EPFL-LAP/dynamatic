#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "polly/RegisterPasses.h"

#include "MemDepAnalysis.h" // Include your pass header

using namespace llvm;

int main(int argc, char **argv) {
  InitLLVM x(argc, argv);
  // cl::ParseCommandLineOptions(argc, argv, "My custom opt tool\n");

  LLVMContext context;
  SMDiagnostic err;
  auto m = parseIRFile(argv[1], err, context);
  if (!m) {
    err.print(argv[0], errs());
    return 1;
  }

  PassBuilder pb;
  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;

  pb.registerModuleAnalyses(MAM);
  pb.registerFunctionAnalyses(FAM);
  pb.registerLoopAnalyses(LAM);
  pb.registerCGSCCAnalyses(CGAM);
  pb.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // polly::registerPollyPasses(pb);

  // Build pipeline
  ModulePassManager mpm;
  FunctionPassManager fpm;

  // Add your custom pass
  FAM.registerPass([]() { return polly::ScopAnalysis(); });
  FAM.registerPass([]() { return polly::ScopInfoAnalysis(); });
  fpm.addPass(MemDepAnalysisPass());

  mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));

  // Run pipeline
  mpm.run(*m, MAM);

  // Print transformed IR
  m->print(outs(), nullptr);

  return 0;
}
