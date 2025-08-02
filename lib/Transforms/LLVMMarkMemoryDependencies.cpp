#include "dynamatic/Transforms/LLVMMarkMemoryDependencies.h"

#include "dynamatic/Conversion/LLVMToControlFlow.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/MemoryDependency.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

#include "dynamatic/Analysis/NameAnalysis.h"

#include "dynamatic/Support/MemoryDependency.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

// Boilerplate: Include this for the pass option defintitions
namespace dynamatic {
// import auto-generated base class definition LLVMMarkMemoryDependenciesBase
// and put it under the dynamatic namespace.
#define GEN_PASS_DEF_LLVMMARKMEMORYDEPENDENCIES
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic

namespace {
// Metadata unseralized from the memory dependency analysis pass
struct LLVMMarkMemoryDependenciesPass
    : public dynamatic::impl::LLVMMarkMemoryDependenciesBase<
          LLVMMarkMemoryDependenciesPass> {

  /// \note: Use the auto-generated construtors from tblgen
  using LLVMMarkMemoryDependenciesBase::LLVMMarkMemoryDependenciesBase;
  void runOnOperation() override;
};

/// \brief: Visit all the instructions in an LLVM IR and extract the dependency
/// information from all load and store operations.
std::vector<LLVMMemDependency>
retriveLoadStoreAnalysisDataFromMetaData(Function &f) {
  std::vector<LLVMMemDependency> items;

  for (llvm::BasicBlock &bb : f) {
    for (llvm::Instruction &instr : bb) {
      if (llvm::LoadInst *loadInstr = llvm::dyn_cast<llvm::LoadInst>(&instr)) {
        std::optional<LLVMMemDependency> dep =
            LLVMMemDependency::fromLLVMInstruction(loadInstr);
        if (dep.has_value()) {
          items.push_back(dep.value());
        }
      } else if (llvm::StoreInst *storeInstr =
                     llvm::dyn_cast<llvm::StoreInst>(&instr)) {
        std::optional<LLVMMemDependency> dep =
            LLVMMemDependency::fromLLVMInstruction(storeInstr);
        if (dep.has_value()) {
          items.push_back(dep.value());
        }
      }
    }
  }
  return items;
}

/// \brief: performs two actions:
/// - Propagate the unique handshake names for loads and stores from LLVM IR to
/// MLIR.
/// - mark the RAW and WAW dependency specified by MemoryDependency on all
/// memory operations.
LogicalResult markMemoryDependency(LLVM::LLVMFuncOp funcOp,
                                   std::vector<LLVMMemDependency> loadStoreData,
                                   OpBuilder &builder, MLIRContext &ctx) {

  // Following the visiting order in the funcOp, collect all the load store ops.
  std::vector<Operation *> loadStoreOperations;
  for (auto &regions : funcOp->getRegions()) {
    for (auto &block : regions.getBlocks()) {
      for (auto &op : block.getOperations()) {
        if (isa<LLVM::LoadOp>(op) || isa<LLVM::StoreOp>(op)) {
          loadStoreOperations.push_back(&op);
        }
      }
    }
  }

  // NOTE: we should definitely add more sanity checks in this pass.
  if (loadStoreData.size() != loadStoreOperations.size()) {
    funcOp.emitError("The number of loads and stores in the MLIR LLVM dialect "
                     "does not match the number in the LLVM IR!");
    return failure();
  }

  for (auto [op, data] : llvm::zip_equal(loadStoreOperations, loadStoreData)) {
    // Assign the operations using the natual handshake names. TODO: maybe we
    // separate the naming for loads and stores?
    std::string opName = data.name;
    SmallVector<dynamatic::handshake::MemDependenceAttr> deps =
        data.getMemoryDependenceAttrs(ctx);
    op->setAttr(StringRef(NameAnalysis::ATTR_NAME),
                builder.getStringAttr(opName));
    if (!deps.empty()) {
      setDialectAttr<dynamatic::handshake::MemDependenceArrayAttr>(op, &ctx,
                                                                   deps);
    }
  }
  return success();
}

void LLVMMarkMemoryDependenciesPass::runOnOperation() {
  LLVMContext llvmCtx;
  SMDiagnostic err;
  std::unique_ptr<Module> llvmModule =
      parseAssemblyFile(StringRef(llvmir), err, llvmCtx);
  if (!llvmModule) {
    llvm::errs() << "Failed to parse LLVM IR: " << llvmir << "\n";
    signalPassFailure();
  }

  std::map<std::string, std::vector<LLVMMemDependency>> data;
  for (Function &f : llvmModule->functions()) {
    data.emplace(f.getName(), retriveLoadStoreAnalysisDataFromMetaData(f));
  }

  MLIRContext *ctx = &getContext();
  ModuleOp modOp = llvm::dyn_cast<ModuleOp>(getOperation());
  OpBuilder builder(ctx);

  for (auto func : modOp.getOps<LLVM::LLVMFuncOp>()) {
    assert(func->use_empty());
    auto loadStoreData = data[func.getName().str()];
    if (failed(markMemoryDependency(func, loadStoreData, builder, *ctx))) {
      llvm::errs()
          << "Failed to name memory operations and mark dependency edges!\n";
      signalPassFailure();
    }
  }
}

} // namespace