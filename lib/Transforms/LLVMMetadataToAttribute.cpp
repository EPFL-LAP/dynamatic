#include "dynamatic/Transforms/LLVMMetadataToAttribute.h"

#include "dynamatic/Conversion/LLVMToControlFlow.h"
#include "dynamatic/Support/LLVM.h"
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <optional>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

// Boilerplate: Include this for the pass option defintitions
namespace dynamatic {
// import auto-generated base class definition LLVMMetadataToAttributeBase and
// put it under the dynamatic namespace.
#define GEN_PASS_DEF_LLVMMETADATATOATTRIBUTE
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic

struct LoadStoreData {
  // LLVM instruction
  Instruction *llvmInstr;

  // Metadata that holds the ID of the operation
  int32_t id;

  // The IDs of the destinations of the dependency edges.
  std::vector<int32_t> destinations;
};

namespace {
struct LLVMMetadataToAttributePass
    : public dynamatic::impl::LLVMMetadataToAttributeBase<
          LLVMMetadataToAttributePass> {

  /// \note: Use the auto-generated construtors from tblgen
  using LLVMMetadataToAttributeBase::LLVMMetadataToAttributeBase;
  void runOnOperation() override;
};
} // namespace

unsigned getMemId(Instruction *instr) {
  auto *nameMetaData = instr->getMetadata("mem.op");
  assert(nameMetaData);
  llvm::Metadata *data =
      llvm::dyn_cast<llvm::MDString>(nameMetaData->getOperand(0));
  assert(data);
  llvm::MDString *strData = llvm::dyn_cast<llvm::MDString>(data);
  assert(strData);
  // It's a metadata string
  LoadStoreData item;
  return std::stoi(strData->getString().str());
}

std::vector<int32_t> getDestOps(Instruction *instr) {
  auto *nameMetaData = instr->getMetadata("dest.ops");

  if (!nameMetaData) {
    return {};
  }

  std::vector<int32_t> destOps;

  for (int i = 0; i < nameMetaData->getNumOperands(); i++) {

    llvm::Metadata *op = nameMetaData->getOperand(i);

    if (llvm::MDString *mds = llvm::dyn_cast<llvm::MDString>(op)) {
      // It's a metadata string
      destOps.push_back(std::stoi(mds->getString().str()));
    }
  }

  return destOps;
}

std::vector<LoadStoreData>
retriveLoadStoreAnalysisDataFromMetaData(Function &f) {
  std::vector<LoadStoreData> items;

  for (llvm::BasicBlock &bb : f) {
    for (llvm::Instruction &instr : bb) {
      if (llvm::LoadInst *loadInstr = llvm::dyn_cast<llvm::LoadInst>(&instr)) {
        LoadStoreData item;
        item.llvmInstr = &instr;
        item.id = getMemId(loadInstr);
        item.destinations = getDestOps(loadInstr);
        items.push_back(item);
      } else if (llvm::StoreInst *storeInstr =
                     llvm::dyn_cast<llvm::StoreInst>(&instr)) {
        LoadStoreData item;
        item.llvmInstr = &instr;
        item.id = getMemId(storeInstr);
        item.destinations = getDestOps(storeInstr);
        items.push_back(item);
      }
    }
  }

  return items;
}

void LLVMMetadataToAttributePass::runOnOperation() {

  LLVMContext llvmCtx;
  SMDiagnostic err;

  std::unique_ptr<Module> llvmModule =
      parseAssemblyFile(StringRef(llvmir), err, llvmCtx);
  if (!llvmModule) {
    llvm::errs() << "Failed to parse LLVM IR: " << llvmir << "\n";
    signalPassFailure();
  }

  std::map<std::string, std::vector<LoadStoreData>> data;
  for (Function &f : llvmModule->functions()) {
    data.emplace(f.getName(), retriveLoadStoreAnalysisDataFromMetaData(f));
  }

  MLIRContext *ctx = &getContext();
  ModuleOp modOp = llvm::dyn_cast<ModuleOp>(getOperation());
  OpBuilder builder(ctx);

  // note:
  // - llvm::make_early_inc_range: safety iterate through an iterator when there
  // might be changes to them.
  for (auto func : modOp.getOps<LLVM::LLVMFuncOp>()) {
    assert(func->use_empty());

    auto loadStoreData = data[func.getName().str()];

    std::vector<Operation *> loadStoreOperations;

    for (auto &regions : func->getRegions()) {
      for (auto &block : regions.getBlocks()) {
        for (auto &op : block.getOperations()) {
          if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
            loadStoreOperations.push_back(&op);
          } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
            loadStoreOperations.push_back(&op);
          }
        }
      }
    }
    assert(loadStoreData.size() == loadStoreOperations.size());

    for (auto [op, data] :
         llvm::zip_equal(loadStoreOperations, loadStoreData)) {
      op->setAttr(StringRef("mem.op"), builder.getI32IntegerAttr(data.id));
      op->setAttr(StringRef("mem.dest"),
                  builder.getI32ArrayAttr(ArrayRef(data.destinations)));
    }
  }
}