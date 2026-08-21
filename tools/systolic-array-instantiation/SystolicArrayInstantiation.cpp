#include "polly/DependenceInfo.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
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

#include "llvm/Analysis/ValueTracking.h"

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/MemoryDependency.h"
#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;
using namespace polly;

/// \brief: A class that represents a systolic array instance.
class SystolicArray {
public:
  SystolicArray(int rows, int cols, int weights, int sa_rows, int sa_cols)
      : numRows(rows), numCols(cols), numWeights(weights), SA_rows(sa_rows),
        SA_cols(sa_cols) {}
  int numRows;
  int numCols;
  int numWeights;
  int SA_rows;
  int SA_cols;

  void print() {
    llvm::errs() << "Systolic Array Instance:\n";
    llvm::errs() << "Matrix Rows: " << numRows << "\n";
    llvm::errs() << "Matrix Columns: " << numCols << "\n";
    llvm::errs() << "Matrix Weights: " << numWeights << "\n";
    llvm::errs() << "Systolic Array Rows: " << SA_rows << "\n";
    llvm::errs() << "Systolic Array Columns: " << SA_cols << "\n";
  }
};

Value *findMatrixDef(Value *arg) {
  Value *def_arg = arg->stripPointerCasts();
  // Base cases
  if (isa<AllocaInst>(def_arg) || isa<GlobalVariable>(def_arg))
    return def_arg;

  // Case 1: Value is a function argument — need to look at call sites
  if (auto *arg = dyn_cast<Argument>(def_arg)) {
    Function *callee = arg->getParent();
    unsigned argIndex = arg->getArgNo();

    // Iterate over all uses of this function
    for (User *U : callee->users()) {
      if (auto *call = dyn_cast<CallBase>(U)) {
        if (call->getCalledFunction() == callee) {
          Value *actualArg = call->getArgOperand(argIndex);
          Value *origin = findMatrixDef(actualArg);
          if (origin)
            return origin; // return first found origin
        }
      }
    }
    return nullptr;
  }

  // Case 2: GEP or BitCast — step backwards
  if (auto *gep = dyn_cast<GetElementPtrInst>(def_arg))
    return findMatrixDef(gep->getPointerOperand());

  if (auto *bc = dyn_cast<BitCastInst>(def_arg))
    return findMatrixDef(bc->getOperand(0));

  return nullptr; // Unknown source
}

/// \brief: Extracts the dimensions of a matrix from the LLVM Value
/// representing the matrix. It assumes a 2D array type.
void getMatrixDims(Value *arg, int &rows, int &cols) {
  Value *def = findMatrixDef(arg);
  if (!def) {
    llvm::errs() << "Error: Could not find definition of matrix\n";
    exit(1);
  }
  // Assert it is an alloca
  if (auto *alloca = dyn_cast<AllocaInst>(def)) {
    Type *ty = alloca->getAllocatedType();
    if (auto *arrayTy = dyn_cast<ArrayType>(ty)) {
      if (auto *innerArrayTy = dyn_cast<ArrayType>(arrayTy->getElementType())) {
        rows = arrayTy->getNumElements();
        cols = innerArrayTy->getNumElements();
        return;
      }
    }
  }
  llvm::errs() << "Error: Matrix is not a 2D array type\n";
  exit(1);
}

/// \brief: Generates the systolic array instance by replacing the function
/// call to fpsa_mat_mul with systolic array instance.
SystolicArray generateFromFuncCall(Function &f) {
  // Extract the matrix sizes and systolic array sizes from the function call
  // fpsa_mat_mul
  int mat_rows = -1, mat_cols = -1, mat_weights = -1;
  std::string A_name, B_name, C_name;
  int SA_ROWS = -1, SA_COLS = -1;
  for (auto &bb : f) {
    for (auto &inst : bb) {
      if (auto *call = dyn_cast<CallInst>(&inst)) {
        if (call->getCalledFunction()->getName() == "fpsa_mat_mul") {
          // Get the sizes of the input matrices
          auto *argA = call->getArgOperand(0);
          auto *argB = call->getArgOperand(1);
          auto *argC = call->getArgOperand(2);
          int rowsA, colsA, rowsB, colsB, rowsC, colsC;
          getMatrixDims(argA, rowsA, colsA);
          getMatrixDims(argB, rowsB, colsB);
          getMatrixDims(argC, rowsC, colsC);
          if (colsA != rowsB || rowsA != rowsC || colsB != colsC) {
            llvm::errs() << "Error: Incompatible matrix dimensions\n";
            exit(1);
          }
          mat_rows = rowsA;
          mat_cols = colsB;
          mat_weights = colsA;

          if (auto *constSA_ROWS =
                  dyn_cast<ConstantInt>(call->getArgOperand(3))) {
            SA_ROWS = constSA_ROWS->getSExtValue();
          } else {
            llvm::errs() << "Error: SA_ROWS is not a constant integer\n";
            exit(1);
          }
          if (auto *constSA_COLS =
                  dyn_cast<ConstantInt>(call->getArgOperand(4))) {
            SA_COLS = constSA_COLS->getSExtValue();
          } else {
            llvm::errs() << "Error: SA_COLS is not a constant integer\n";
            exit(1);
          }
        }
      }
    }
  }
  if (SA_ROWS == -1 || SA_COLS == -1 || mat_rows == -1 || mat_cols == -1 ||
      mat_weights == -1) {
    llvm::errs()
        << "Error: Could not extract matrix sizes or SA size from call\n";
    exit(1);
  }
  return SystolicArray(mat_rows, mat_cols, mat_weights, SA_ROWS, SA_COLS);
}

/// \brief: Function to extract the arguments of the top function
/// reserved for the systolic array
void extractTopFuncArgs(Function &f, Value *&data_load, Value *&data_store,
                        Value *&addr_load, Value *&addr_store) {
  data_load = nullptr;
  data_store = nullptr;
  addr_load = nullptr;
  addr_store = nullptr;
  // Extract the arguments by name
  // We assume the top function has arguments named:
  for (auto &arg : f.args()) {
    if (arg.getName() == "data_load")
      data_load = &arg;
    else if (arg.getName() == "data_store")
      data_store = &arg;
    else if (arg.getName() == "addr_load")
      addr_load = &arg;
    else if (arg.getName() == "addr_store")
      addr_store = &arg;
  }
  if (!data_load || !data_store || !addr_load || !addr_store) {
    llvm::errs() << "Error: Could not find all top function arguments\n";
    exit(1);
  }
}

/// \brief: Function to generate the function in the LLVM IR that
/// represents output signals
Function *generateOutputFunc(Module &M, Function &f) {
  FunctionType *NewFuncTy =
      FunctionType::get(Type::getInt32Ty(f.getContext()), false);
  Function *NewFunc =
      Function::Create(NewFuncTy, Function::ExternalLinkage, "__init1", &M);
  return NewFunc;
}

/// \brief: Function to generate the function in the LLVM IR that
/// represents a demux
Function *generateDemuxFunc(Module &M, Function &f, int num_outputs) {
  std::vector<Type *> ArgTypes;
  ArgTypes.push_back(Type::getInt32Ty(f.getContext())); // input
  for (int i = 1; i <= num_outputs; ++i) {
    ArgTypes.push_back(Type::getInt32Ty(f.getContext())); // output i
  }
  FunctionType *NewFuncTy =
      FunctionType::get(Type::getInt32Ty(f.getContext()), ArgTypes, false);
  Function *NewFunc =
      Function::Create(NewFuncTy, Function::ExternalLinkage, "demux", &M);
  unsigned idx = 0;
  for (auto &Arg : NewFunc->args()) {
    if (idx == 0)
      Arg.setName("input_demux");
    else
      Arg.setName("output_id_" + std::to_string(idx));
    ++idx;
  }
  return NewFunc;
}

/// \brief: Function to generate the function in the LLVM IR that
/// represents an FPSA unit
Function *generateFPSAFunc(Module &M, Function &f) {
  std::vector<Type *> ArgTypes;
  ArgTypes.push_back(Type::getInt32Ty(f.getContext())); // input north
  ArgTypes.push_back(Type::getInt32Ty(f.getContext())); // input west
  ArgTypes.push_back(Type::getInt32Ty(f.getContext())); // output south
  ArgTypes.push_back(Type::getInt32Ty(f.getContext())); // output east
  FunctionType *NewFuncTy =
      FunctionType::get(Type::getInt32Ty(f.getContext()), ArgTypes, false);
  Function *NewFunc =
      Function::Create(NewFuncTy, Function::ExternalLinkage, "fpsa_unit", &M);
  unsigned idx = 0;
  for (auto &Arg : NewFunc->args()) {
    if (idx == 0)
      Arg.setName("input_north");
    else if (idx == 1)
      Arg.setName("input_west");
    else if (idx == 2)
      Arg.setName("output_south");
    else if (idx == 3)
      Arg.setName("output_east");
    ++idx;
  }
  return NewFunc;
}

namespace {

/// \brief: an LLVM pass that generates systolic array instances starting
/// from a matrix multiplication function in the LLVM IR.
struct SystolicArrayInstantiationPass
    : PassInfoMixin<SystolicArrayInstantiationPass> {
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);
};

// Main function to run the pass
PreservedAnalyses
SystolicArrayInstantiationPass::run(Function &f, FunctionAnalysisManager &fam) {

  int FPSA_N = 2;
  int FPSA_K = 1;

  // If the function name is main, skip it
  // We assume that main only calls the kernel function
  if (f.getName() == "main") {
    return PreservedAnalyses::all();
  }

  // Check if there is any called function inside this function
  // that matches string name "fpsa_mat_mul"
  bool hasFpsaCall = false;
  for (auto &bb : f) {
    for (auto &inst : bb) {
      if (auto *call = dyn_cast<CallInst>(&inst)) {
        if (call->getCalledFunction()->getName() == "fpsa_mat_mul") {
          if (hasFpsaCall) {
            llvm::errs() << "Error: Multiple calls to fpsa_mat_mul in the same "
                            "function are not yet supported\n";
            exit(1);
          }
          hasFpsaCall = true;
        }
      }
    }
  }
  if (!hasFpsaCall) {
    // If there is no call to fpsa_mat_mul, skip this function
    return PreservedAnalyses::all();
  }

  // If we reach here, it means that the function contains a call to
  // fpsa_mat_mul
  // We can replace the call with multiple systolic array instances
  // First extract information related to input matrices and Systolic Array
  // size
  SystolicArray sa = generateFromFuncCall(f);
  sa.print();

  // Find the BB in which the call to fpsa_mat_mul is located
  BasicBlock *callBB = nullptr;
  for (auto &bb : f) {
    for (auto &inst : bb) {
      if (auto *call = dyn_cast<CallInst>(&inst)) {
        if (call->getCalledFunction()->getName() == "fpsa_mat_mul") {
          callBB = &bb;
        }
      }
    }
  }

  // Find the argument of function f with name conventions
  // "data_load", "data_store", "addr_load", "addr_store"
  Value *data_load, *data_store, *addr_load, *addr_store;
  extractTopFuncArgs(f, data_load, data_store, addr_load, addr_store);

  int rows_per_su = sa.numRows / sa.SA_rows;
  int cols_per_su = sa.numCols / sa.SA_cols;

  int FPSA_rows = sa.SA_rows / (FPSA_N * FPSA_K);
  int FPSA_cols = sa.SA_cols / (FPSA_N * FPSA_K);
  int n_demux_outputs = FPSA_rows + FPSA_cols;

  // Get module from function
  Module &M = *f.getParent();
  // Creating the function for connecting output signals
  Function *OutputFunc = generateOutputFunc(M, f);
  // Creating the function for demux
  Function *DemuxFunc = generateDemuxFunc(M, f, n_demux_outputs);
  // Creating the function for fpsa_unit
  Function *FPSAFunc = generateFPSAFunc(M, f);
  // Create demux call
  std::vector<Value *> demux_outputs;
  // createDemuxCall(DemuxFunc, callBB, data_load, n_demux_outputs,
  // demux_outputs);

  BasicBlock &EntryBB = f.getEntryBlock();
  IRBuilder<> Builder(&EntryBB, EntryBB.begin());

  Builder.CreateCall(DemuxFunc, {Builder.getInt32(0), Builder.getInt32(0),
                                 Builder.getInt32(0), Builder.getInt32(0),
                                 Builder.getInt32(0)});

  Value *input_north = ConstantInt::get(Type::getInt32Ty(f.getContext()), 16);
  Value *input_west = ConstantInt::get(Type::getInt32Ty(f.getContext()), 32);
  Value *output_south = ConstantInt::get(Type::getInt32Ty(f.getContext()), 64);
  Value *output_east = ConstantInt::get(Type::getInt32Ty(f.getContext()), 128);

  // Builder.CreateCall(NewFunc,
  //                    {input_north, input_west, output_south, output_east});

  // We do not modify the IR
  return PreservedAnalyses::all();
}

} // namespace

// Register the pass for opt-style loading
// Important note: you need to enable shared libarary in LLVM to load pass
// plugin:
// https://stackoverflow.com/questions/51474188/using-shared-object-so-by-command-opt-in-llvm
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SystolicArrayInstantiation",
          LLVM_VERSION_STRING, [](PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](StringRef name, FunctionPassManager &fpm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "systolic-array-instantiation") {
                    fpm.addPass(SystolicArrayInstantiationPass());
                    return true;
                  }
                  return false;
                });
          }};
}
