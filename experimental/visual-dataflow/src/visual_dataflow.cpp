#include "visual_dataflow.h"
#include "dynamatic/Support/DOTPrinter.h"
#include "dynamatic/Support/TimingModels.h"
#include "godot_cpp/core/class_db.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace godot;

void VisualDataflow::_bind_methods() {
  ClassDB::bind_method(D_METHOD("getNumber"), &VisualDataflow::getNumber);
}

VisualDataflow::VisualDataflow() {
  timePassed = 0.0;

  std::string filename = "test.mlir";
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(filename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    // llvm::errs() << "could not open input file '" << filename
    //              << "': " << error.message() << "\n";
    return;
  }

  MLIRContext context;
  context.loadDialect<func::FuncDialect, memref::MemRefDialect,
                      arith::ArithDialect, LLVM::LLVMDialect,
                      handshake::HandshakeDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!module)
    return;
}

unsigned VisualDataflow::getNumber() { return 100; }

void VisualDataflow::_process(double delta) {
  timePassed += delta;
  Vector2 newPos = Vector2(10.0 + (10.0 * sin(timePassed * 2.0)),
                           10.0 + (10.0 * cos(timePassed * 1.5)));
  set_position(newPos);
}
