//===- ExportCFG.cpp - Export CFG of std-level function ---------*- C++ -*-===//
//
// This file contains the implementation of the export to CFG pass. It
// produces a .dot file (in the DOT language) parsable by Graphviz and
// containing the graph representation of a function's CFG. It only supports one
// function at the moment and will fail if there is more than one func.func
// operation in the module.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/ExportCFG.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dynamatic;

namespace {

/// Driver for the ExportCFG pass.
struct ExportCFGPass : public ExportCFGBase<ExportCFGPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    markAllAnalysesPreserved();

    // We support at most one function per module
    auto funcs = mod.getOps<func::FuncOp>();
    if (funcs.empty())
      return;
    if (++funcs.begin() != funcs.end()) {
      mod->emitOpError()
          << "we currently only support one standard function per module";
      return signalPassFailure();
    }
    func::FuncOp funcOp = *funcs.begin();

    // Create the file to store the graph
    std::error_code ec;
    llvm::raw_fd_ostream outfile(funcOp.getNameAttr().str() + "_bbgraph.dot",
                                 ec);
    mlir::raw_indented_ostream os(outfile);

    // Print the graph
    os << "Digraph G {\n";
    os.indent();
    os << "splines=spline;\n";
    printFuncCFG(os, funcOp);
    os.unindent();
    os << "}\n";

    outfile.close();
  };

private:
  /// Prints the given function's CFG to the output stream. First prints a node
  /// for each block in the function, then an edge for each
  /// predecessor-successor pair in the function's CFG.
  void printFuncCFG(mlir::raw_indented_ostream &os, func::FuncOp funcOp);
};

} // namespace

void ExportCFGPass::printFuncCFG(mlir::raw_indented_ostream &os,
                                 func::FuncOp funcOp) {

  DenseMap<Block *, std::string> blockNames;
  unsigned blockIdx = 1;

  // Assign a unique name to each block and create a corresponding node for each
  // in the DOT
  for (auto &block : funcOp.getBody()) {
    auto name = "block" + std::to_string(blockIdx++);
    blockNames[&block] = name;
    os << "\"" << name << "\" [shape=box]\n";
  }

  // Determine the successors of each block and create a corresponding edge per
  // successot in the DOT
  for (auto &[block, name] : blockNames)
    for (auto *succ : block->getSuccessors())
      os << "\"" << name << "\" -> \"" << blockNames[succ] << "\" [freq = 0]\n";
}

namespace dynamatic {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportCFGPass() {
  return std::make_unique<ExportCFGPass>();
}
} // namespace dynamatic
