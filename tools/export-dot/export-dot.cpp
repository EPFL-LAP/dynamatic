//===- export-dot.cpp - Export Handshake-level IR to DOT --------*- C++ -*-===//
//
// This file implements the export-dot tool, which outputs on stdout the
// Graphviz-formatted representation on an input Handshake-level IR. The tool
// may be configured so that its output is compatible with .dot files expected
// by legacy Dynamatic, assuming the the inpur IR respects some constraints
// imposed in legacy dataflow circuits. This tools enables the creation of a
// bridge between Dynamatic++ and legacy Dynamatic, which is very useful in
// practice.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/DOTPrinter.h"
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

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<bool> legacy(
    "legacy", cl::Optional,
    cl::desc("If true, the exported DOT file will be made compatible with "
             "legacy Dynamatic (i.e. the file will be parsable by legacy "
             "Dynamatic's post-processing passes). Enabling this option adds "
             "some constraints on the input IR, which must also be compatible "
             "with legacy Dynamatic's elastic circuits."),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> dotDebug(
    "dot-debug", cl::Optional,
    cl::desc(
        "If fase, the exported DOT file will be pretty-printed, making its "
        "visualization easier to look at but harder to debug with. For "
        "example, node names will be shortened (and not uniqued) when "
        "pretty-printing."),
    cl::init(false), cl::cat(mainCategory));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a DOT graph corresponding to the module for visualization\n"
      "and legacy-compatibility purposes.The pass only supports exporting\n"
      "the graph of a single Handshake function at the moment, and will fail\n"
      "if there is more than one Handhsake function in the module.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er) level
  // dialects or parsers. Allow unregistered dialects to not fail in these
  // cases
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
    return 1;

  dynamatic::DOTPrinter printer(legacy, dotDebug);
  return failed(printer.printDOT(*module));
}
