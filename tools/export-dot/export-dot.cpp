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
#include "dynamatic/Support/TimingModels.h"
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

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> timingDBFilepath(
    "timing-models", cl::Optional,
    cl::desc(
        "Relative path to JSON-formatted file containing timing models for "
        "dataflow components. The tool only tries to read from this file if it "
        "is ran in one of the legacy-compatible modes, where timing "
        "annotations are given to all nodes in the graph. By default, contains "
        "the relative path (from the project's top-level directory) to the "
        "file defining the default timing models in Dynamatic++."),
    cl::init("data/components.json"), cl::cat(mainCategory));

static cl::opt<std::string> modeArg(
    "mode", cl::Optional,
    cl::desc(
        "Mode in which to run the export tool. This allows to control the "
        "structure of the resulting DOT. In particular, the tool's output "
        "can be made compatible with some of legacy Dynamatic's tools, at "
        "the cost of added constraints on the input IR (which may be "
        "\"too general\" for legacy Dynamatic). Available modes are:\n"
        "\t- visual (cleanest to look at, default)\n"
        "\t- legacy (compatible with legacy dot2vhdl tool)\n"
        "\t- legacy-buffers (compatible with legacy Buffers/dot2vhdl tools)"),
    cl::init("visual"), cl::cat(mainCategory));

static cl::opt<std::string> edgeStyleArg(
    "edge-style", cl::Optional,
    cl::desc("Style in which to render edges in the resulting DOTs (this is "
             "essentially the 'splines' attribute of the top-level DOT graph). "
             "Available options are:\n"
             "\t- spline (default)\n"
             "\t- ortho (orthogonal polylines)\n"),
    cl::init("spline"), cl::cat(mainCategory));

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
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
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

  // Decode the "mode" argument
  DOTPrinter::Mode mode;
  if (modeArg == "visual")
    mode = DOTPrinter::Mode::VISUAL;
  else if (modeArg == "legacy")
    mode = DOTPrinter::Mode::LEGACY;
  else if (modeArg == "legacy-buffers")
    mode = DOTPrinter::Mode::LEGACY_BUFFERS;
  else {
    llvm::errs() << "Unkwown mode \"" << modeArg
                 << "\" provided, must be one of \"visual\", \"legacy\", "
                    "\"legacy-buffers\".\n";
    return 1;
  }

  // Decode the "edgeStyle" argument
  DOTPrinter::EdgeStyle edgeStyle;
  if (edgeStyleArg == "spline")
    edgeStyle = DOTPrinter::EdgeStyle::SPLINE;
  else if (edgeStyleArg == "ortho")
    edgeStyle = DOTPrinter::EdgeStyle::ORTHO;
  else {
    llvm::errs() << "Unkwown edge style \"" << edgeStyleArg
                 << "\" provided, must be one of \"spline\", \"ortho\".\n";
    return 1;
  }

  bool legacyCompMode = mode == DOTPrinter::Mode::LEGACY ||
                        mode == DOTPrinter::Mode::LEGACY_BUFFERS;
  if (legacyCompMode) {
    // In legacy mode, read timing models for dataflow components from a
    // JSON-formatted database
    TimingDatabase timingDB(&context);
    if (failed(TimingDatabase::readFromJSON(timingDBFilepath, timingDB))) {
      llvm::errs() << "Failed to read timing database at \"" << timingDBFilepath
                   << "\"\n";
      return 1;
    }
    DOTPrinter printer(mode, edgeStyle, &timingDB);
    return failed(printer.printDOT(*module));
  }

  DOTPrinter printer(mode, edgeStyle);
  return failed(printer.printDOT(*module));
}
