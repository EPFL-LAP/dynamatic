// Create an adapter module from a handshake-level mlir that is the same as the
// interface of Vitis HLS.
//
// For example:
// int fir(int d_i[N], int idx[N])
//
// Will be converted to the following interface:
//
// module fir(
//   clk,
//   rst,
//   ap_start,
//   ap_done,
//   ap_idle,
//   ap_ready,
//   d_i_address0,
//   d_i_ce0,
//   d_i_q0,
//   idx_address0,
//   idx_ce0,
//   idx_q0,
//   ap_return
// );

#include "dynamatic/Conversion/HandshakeToHW.h"
#include "dynamatic/Dialect/HW/HWDialect.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

std::string formatSignalWidth(Type type) {

  if (auto channelType = dyn_cast<ChannelType>(type)) {
    int width = channelType.getDataBitWidth();
    if (width == 1) {
      return "";
    }
    return "[" + std::to_string(width - 1) + ":0]";
  }

  if (auto intType = dyn_cast<IntegerType>(type)) {
    int width = intType.getWidth();
    if (width == 1) {
      return "";
    }
    return "[" + std::to_string(width - 1) + ":0]";
  }
  return "";
}

void createModuleInterface(handshake::FuncOp funcOp, raw_indented_ostream &os) {

  os << "input clk,\n";
  os << "input rst,\n";

  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    if (isa<handshake::ChannelType>(arg.getType())) {
      os << ",\n";
      os << "input " << formatSignalWidth(arg.getType())
         << portAttr.dyn_cast<StringAttr>().data();
    }
  }

  os << "input go\n";

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp.getResultTypes(), funcOp.getResNames())) {
    if (isa<handshake::ChannelType>(resType)) {
      os << ",\n";
      os << "output " << formatSignalWidth(resType)
         << portAttr.dyn_cast<StringAttr>().data();
    }
  }
  os << ",\n";
  os << "output done\n";
}

void createAdapter(handshake::FuncOp funcOp, raw_indented_ostream &os) {
  os << "module " << funcOp->getName() << "(\n";
  createModuleInterface(funcOp, os);
  os << ")\n";
  os.indent();

  os.unindent();
  os << "endmodule\n";
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv, "Create a rigid wrapper for a given HW module.\n");

  // We only need the Handshake and HW dialects
  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect>();

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    return 1;
  }

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  mlir::raw_indented_ostream os(llvm::outs());

  // Write each module's RTL implementation to a separate file
  for (handshake::FuncOp funcOp : modOp->getOps<handshake::FuncOp>()) {
    createAdapter(funcOp, os);
    return 0;
  }
}
