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

#include "VerilogModule.h"
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <memory>
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

constexpr int clog2(uint32_t x) {
  return (x <= 1) ? 0 : 1 + clog2((x + 1) >> 1);
}

// This creates the module declaration
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
void createModuleInterface(VerilogModule &module, handshake::FuncOp funcOp,
                           raw_indented_ostream &os) {

  module.port("rst", 1, false);
  module.port("clk", 1, false);
  module.port("go", 1, false);
  module.port("done", 1, true);

  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      module.port(portAttr.dyn_cast<StringAttr>().data(),
                  type.getDataBitWidth(), false);
    } else if (mlir::MemRefType type =
                   dyn_cast<mlir::MemRefType>(arg.getType())) {

      std::string signalName = portAttr.dyn_cast<StringAttr>().str();

      unsigned addressWidth = clog2(type.getNumElements());
      unsigned dataWidth = type.getElementTypeBitWidth();
      module.port(signalName + "_loadAddr", addressWidth, true);
      module.port(signalName + "_loadEn", 1, true);
      module.port(signalName + "_storeEn", 1, true);
      module.port(signalName + "_storeAddr", addressWidth, true);
      module.port(signalName + "_storeData", dataWidth, true);
      module.port(signalName + "_loadData", dataWidth, false);
    }
  }

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp.getResultTypes(), funcOp.getResNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(resType)) {
      module.port(signalName, type.getDataBitWidth(), true);
    }
  }
}

void createInternalSignals(VerilogModule &module, handshake::FuncOp funcOp,
                           raw_indented_ostream &os) {

  // Storing the current state of the module
  // (kernel_state = 0) => idle state
  // (kernel_state = 1) => working state
  module.reg("kernel_state", 2);

  // State registers
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();

    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      // Storing the input data (in case the function is not ready to take all
      // arguments at once).
      module.reg(signalName + "_reg", type.getDataBitWidth());
      // Indicates if the input data has been taken by the handshake function.
      module.reg(signalName + "_taken", type.getDataBitWidth());

      auto reset = std::make_unique<IfElseBlock>("rst");
      reset
          ->addIf(std::make_unique<NonBlockingAssign>(signalName + "_reg", "0"))
          .addElse(std::make_unique<NonBlockingAssign>(signalName + "_reg",
                                                       signalName));

      module.always(std::move(
          AlwaysBlock("posedge clk").add(dyn_cast<Statement>(reset))));
    }
  }

  // The handshake signals are internally generated:
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      module.wire(signalName + "_valid", 1);
      module.wire(signalName + "_ready", 1);

    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(arg.getType())) {
      module.wire(signalName + "_valid", 1);
      module.wire(signalName + "_ready", 1);
    }
  }
}

void createInstantiation(VerilogModule &module, handshake::FuncOp funcOp,
                         raw_indented_ostream &os) {
  Instance inst(funcOp.getName().str(), funcOp.getName().str() + "_inst");
  inst.connect("clk", "clk");
  inst.connect("rst", "rst");
  inst.connect("go", "go");
  inst.connect("done", "done");
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      inst.connect(signalName, signalName);
      inst.connect(signalName + "_valid", signalName + "_valid");
      inst.connect(signalName + "_ready", signalName + "_ready");
    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(arg.getType())) {
      inst.connect(signalName + "_valid", signalName + "_valid");
      inst.connect(signalName + "_ready", signalName + "_ready");
    } else if (mlir::IntegerType type =
                   dyn_cast<mlir::IntegerType>(arg.getType())) {
      inst.connect(signalName, signalName);
    } else if (mlir::MemRefType type =
                   dyn_cast<mlir::MemRefType>(arg.getType())) {
      inst.connect(signalName + "_loadAddr", signalName + "_loadAddr");
      inst.connect(signalName + "_loadEn", signalName + "_loadEn");
      inst.connect(signalName + "_storeEn", signalName + "_storeEn");
      inst.connect(signalName + "_storeAddr", signalName + "_storeAddr");
      inst.connect(signalName + "_storeData", signalName + "_storeData");
      inst.connect(signalName + "_loadData", signalName + "_loadData");
    }
  }

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp.getResultTypes(), funcOp.getResNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(resType)) {
      inst.connect(signalName, signalName);
      inst.connect(signalName + "_valid", signalName + "_valid");
      inst.connect(signalName + "_ready", signalName + "_ready");
    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(resType)) {
      inst.connect(signalName + "_valid", signalName + "_valid");
      inst.connect(signalName + "_ready", signalName + "_ready");
    } else if (mlir::IntegerType type = dyn_cast<mlir::IntegerType>(resType)) {
      inst.connect(signalName, signalName);
    }
  }
  module.instantiate(inst);
}

void createAdapter(handshake::FuncOp funcOp, raw_indented_ostream &os) {

  auto m = VerilogModule(funcOp.getName().str());
  createModuleInterface(m, funcOp, os);

  createInstantiation(m, funcOp, os);

  createInternalSignals(m, funcOp, os);
  m.emit(os);
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
