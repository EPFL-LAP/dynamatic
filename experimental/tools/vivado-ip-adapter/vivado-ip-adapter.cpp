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

// Suffix for the names of the memory signals of the instantiated module
static const std::string INST_STORE_ADDR = ("_storeAddr");
static const std::string INST_STORE_EN = ("_storeEn");
static const std::string INST_STORE_DATA = ("_storeData");
static const std::string INST_LOAD_ADDR = ("_loadAddr");
static const std::string INST_LOAD_EN = ("_loadEn");
static const std::string INST_LOAD_DATA = ("_loadData");
// Suffix for the names of the memory signals of the interface
static const std::string ITF_STORE_ADDR = ("_address0");
static const std::string ITF_STORE_EN = ("_ce0");
static const std::string ITF_STORE_DATA = ("_din0");
static const std::string ITF_LOAD_ADDR = ("_address1");
static const std::string ITF_LOAD_EN = ("_ce1");
static const std::string ITF_LOAD_DATA = ("_dout1");

std::string STR_STD_REG = R"DELIM(
`timescale 1ns / 1ps
module standard_reg #(
  parameter DATA_TYPE = 32
) (
  clk,
  rst,
  data_in,
  enable,
  data_out
);

input clk;
input rst;
input [DATA_TYPE-1:0]data_in;
input enable;
output [DATA_TYPE-1:0]data_out;

reg [DATA_TYPE-1:0]data_reg = 0;

always @(posedge clk) begin
  if (rst) 
    data_reg <= 0;
  else begin
    if (enable) 
      data_reg <= data_in;
  end
end

assign data_out = data_reg;
endmodule)DELIM";

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
  module.port("ap_start", 1, false);
  module.port("ap_done", 1, true);

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
      module.port(signalName + ITF_LOAD_ADDR, addressWidth, true);
      module.port(signalName + ITF_LOAD_EN, 1, true);
      module.port(signalName + ITF_STORE_EN, 1, true);
      module.port(signalName + ITF_STORE_ADDR, addressWidth, true);
      module.port(signalName + ITF_STORE_DATA, dataWidth, true);
      module.port(signalName + ITF_LOAD_DATA, dataWidth, false);
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
  module.wire("kernel_state", 1);
  module.wire("all_finish", 1);

  Instance instKernelStateReg("standard_reg", "kernel_state_reg", {1});

  // kernel_state == 0 && ap_start == 1 -> go to kernel_state == 1
  // kernel_state == 1 && all_finish == 1 -> go to kernel_state == 0
  instKernelStateReg.connect("clk", "clk")
      .connect("rst", "(rst) || (kernel_state == 1 && all_finish == 1)")
      .connect("data_in", "1")
      .connect("data_out", "kernel_state")
      .connect("enable", "(kernel_state == 0 && ap_start == 1)");
  module.instantiate(instKernelStateReg);

  // State registers
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();

    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      // Storing the input data (in case the function is not ready to take all
      // arguments at once).

      module.wire(signalName + "_data_reg", type.getDataBitWidth());

      Instance instDataReg("standard_reg", signalName + "_data_reg_inst",
                           {type.getDataBitWidth()});
      instDataReg.connect("clk", "clk")
          .connect("rst", "rst")
          .connect("data_in", signalName)
          .connect("data_out", signalName + "_data_reg")
          .connect("enable", "(kernel_state == 0) && (ap_start == 1)");
      module.instantiate(instDataReg);

      module.wire(signalName + "_sent", 1);

      Instance instSentReg("standard_reg", signalName + "_sent_reg_inst", {1});
      instSentReg.connect("clk", "clk")
          .connect("rst", "rst")
          .connect("data_in", "1")
          .connect("data_out", signalName + "_sent")
          .connect("enable",
                   "(kernel_state == 1) && (" + signalName + "_ready == 1)");
      module.instantiate(instSentReg);
    }
    if (handshake::ControlType type =
            dyn_cast<handshake::ControlType>(arg.getType())) {
      module.wire(signalName + "_sent", 1);
      Instance instSentReg("standard_reg", signalName + "_sent_reg_inst", {1});
      instSentReg.connect("clk", "clk")
          .connect("rst", "rst")
          .connect("data_in", "1")
          .connect("data_out", signalName + "_sent")
          .connect("enable",
                   "(kernel_state == 1) && (" + signalName + "_ready == 1)");
      module.instantiate(instSentReg);
    }
  }

  // Communicating with the input channels of the kernel
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      module.wire(signalName + "_valid", 1);
      module.wire(signalName + "_ready", 1);
      module.assign(signalName + "_valid",
                    "(kernel_state == 1) && !" + signalName + "_sent");

    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(arg.getType())) {
      module.wire(signalName + "_valid", 1);
      module.wire(signalName + "_ready", 1);
      module.assign(signalName + "_valid",
                    "(kernel_state == 1) && !" + signalName + "_sent");
    }
  }

  // Communicating with the output channels of the kernel
  SmallVector<std::string> outputValidSignals;
  SmallVector<std::string> outputReadySignals;
  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp.getResultTypes(), funcOp.getResNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();

    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(resType)) {
      module.wire(signalName + "_valid", 1);
      module.wire(signalName + "_ready", 1);
      outputValidSignals.push_back(signalName + "_valid");
      outputReadySignals.push_back(signalName + "_ready");
    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(resType)) {
      module.wire(signalName + "_valid", 1);
      module.wire(signalName + "_ready", 1);
      outputValidSignals.push_back(signalName + "_valid");
      outputReadySignals.push_back(signalName + "_ready");
    }
  }

  // The output is ready if all other outputs are valid (this is used to avoid
  // any combinational dependency between valid and ready).
  for (auto [id, ready] : llvm::enumerate(outputReadySignals)) {
    SmallVector<std::string> validSignals;
    for (auto [id2, valid] : llvm::enumerate(outputValidSignals)) {
      if (id != id2) {
        validSignals.push_back(valid);
      }
    }
    module.assign(ready, llvm::join(validSignals, " && "));
  }

  module.assign("all_finish", llvm::join(outputValidSignals, " && "));
  module.assign("ap_done", "all_finish");
}

void createInstantiation(VerilogModule &module, handshake::FuncOp funcOp,
                         raw_indented_ostream &os) {
  Instance inst(funcOp.getName().str(), funcOp.getName().str() + "_inst");
  inst.connect("clk", "clk")
      .connect("rst", "rst")
      .connect("go", "go")
      .connect("done", "done");
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      inst.connect(signalName, signalName)
          .connect(signalName + "_valid", signalName + "_valid")
          .connect(signalName + "_ready", signalName + "_ready");
    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(arg.getType())) {
      inst.connect(signalName + "_valid", signalName + "_valid")
          .connect(signalName + "_ready", signalName + "_ready");
    } else if (mlir::IntegerType type =
                   dyn_cast<mlir::IntegerType>(arg.getType())) {
      inst.connect(signalName, signalName);
    } else if (mlir::MemRefType type =
                   dyn_cast<mlir::MemRefType>(arg.getType())) {
      inst.connect(signalName + INST_LOAD_EN, signalName + ITF_LOAD_EN)
          .connect(signalName + INST_LOAD_ADDR, signalName + ITF_LOAD_ADDR)
          .connect(signalName + INST_LOAD_DATA, signalName + ITF_LOAD_DATA)
          .connect(signalName + INST_STORE_EN, signalName + ITF_STORE_EN)
          .connect(signalName + INST_STORE_ADDR, signalName + ITF_STORE_ADDR)
          .connect(signalName + INST_STORE_DATA, signalName + ITF_STORE_DATA);
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

  auto m = VerilogModule(funcOp.getName().str() + "_vivado_ip_adapter");
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

  os << STR_STD_REG << "\n\n";
  // Write each module's RTL implementation to a separate file
  for (handshake::FuncOp funcOp : modOp->getOps<handshake::FuncOp>()) {
    createAdapter(funcOp, os);
    return 0;
  }
}
