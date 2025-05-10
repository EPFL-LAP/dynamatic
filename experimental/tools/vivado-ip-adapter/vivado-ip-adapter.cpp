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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

constexpr int clog2(uint32_t x) {
  return (x <= 1) ? 0 : 1 + clog2((x + 1) >> 1);
}

// Suffix for the names of the memory signals of the instantiated module
static const std::string INST_STORE_ADDR = "_storeAddr";
static const std::string INST_STORE_EN = "_storeEn";
static const std::string INST_STORE_DATA = "_storeData";
static const std::string INST_LOAD_ADDR = "_loadAddr";
static const std::string INST_LOAD_EN = "_loadEn";
static const std::string INST_LOAD_DATA = "_loadData";
// Suffix for the names of the memory signals of the interface
static const std::string ITF_STORE_ADDR = "_storeAddr";
static const std::string ITF_STORE_EN = "_storeEn";
static const std::string ITF_STORE_DATA = "_storeData";
static const std::string ITF_LOAD_ADDR = "_loadAddr";
static const std::string ITF_LOAD_EN = "_loadEn";
static const std::string ITF_LOAD_DATA = "_loadData";

static const std::string CLK = "clk";
static const std::string RST = "rst";

static const std::string VALID = "valid";
static const std::string READY = "ready";

// Interface signals of the standard register
static const std::string REG_IN = "data_in";
static const std::string REG_OUT = "data_out";
static const std::string ENABLE = "enable";

static const std::string GLOBAL_START = "ap_start";
static const std::string GLOBAL_FINISH = "ap_done";
static const std::string GLOBAL_READY = "ap_ready";

static const std::string KERNEL_STATE = "kernel_state";

static const std::string SUPPORT_MODULES = R"DELIM(
// An 1-slot buffer that stores the argument
module kernel_arg_buffer #(
  parameter DATA_TYPE = 32
)(
  clk,
  rst,
  kernel_state,
  ap_start,
  data_in,
  data_out,
  valid,
  ready
);
  input clk;
  input rst;
  input [DATA_TYPE-1:0]data_in;
  output [DATA_TYPE-1:0]data_out;
  output valid;
  input ready;
  input ap_start;
  input kernel_state;

  reg [DATA_TYPE-1:0]data_reg = 0;
  reg sent = 0;

  always @(posedge clk) begin
    if (rst) 
      data_reg <= 0;
    else begin
      if (ap_start && kernel_state == 0) 
        data_reg <= data_in;
    end
  end

  always @(posedge clk) begin
    if (rst || (ap_start && kernel_state == 0))
      sent <= 0;
    else begin
      if (valid && ready)
        sent <= 1;
    end
  end 

  assign valid = (!sent) && (kernel_state == 1);
  assign data_out = data_reg;
endmodule

module kernel_state_reg (
  clk,
  rst, 
  ap_start,
  ap_ready,
  kernel_state,
  all_finish,
  ap_done
);
  input clk;
  input rst;
  input ap_start;
  output ap_ready;
  output kernel_state;
  // All the output channels of the handshake kernel are valid
  input all_finish;
  output ap_done;

  reg kernel_state_reg = 0;

  assign kernel_state = kernel_state_reg;
  assign ap_done = all_finish && (kernel_state_reg == 1);
  assign ap_ready = (kernel_state == 0);

  always @(posedge clk) begin
    if (rst) begin
      kernel_state_reg <= 0;
    end else if (ap_start && kernel_state_reg == 0)
      kernel_state_reg <= 1;
    // ap_done will be high for one cycle (when all_finish and we are in the
    // running state). After that we return to the idle state
    else if (ap_done)
      kernel_state_reg <= 0;
  end
endmodule

module kernel_join #(
  parameter SIZE = 2
)(
  input [SIZE - 1 : 0] ins_valid,
  input outs_ready,

  output reg  [SIZE - 1 : 0] ins_ready = 0,
  output outs_valid
);
  // AND of all the bits in ins_valid vector
  assign outs_valid = &ins_valid;
  
  reg [SIZE - 1 : 0] singleValid = 0;
  integer i, j;
  
  always @(*)begin
    for (i = 0; i < SIZE; i = i + 1) begin
      singleValid[i] = 1;
      for (j = 0; j < SIZE; j = j + 1)
        if (i != j)
          singleValid[i] = singleValid[i] & ins_valid[j];
    end
    for (i = 0; i < SIZE; i = i + 1) begin
      ins_ready[i] = singleValid[i] & outs_ready;
    end
  end
endmodule
)DELIM";

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

// Get input argument of type Ty and their associated names
template <typename Ty>
llvm::SmallVector<std::pair<Ty, std::string>>
getInputArguments(handshake::FuncOp funcOp) {
  llvm::SmallVector<std::pair<Ty, std::string>> interfaces;
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    if (Ty type = dyn_cast<Ty>(arg.getType())) {
      std::string argName = portAttr.dyn_cast<StringAttr>().data();
      interfaces.emplace_back(type, argName);
    }
  }
  return interfaces;
}

template <typename Ty>
llvm::SmallVector<std::pair<Ty, std::string>>
getOutputArguments(handshake::FuncOp funcOp) {
  llvm::SmallVector<std::pair<Ty, std::string>> interfaces;

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp.getResultTypes(), funcOp.getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (Ty type = dyn_cast<Ty>(resType)) {
      interfaces.emplace_back(type, argName);
    }
  }
  return interfaces;
}
void declareHandshakeSignals(VerilogModule &m, const std::string &sigName) {
  m.wire(sigName + "_" + VALID, /* width = */ 1);
  m.wire(sigName + "_" + READY, /* width = */ 1);
}

// A buffer module that buffers the data from the input before the dataflow
// circuit becomes ready to take it
void createArgumentBuffer(VerilogModule &m, const std::string &sigName,
                          std::optional<unsigned> dataWidth) {

  unsigned bitWidth = dataWidth ? dataWidth.value() : 1;
  // Drive the valid signal that goes to the handshake kernel
  Instance instDataReg("kernel_arg_buffer", sigName + "_data_reg_inst",
                       {bitWidth});

  std::string dataIn = dataWidth ? sigName : "1";
  std::string dataOut = dataWidth ? sigName + "_" + REG_OUT : "";
  instDataReg.connect(CLK, CLK)
      .connect(RST, RST)
      .connect(KERNEL_STATE, KERNEL_STATE)
      .connect(GLOBAL_START, GLOBAL_START)
      .connect(REG_IN, dataIn)
      .connect(REG_OUT, dataOut)
      .connect(VALID, sigName + "_" + VALID)
      .connect(READY, sigName + "_" + READY);
  m.instantiate(instDataReg);
}

void createModuleInterface(VerilogModule &module, handshake::FuncOp funcOp,
                           raw_indented_ostream &os) {
  module.port(RST, 1, false);
  module.port(CLK, 1, false);
  module.port(GLOBAL_START, 1, false);
  module.port(GLOBAL_READY, 1, false);
  module.port(GLOBAL_FINISH, 1, true);

  // Only create the data interface for each input channel
  for (auto [type, sigName] : getInputArguments<ChannelType>(funcOp)) {
    module.port(sigName, type.getDataBitWidth(), /* isOutput = */ false);
  }

  for (auto [type, sigName] : getInputArguments<MemRefType>(funcOp)) {
    unsigned addressWidth = clog2(type.getNumElements());
    unsigned dataWidth = type.getElementTypeBitWidth();
    module.port(sigName + ITF_LOAD_ADDR, addressWidth, /*isOutput = */ true);
    module.port(sigName + ITF_LOAD_EN, 1, /*isOutput = */ true);
    module.port(sigName + ITF_STORE_EN, 1, /*isOutput = */ true);
    module.port(sigName + ITF_STORE_ADDR, addressWidth, /*isOutput =*/true);
    module.port(sigName + ITF_STORE_DATA, dataWidth, /*isOutput = */ true);
    module.port(sigName + ITF_LOAD_DATA, dataWidth, /*isOutput = */ false);
  }

  // Only create the data interface for each output channel
  for (auto [type, sigName] : getOutputArguments<ChannelType>(funcOp)) {
    module.port(sigName, type.getDataBitWidth(), /* isOutput = */ true);
  }
}

void createInternalSignals(VerilogModule &module, handshake::FuncOp funcOp) {

  // Storing the current state of the module
  // (kernel_state = 0) => idle state
  // (kernel_state = 1) => working state
  module.wire("kernel_state", 1);
  module.wire("all_finish", 1);

  Instance instKernelStateReg("kernel_state_reg", "kernel_state_reg_inst");

  // kernel_state == 0 && ap_start == 1 -> go to kernel_state == 1
  // kernel_state == 1 && all_finish == 1 -> go to kernel_state == 0
  instKernelStateReg.connect(CLK, CLK)
      .connect(RST, RST)
      .connect(GLOBAL_START, GLOBAL_START)
      .connect(GLOBAL_READY, GLOBAL_READY)
      .connect(KERNEL_STATE, KERNEL_STATE)
      .connect("all_finish", "all_finish")
      .connect(GLOBAL_FINISH, GLOBAL_FINISH);
  module.instantiate(instKernelStateReg);

  for (auto [type, sigName] : getInputArguments<ChannelType>(funcOp)) {
    module.wire(sigName + "_" + REG_OUT, type.getDataBitWidth());
    declareHandshakeSignals(module, sigName);
    createArgumentBuffer(module, sigName, type.getDataBitWidth());
  }

  for (auto [type, sigName] : getInputArguments<ControlType>(funcOp)) {
    declareHandshakeSignals(module, sigName);
    createArgumentBuffer(module, sigName, std::nullopt);
  }

  // Communicating with the output channels of the kernel
  SmallVector<std::string> outputValidSignals;
  SmallVector<std::string> outputReadySignals;

  for (auto [type, sigName] : getOutputArguments<ChannelType>(funcOp)) {
    declareHandshakeSignals(module, sigName);
    outputValidSignals.push_back(sigName + "_" + VALID);
    outputReadySignals.push_back(sigName + "_" + READY);
  }

  for (auto [type, sigName] : getOutputArguments<ControlType>(funcOp)) {
    declareHandshakeSignals(module, sigName);
    outputValidSignals.push_back(sigName + "_" + VALID);
    outputReadySignals.push_back(sigName + "_" + READY);
  }

  // The output channels are regulated by a join: the are stalled until all
  // output channels become valid
  Instance adapterJoin("kernel_join", "join_inst", {funcOp.getNumResults()});

  adapterJoin.connect("ins_" + VALID,
                      "{" + llvm::join(outputValidSignals, ", ") + "}");
  adapterJoin.connect("ins_" + READY,
                      "{" + llvm::join(outputReadySignals, ", ") + "}");
  adapterJoin.connect("outs_" + VALID, "all_finish");
  adapterJoin.connect("outs_" + READY, "1");
  module.instantiate(adapterJoin);
}

void connectHandshakeSignals(Instance &inst, const std::string &sigName) {
  inst.connect(sigName + "_" + VALID, sigName + "_" + VALID);
  inst.connect(sigName + "_" + READY, sigName + "_" + READY);
}

void createInstantiation(VerilogModule &module, handshake::FuncOp funcOp,
                         raw_indented_ostream &os) {
  Instance inst(funcOp.getName().str(), funcOp.getName().str() + "_inst");
  inst.connect(CLK, CLK).connect(RST, RST);
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp.getBodyBlock()->getArguments(), funcOp.getArgNames())) {
    std::string signalName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      inst.connect(signalName, signalName + "_" + REG_OUT);
      connectHandshakeSignals(inst, signalName);
    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(arg.getType())) {
      connectHandshakeSignals(inst, signalName);
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
      connectHandshakeSignals(inst, signalName);
    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(resType)) {
      connectHandshakeSignals(inst, signalName);
    } else if (mlir::IntegerType type = dyn_cast<mlir::IntegerType>(resType)) {
      inst.connect(signalName, signalName);
    }
  }
  module.instantiate(inst);
}

void createAdapter(handshake::FuncOp funcOp, raw_indented_ostream &os) {

  auto m = VerilogModule(funcOp.getName().str() + "_ip");
  createModuleInterface(m, funcOp, os);

  createInstantiation(m, funcOp, os);

  createInternalSignals(m, funcOp);
  m.emit(os);
}

int main(int argc, char **argv) {

  static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                            cl::desc("<input file>"));
  cl::ParseCommandLineOptions(
      argc, argv,
      "Create a rigid wrapper for each handshake function in an MLIR file.\n");

  // We only need the Handshake and HW dialects
  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect>();
  context.allowUnregisteredDialects();

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

  os << "`timescale 1ns / 1ps\n\n";
  // Write each module's RTL implementation to a separate file
  for (handshake::FuncOp funcOp : modOp->getOps<handshake::FuncOp>()) {
    os << llvm::formatv(SUPPORT_MODULES.c_str(), funcOp.getName().str())
       << "\n\n";
    createAdapter(funcOp, os);
    return 0;
  }
}
