#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Support/HandshakeSimulator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::list<std::string> inputArgs(cl::Positional, cl::desc("<input args>"),
                                       cl::ZeroOrMore, cl::cat(mainCategory));

using namespace dynamatic::experimental;

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "simulator");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect, arith::ArithDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Assume a single Handshake function in the module
  handshake::FuncOp funcOp = *modOp->getOps<handshake::FuncOp>().begin();
  // Map to store numbers of times he value was changed
  mlir::DenseMap<Value, unsigned> heatMap;

  Simulator sim(funcOp);
  // An example of user defined callbacks that count the number of times the
  // value has been changed
  for (Operation &op : funcOp.getOps())
    for (auto res : op.getResults()) {
      heatMap.insert({res, 0});
      ValueFunc valueCallback = [&heatMap, res](const ValueState *oldState,
                                                const ValueState *newState,
                                                unsigned long cycleNum) {
        ++heatMap[res];
      };
      sim.onStateChange(res, valueCallback);
    }

  // An example of user defined callback function that prints the circuit state
  // on each clk rising edge
  ClkFunc clkCallBack =
      [](const mlir::DenseMap<Value, ValueState *> &curState) {
        llvm::outs() << "============================== clk! "
                        "=============================="
                     << "\n";
        for (auto &cur : curState) {
          auto val = cur.first;
          auto *state = cur.second;
          llvm::TypeSwitch<mlir::Type>(val.getType())
              .Case<handshake::ChannelType>(
                  [&](handshake::ChannelType channelType) {
                    auto *ch = static_cast<ChannelState *>(state);
                    llvm::outs() << ch->valid << " " << ch->ready << " "
                                 << ch->data.bitwidth << "\n";
                  })
              .Case<handshake::ControlType>(
                  [&](handshake::ControlType controlType) {
                    auto *ch = static_cast<ControlState *>(state);
                    llvm::outs() << ch->valid << " " << ch->ready << "\n";
                  })
              .Default([&](auto) {
                llvm::errs() << "Value " << val
                             << " has unsupported type, we should probably "
                                "report an error and stop";
                return 1;
              });
        }
      };

  sim.onClkRisingEdge(clkCallBack);

  sim.simulate(inputArgs);

  // Get the value which is under the highest stress
  unsigned stressedVal = 0;
  Value val;

  for (Operation &op : funcOp.getOps())
    for (auto res : op.getResults()) {
      stressedVal = std::max(stressedVal, heatMap[res]);
      val = res;
    }

  sim.printResults();
  llvm::outs() << "The most stressed value is:\n";
  llvm::outs() << val.getDefiningOp()->getAttr("handshake.name") << "-"
               << val.getUsers().begin()->getAttr("handshake.name") << " "
               << stressedVal << "\n";
}
