//===- ParseCircuitJson.h -Parse circuit json file  -----------*- C++ -*-===//
//
// This file declares functions to parse the elements in the circuit json file.
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"

namespace dynamatic {
namespace buffer {

struct UnitInfo {
public:
  std::vector<std::pair<unsigned, double>> latency;
  std::vector<std::pair<unsigned, double>> dataDelay;
  double validDelay = 0.0, readyDelay = 0.0;
  std::vector<std::pair<unsigned, double>> inPortDataDelay;
  std::vector<std::pair<unsigned, double>> outPortDataDelay;
  double inPortValidDelay = 0.0, inPortReadyDelay = 0.0;
  double outPortValidDelay = 0.0, outPortReadyDelay = 0.0;

  unsigned inPortTransBuf = 0, inPortOpBuf = 0;
  unsigned outPortTransBuf = 0, outPortOpBuf = 0;

  template <typename T1, typename T2>
  void printPairs(const std::vector<std::pair<T1, T2>> &pairs) {
    for (const auto &pair : pairs) {
      llvm::errs() << pair.first << ", " << pair.second << "\n";
      ;
    }
  }
  void print() {
    // Printing the contents of the vectors
    llvm::errs() << "Contents of latency:"
                 << "\n";
    printPairs(latency);

    llvm::errs() << "Contents of dataDelay:"
                 << "\n";
    printPairs(dataDelay);

    llvm::errs() << "Contents of inPortDataDelay:"
                 << "\n";
    printPairs(inPortDataDelay);

    llvm::errs() << "Contents of outPortDataDelay:"
                 << "\n";
    printPairs(outPortDataDelay);

    llvm::errs() << "validDelay" << validDelay << "\n";
    llvm::errs() << "readyDelay" << readyDelay << "\n";

    llvm::errs() << "inportValidDelay" << inPortValidDelay << "\n";
    llvm::errs() << "inportReadyDelay" << inPortReadyDelay << "\n";

    llvm::errs() << "outPortValidDelay" << outPortValidDelay << "\n";
    llvm::errs() << "outPortReadyDelay" << outPortReadyDelay << "\n";

    llvm::errs() << "inport{ "
                 << "transBuf " << inPortTransBuf << " opBuf " << inPortOpBuf
                 << "}\n";
    llvm::errs() << "outport{ "
                 << "transBuf " << outPortTransBuf << " opBuf " << outPortOpBuf
                 << "}\n";
  }
};

// double getTimeInfo(Operation *op, std::map<std::string, UnitInfo> &unitInfo);
std::string getOperationShortStrName(Operation *op);

double getUnitDelay(Operation *op,
                    std::map<std::string, buffer::UnitInfo> &unitInfo,
                    std::string type = "data");

double getUnitLatency(Operation *op,
                      std::map<std::string, buffer::UnitInfo> &unitInfo);

double getCombinationalDelay(Operation *op,
                             std::map<std::string, buffer::UnitInfo> &unitInfo,
                             std::string type = "data");

double getPortDelay(Value channel,
                    std::map<std::string, buffer::UnitInfo> &unitInfo,
                    std::string type = "in");

/// Read timing info for units and channels in CFDFC from the input json.
/// The units delay and latency are determined by the units type
/// The channels timing info are described by the input ports and output
/// ports timing characteristics
LogicalResult parseJson(const std::string &jsonString,
                        std::map<std::string, UnitInfo> &unitInfo);

} // namespace buffer
} // namespace dynamatic