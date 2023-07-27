//===- ParseCircuitJson.h -Parse circuit json file  -------------*- C++ -*-===//
//
// This file declares functions to parse the elements in the circuit json file.
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"

namespace dynamatic {
namespace buffer {

/// Data structure to store the information of a unit.
/// The information include the timing information and the buffer placement
/// prerequisite of the connected channels. The timing information include the
/// latency and data delay of the unit, as well as the data, valid, ready delay
/// information of the input and output ports. The buffer placement prerequisite
/// specify the transparent buffer and the opaque buffer connected to the input
/// and output ports.
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
};

/// Get the short name of the operation, e.g., "add" for "handshake.add".
std::string getOperationShortStrName(Operation *op);

/// Get the unit delay with respect to the operation type, including data, valid
/// and ready.
double getUnitDelay(Operation *op,
                    std::map<std::string, buffer::UnitInfo> &unitInfo,
                    std::string type = "data");

/// Get the unit(op) latency.
double getUnitLatency(Operation *op,
                      std::map<std::string, buffer::UnitInfo> &unitInfo);

/// Get the combinational delay of the operation, including its input port
/// delay, unit delay, and output port delay.
double getCombinationalDelay(Operation *op,
                             std::map<std::string, buffer::UnitInfo> &unitInfo,
                             std::string type = "data");

/// Get the delay of a port(in, out) connected to a channel.
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