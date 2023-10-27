//===- TimingModels.h - Parse/Represent comp. timing models -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the infrastrucure to parse, represent, and query timing
// models for RTL components, here represented as MLIR operations. The component
// timing characterizations that are at the source of our timing models, notably
// used during buffer placement, can be obtained from external tools. These
// characterizations must then be represented in our JSON data format (needs
// formal documentation!) which can then be deserialized into the C++ data types
// defined in this file.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_TIMINGMODELS_H
#define DYNAMATIC_SUPPORT_TIMINGMODELS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <unordered_map>

namespace dynamatic {

/// Maximum datawidth supported by the timing models.
const unsigned MAX_DATAWIDTH = 64;

/// The type of a signal in a handshake channel: DATA, VALID, or READY.
enum class SignalType { DATA, VALID, READY };

/// The type of a port: IN or OUT.
enum class PortType { IN, OUT };

/// Represents a metric of type M that is bitwidth-dependent i.e., whose value
/// changes depending on the bitwidth of the signal it refers to. Internally, it
/// maps any number of the metric's data points (with no specific order) with
/// the bitwidth at which they were measured.
template <typename M>
struct BitwidthDepMetric {
public:
  /// Data points for the metric, mapping a bitwidth with the metric's value at
  /// that bitwidth.
  std::unordered_map<unsigned, M> data;

  /// Determines the value of the metric at the bitwidth that is closest and
  /// greater than or equal to the passed bitwidth. Fails if the bitwidth is
  /// strictly larger than the data point with largest bitwidth; succeeds and
  /// sets the metric's value otherwise. For the returned metric to make any
  /// sense, the metric must be monotonically increasing with respect to the
  /// bitwidth.
  LogicalResult getCeilMetric(unsigned bitwidth, M &metric) const;

  /// Determines the value of the metric at the bitwidth that is closest and
  /// greater than or equal to the passed operation's datawidth. See override's
  /// documentation for more details.
  LogicalResult getCeilMetric(Operation *op, M &metric) const;
};

/// Deserializes a JSON value into a BitwidthDepMetric<double>. See
/// ::llvm::json::Value's documentation for a longer description of this
/// function's behavior.
bool fromJSON(const llvm::json::Value &value, BitwidthDepMetric<double> &metric,
              llvm::json::Path path);

/// Stores the timing model for an operation's type, usually parsed from a JSON
/// file. It stores the operation's (datawidth-dependent) latencies,
/// (datawidth-dependent) data delays, valid wire delay, and ready wire delay.
/// It also stores that same information for the operation's input ports (as a
/// whole) and output ports (as a whole), accompanied by a number of transparent
/// and opaque buffer slots that may be present at the operation's ports.
struct TimingModel {
public:
  struct PortModel {
    /// Port's data delay, depending on its bitwidth.
    BitwidthDepMetric<double> dataDelay;
    /// Delay of port's valid wire.
    double validDelay = 0.0;
    /// Delay of port's ready wire.
    double readyDelay = 0.0;
    /// Number of transparent buffer slots on the port.
    unsigned transparentSlots = 0;
    /// Number of opaque buffer slots on the port.
    unsigned opaqueSlots = 0;
  };

  /// Operation's latency, depending on its bitwidth.
  BitwidthDepMetric<double> latency;
  /// Operation's data delay, depending on its bitwidth.
  BitwidthDepMetric<double> dataDelay;
  /// Delay of valid wire.
  double validDelay = 0.0;
  /// Delay of ready wire.
  double readyDelay = 0.0;

  /// Input ports' timing model.
  PortModel inputModel;
  /// Output ports' timing model.
  PortModel outputModel;

  /// Combinational delay from any valid input to a ready output pin.
  double validToReady = 0.0;
  /// Combinational delay from the condition input pin  to any valid output pin.
  double condToValid = 0.0;
  /// Combinational delay from the condition input pin to any ready output pin.
  double condToReady = 0.0;
  /// Combinational delay from any valid input pin to the condition output pin.
  double validToCond = 0.0;
  /// Combinational delay from any valid input to any data output.
  double validToData = 0.0;

  /// Computes the total data delay (input + internal + output delays) for a
  /// specific bitwidth. Fails if the provided bitwidth exceeds the highest data
  /// bitwidth in the model. On success, sets the last argument to the data
  /// delay.
  LogicalResult getTotalDataDelay(unsigned bitwidth, double &delay) const;

  /// Returns the total valid delay (input + internal + output delays).
  double getTotalValidDelay() const {
    return inputModel.validDelay + validDelay + outputModel.validDelay;
  };

  /// Returns the total ready delay (input + internal + output delays).
  double getTotalReadyDelay() const {
    return inputModel.readyDelay + readyDelay + outputModel.readyDelay;
  };
};

/// Deserializes a JSON value into a TimingModel. See ::llvm::json::Value's
/// documentation for a longer description of this function's behavior.
bool fromJSON(const llvm::json::Value &jsonValue, TimingModel &model,
              llvm::json::Path path);

/// Deserializes a JSON value into a TimingModel. See ::llvm::json::Value's
/// documentation for a longer description of this function's behavior.
bool fromJSON(const llvm::json::Value &jsonValue, TimingModel::PortModel &model,
              llvm::json::Path path);

/// Holds the timing models for a set of operations (internally identified by
/// their unique name), usually parsed from a JSON file. The class provides
/// accessor methods to quickly get specific information from the underlying
/// timing models, which can also be retrieved in their entirety.
class TimingDatabase {
public:
  /// Creates a TimingDatabase with an MLIR context used internally to identify
  /// MLIR operations from their name.
  inline TimingDatabase(MLIRContext *ctx) : ctx(ctx) {}

  /// Inserts a timing model in the database with the provided name. Returns
  /// true if no timing model existed for this name prior to the calls, or false
  /// otherwise.
  bool insertTimingModel(StringRef name, TimingModel &model);

  /// Returns the timing model corresponding to the operation whose name is
  /// passed as argument, if any exists.
  const TimingModel *getModel(OperationName opName) const;

  /// Returns the timing model corresponding to the operation, if any exists.
  const TimingModel *getModel(Operation *op) const;

  /// Attempts to get an operation's latency. On success, sets the last argument
  /// to the requested latency.
  LogicalResult getLatency(Operation *op, double &latency) const;

  /// Attempts to get an operation's internal delay for a specific signal type.
  /// On success, sets the last argument to the requested delay.
  LogicalResult getInternalDelay(Operation *op, SignalType type,
                                 double &delay) const;

  /// Attempts to get an operation's port delay for a specific signal and port
  /// type. On success, sets the last argument to the requested delay.
  LogicalResult getPortDelay(Operation *op, SignalType signalType,
                             PortType portType, double &delay) const;

  /// Attempts to get an operation's total delay (internal delay + input delay +
  /// output delay) for a specific signal type. On success, sets the last
  /// argument to the requested delay.
  LogicalResult getTotalDelay(Operation *op, SignalType type,
                              double &delay) const;

  /// Parses a JSON file whose path is given as argument and adds all the timing
  /// models it contains to the passed timing database.
  static LogicalResult readFromJSON(std::string &jsonPath,
                                    TimingDatabase &timingDB);

private:
  /// MLIR context with which to identify MLIR operations from their name.
  MLIRContext *ctx;

  /// Maps operation names to their timing model.
  DenseMap<OperationName, TimingModel> models;
};

/// Deserializes a JSON value into a TimingDatabase. See ::llvm::json::Value's
/// documentation for a longer description of this function's behavior.
bool fromJSON(const llvm::json::Value &jsonValue, TimingDatabase &timingDB,
              llvm::json::Path path);

} // namespace dynamatic

namespace llvm {
namespace json {
/// Deserializes a JSON value into an unsigned number. This function is placed
/// inside of the ::llvm::json namespace since the deserialization target type
/// is a standard type. See ::llvm::json::Value's documentation for a longer
/// description of this function's behavior.
bool fromJSON(const llvm::json::Value &value, unsigned &number,
              llvm::json::Path path);
} // namespace json
} // namespace llvm

#endif // DYNAMATIC_SUPPORT_TIMINGMODELS_H
