//===- ExecModels.h - Handshake MLIR and Dynamatic Operations -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Inherited from CIRCT which holds the informations about each operation
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TOOLS_HANDSHAKESIMULATOR_EXECMODELS_H
#define EXPERIMENTAL_TOOLS_HANDSHAKESIMULATOR_EXECMODELS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Any.h"
#include "llvm/Support/Debug.h"

#include <string>

namespace dynamatic {
namespace experimental {

struct ExecutableModel;

/// Maps operations to an internal state everyone can access
using InternalDataMap = llvm::DenseMap<circt::Operation*, llvm::Any>;

/// Type for operations that only modify outs and ins
using ExecuteFunction = const std::function<void(std::vector<llvm::Any> &, std::vector<llvm::Any> &,
                             circt::Operation &)>;

//--- Execution Models -------------------------------------------------------//

/// Maps configurated execution models to their execution structure
using ModelMap =
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>>;

/// Data structure to hold informations passed to tryExecute functions
struct ExecutableData {
  /// Maps value (usually operands) to something to store
  /// (comparable to RAM)
  llvm::DenseMap<mlir::Value, llvm::Any> &valueMap;
  /// Maps memory controller ID to their offset value in store
  /// (store[memoryMap[SOME_ID]] is the beginning of the allocated memory
  /// area for this memory controller)
  llvm::DenseMap<unsigned, unsigned> &memoryMap;
  /// Program's memory. Accessed via loads and stores
  std::vector<std::vector<llvm::Any>> &store;
  /// Maps execution model name to their corresponding structure
  ModelMap &models;
  /// Maps operations to their corresponding internal state
  InternalDataMap &internalDataMap;
  /// Reference to the cycle counter
  unsigned &currentCycle;
};

/// Data structure to hold functions to execute each components
struct ExecutableModel {
  /// A wrapper function to do all the utility stuff in order to simulate
  /// the execution correctly. 
  /// Returns false if nothing was done, true if some type of jobs were done
  virtual bool tryExecute(ExecutableData &data, circt::Operation &op) = 0;

  virtual ~ExecutableModel(){};

  virtual bool isEndPoint() const { return false; };
};

/// Initialises the mapping from operations to the configurated structure
/// ADD YOUR STRUCT TO THE CORRESPONDING MAP IN THIS METHOD
mlir::LogicalResult initialiseMap(llvm::StringMap<std::string> &funcMap,
                                  ModelMap &models);


//--- Memory controller ------------------------------------------------------//

/// Data structure to hold informations about requests toward the mem controller
struct MemoryRequest {
  /// Whether a Load request or a Store request 
  AccessTypeEnum type;
  /// Address index of related to the memory request in the mem controller op
  unsigned addressIdx;
  /// Data index related to the memory request in the mem controller op
  /// Is defined only for Store requests
  unsigned dataIdx;
  /// Cycles it takes to complete the request, in the real circuit
  unsigned cyclesToComplete;
  /// Cycle at which it was previously executed
  unsigned lastExecution;
  /// True if all operands are available
  bool isReady;
};

/// Data structure to hold memory controllers internal state
struct MemoryControllerState {
  /// Stores all the store request towards the memory controller
  llvm::SmallVector<MemoryRequest> storeRequests;
  /// Stores all the loads request towards the memory controller
  llvm::SmallVector<MemoryRequest> loadRequests;
};


//--- Simulation logging -----------------------------------------------------//

/// All possible states
enum class ChannelState {
  NONE = 1, VALID = 2, READY = 3, VALID_READY = 4
};

/// Data structure to hold state of each value
struct StateChange {
  /// Cycle at which the state was changed
  unsigned cycle;
  /// The new state the value changed to
  ChannelState state;
};

using ExecutionMap = llvm::DenseMap<mlir::Value, std::vector<StateChange>>;

//----------------------------------------------------------------------------//
//                  Execution models structure definitions                    //
//----------------------------------------------------------------------------//

//--- Default CIRCT models ---------------------------------------------------//

struct DefaultFork : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DefaultMerge : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DefaultControlMerge : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DefaultMux : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DefaultBranch : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DefaultConditionalBranch : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DefaultSink : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DefaultConstant : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DefaultBuffer : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

//--- Dynamatic models -------------------------------------------------------//

/// Manages all store and load requests and answers them back
struct DynamaticMemController : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

/// Sends the address and the data to the memory controller
struct DynamaticStore : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

/// Sends the address to memory controller and wait for the result, then pass it
/// to successor(s)
struct DynamaticLoad : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

// Passes the operands to the results. A sort of termination for values
struct DynamaticReturn : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

/// Verifies that all returns and memory operations are finished and terminates
/// the program
struct DynamaticEnd : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
  bool isEndPoint() const override { return true; }
};

//--- ARITH IR Models -------------------------------------------------------//

struct ArithAddF : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct ArithAddI : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct ConstantIndexOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct ConstantIntOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct XOrIOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct CmpIOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct CmpFOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct SubIOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct SubFOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct MulIOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct MulFOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DivSIOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DivUIOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct DivFOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct IndexCastOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct ExtSIOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct ExtUIOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct AllocOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct BranchOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

struct CondBranchOp : public ExecutableModel {
  bool tryExecute(ExecutableData &data, circt::Operation &op) override;
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TOOLS_HANDSHAKESIMULATOR_EXECMODELS_H
