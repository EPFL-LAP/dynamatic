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

/// Maps configurated execution models to their execution structure
using ModelMap =
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>>;

/// Maps operations to an internal state everyone can access
using StateMap = llvm::DenseMap<circt::Operation*, llvm::Any>;

/// Data structure to hold memory controllers internal state
struct MemoryControllerState {
  llvm::SmallVector<AccessTypeEnum> accesses; // might be useless
  /// Stores the index of the operands containing stores addr
  llvm::SmallVector<unsigned> storesAddr;
  /// Stores the index of the operands containing stores data
  llvm::SmallVector<unsigned> storesData;
  /// Stores the index of the operands containing loads addr
  llvm::SmallVector<unsigned> loadsAddr;
};

/// Data structure to hold informations passed to tryExecute functions
struct ExecutableData {
  llvm::DenseMap<mlir::Value, llvm::Any> &valueMap;
  llvm::DenseMap<unsigned, unsigned> &memoryMap;
  llvm::DenseMap<mlir::Value, double> &timeMap;
  std::vector<std::vector<llvm::Any>> &store;
  llvm::SmallVector<mlir::Value> &scheduleList;
  ModelMap &models;
  StateMap &stateMap;
};

/// Data structure to hold functions to execute each components
struct ExecutableModel {
  /// A wrapper function to do all the utility stuff in order to simulate
  /// the execution correctly
  virtual bool tryExecute(ExecutableData &data, circt::Operation &op) = 0;

  virtual ~ExecutableModel(){};
};

/// Initialises the mapping from operations to the configurated structure
/// ADD YOUR STRUCT TO THE CORRESPONDING MAP IN THIS METHOD
mlir::LogicalResult initialiseMap(llvm::StringMap<std::string> &funcMap,
                                  ModelMap &models);

//----------------------------------------------------------------------------//
//                  Execution models structure definitions                    //
//----------------------------------------------------------------------------//

//--- Default CIRCT models ---------------------------------------------------//

struct DefaultFork : public ExecutableModel {
  virtual bool tryExecute(ExecutableData &data, circt::Operation &op) override;
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
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TOOLS_HANDSHAKESIMULATOR_EXECMODELS_H
