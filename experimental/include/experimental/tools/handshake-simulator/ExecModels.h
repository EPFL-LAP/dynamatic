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

#ifndef EXEC_MODELS_H
#define EXEC_MODELS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Any.h"
#include "llvm/Support/Debug.h"
#include <string>

namespace dynamatic {
namespace experimental {

struct ExecutableModel;

bool tryToExecute(circt::Operation *op,
                  llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                  llvm::DenseMap<mlir::Value, double> &timeMap,
                  std::vector<mlir::Value> &scheduleList,
                  std::map<std::string, std::unique_ptr<ExecutableModel>> &models,
                  double latency);

/// Data structure to hold functions to execute each components
struct ExecutableModel {
  /// A wrapper function to do all the utility stuff in order to simulate
  /// the execution correctly
  virtual bool tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                  llvm::DenseMap<unsigned, unsigned> &memoryMap,
                  llvm::DenseMap<mlir::Value, double> &timeMap,
                  std::vector<std::vector<llvm::Any>> &store,
                  std::vector<mlir::Value> &scheduleList,
                  std::map<std::string, std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
                  circt::Operation &op) {
    return tryToExecute(&op, valueMap, timeMap, scheduleList, models, 0);
  };

  /// The pure execution of the operation
  virtual void execute(std::vector<llvm::Any> &ins,
                       std::vector<llvm::Any> &outs,
                       circt::Operation &op) = 0;

  virtual ~ExecutableModel() {};
};

//----------------------------------------------------------------------------//
//                  Execution models structure definitions                    //
//----------------------------------------------------------------------------//

//--- Default CIRCT models ---------------------------------------------------//

struct DefaultFork : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

struct DefaultMerge : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

struct DefaultControlMerge : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

struct DefaultMux : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

struct DefaultBranch : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

struct DefaultConditionalBranch : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

struct DefaultSink : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

struct DefaultConstant : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

struct DefaultBuffer : public ExecutableModel {
  bool
  tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
             llvm::DenseMap<unsigned, unsigned> &memoryMap,
             llvm::DenseMap<mlir::Value, double> &timeMap,
             std::vector<std::vector<llvm::Any>> &store,
             std::vector<mlir::Value> &scheduleList,
             std::map<std::string,
                      std::unique_ptr<dynamatic::experimental::ExecutableModel>>
                 &models,
             circt::Operation &op) override;
  void execute(std::vector<llvm::Any> &ins,
               std::vector<llvm::Any> &outs,
               circt::Operation &op) override;
};

//--- Custom models ----------------------------------------------------------//

} // namespace experimental
} // namespace dynamatic

#endif
