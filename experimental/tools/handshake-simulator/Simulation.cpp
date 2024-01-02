//===- Simulation.cpp - Handshake MLIR Operations -----------------------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions used to execute a restricted form of the
// standard dialect, and the handshake dialect.
//
//===----------------------------------------------------------------------===//

#include "experimental/tools/handshake-simulator/Simulation.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/JSON.h"
#include "experimental/tools/handshake-simulator/ExecModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <list>

#define DEBUG_TYPE "runner"
#define INDEX_WIDTH 32

using namespace llvm;
using namespace mlir;

namespace dynamatic {
namespace experimental {

/// This maps mlir instructions to the configurated execution model structure
ModelMap models;

/// A temporary memory map to store internal operations states. Used to make it
/// possible for operations to communicate between themselves without using
/// the store map and operands.
InternalDataMap internalDataMap;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Prettier printing for fatal errors
template <typename T>
static void fatalValueError(StringRef reason, T &value) {
  std::string err;
  llvm::raw_string_ostream os(err);
  os << reason << " ('";
  // Explicitly use ::print instead of << due to possibl operator resolution
  // error between i.e., circt::Operation::<< and operator<<(OStream &&OS, const
  // T &Value)
  value.print(os);
  os << "')\n";
  llvm::report_fatal_error(err.c_str());
}

/// Read a MLIR value from a stringstream and returns a casted version
/// NOLINTNEXTLINE(misc-no-recursion)
static Any readValueWithType(mlir::Type type, std::stringstream &arg) {
  if (type.isIndex()) {
    int64_t x;
    arg >> x;
    int64_t width = INDEX_WIDTH;
    APInt aparg(width, x);
    return aparg;
  }
  if (type.isa<mlir::IntegerType>()) {
    int64_t x;
    arg >> x;
    int64_t width = type.getIntOrFloatBitWidth();
    APInt aparg(width, x);
    return aparg;
  }
  if (type.isF32()) {
    float x;
    arg >> x;
    APFloat aparg(x);
    return aparg;
  }
  if (type.isF64()) {
    double x;
    arg >> x;
    APFloat aparg(x);
    return aparg;
  }
  if (auto tupleType = type.dyn_cast<TupleType>()) {
    char tmp;
    arg >> tmp;
    assert(tmp == '(' && "tuple should start with '('");
    std::vector<Any> values;
    unsigned size = tupleType.getTypes().size();
    values.reserve(size);
    // Parse element by element
    for (unsigned i = 0; i < size; ++i) {
      values.push_back(readValueWithType(tupleType.getType(i), arg));
      // Consumes either the ',' or the ')'
      arg >> tmp;
    }
    assert(tmp == ')' && "tuple should end with ')'");
    assert(
        values.size() == tupleType.getTypes().size() &&
        "expected the number of tuple elements to match with the tuple type");
    return values;
  }
  llvm_unreachable("unknown argument type!");
}

/// readValueWithType overload to output content to a string
static Any readValueWithType(mlir::Type type, StringRef in) {
  std::stringstream stream(in.str());
  return readValueWithType(type, stream);
}

/// Print MLIR value according to it's type
static void printAnyValueWithType(llvm::raw_ostream &out, mlir::Type type,
                                  Any &value) {
  if (type.isa<mlir::IntegerType>() || type.isa<mlir::IndexType>()) {
    out << any_cast<APInt>(value).getSExtValue();
  } else if (type.isa<mlir::FloatType>()) {
    out << any_cast<APFloat>(value).convertToDouble();
  } else if (type.isa<mlir::NoneType>()) {
    out << "none";
  } else if (auto tupleType = type.dyn_cast<mlir::TupleType>()) {
    auto values = any_cast<std::vector<llvm::Any>>(value);
    out << "(";
    llvm::interleaveComma(llvm::zip(tupleType.getTypes(), values), out,
                          [&](auto pair) {
                            auto [type, value] = pair;
                            return printAnyValueWithType(out, type, value);
                          });
    out << ")";
  } else {
    llvm_unreachable("Unknown result type!");
  }
}

/// Allocate a new matrix with dimensions given by the type, in the
/// given store. Puts the pseudo-pointer to the new matrix in the
/// store in memRefOffset (i.e. the first dimension index)
/// Returns a failed result if the shape isn't uni-dimensional
static LogicalResult allocateMemRef(mlir::MemRefType type, std::vector<Any> &in,
                                    std::vector<std::vector<Any>> &store,
                                    unsigned &memRefOffset) {
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() != 1)
    return failure();
  int64_t allocationSize = shape[0];

  unsigned ptr = store.size();
  store.resize(ptr + 1);
  store[ptr].resize(allocationSize);
  mlir::Type elementType = type.getElementType();
  int64_t width = elementType.getIntOrFloatBitWidth();
  for (int i = 0; i < allocationSize; ++i) {
    if (elementType.isa<mlir::IntegerType>()) {
      store[ptr][i] = APInt(width, 0);
    } else if (elementType.isa<mlir::FloatType>()) {
      store[ptr][i] = APFloat(0.0);
    } else {
      fatalValueError("Unknown result type!\n", elementType);
    }
  }
  memRefOffset = ptr;
  return success();
}

//===----------------------------------------------------------------------===//
// Handshake executer
//===----------------------------------------------------------------------===//

class HandshakeExecuter {
public:
  /// Entry point for circt::handshake::FuncOp top-level functions
  HandshakeExecuter(circt::handshake::FuncOp &func, CircuitState &circuitState,
                    std::vector<Any> &results,
                    std::vector<std::vector<Any>> &store,
                    mlir::OwningOpRef<mlir::ModuleOp> &module,
                    ModelMap &models);

  bool succeeded() const { return successFlag; }

private:
  /// Flag indicating whether execution was successful.
  bool successFlag = true;
};

/// Management of the state of each channel
struct StateManager {
  unsigned currentCycle;
  // True if any value has changed during the cycle
  bool valueChangedThisCycle;
  // True if any internal data has changed during the cycle
  bool internalDataChanged;
};

HandshakeExecuter::HandshakeExecuter(circt::handshake::FuncOp &func,
                                     CircuitState &circuitState,
                                     std::vector<Any> &results,
                                     std::vector<std::vector<Any>> &store,
                                     mlir::OwningOpRef<mlir::ModuleOp> &module,
                                     ModelMap &models) {
  successFlag = true;
  mlir::Block &entryBlock = func.getBody().front();
  // The arguments of the entry block.
  // A list of operations which might be ready to execute.
  std::list<circt::Operation *> readyList;
  // A map of memory ops
  llvm::DenseMap<unsigned, unsigned> memoryMap;

  // Initialize some operations
  bool hasEnd = false;
  func.walk([&](Operation *op) {
    // Set all return flags to false
    if (isa<circt::handshake::DynamaticReturnOp>(op)) {
      internalDataMap[op] = false;
    } else if (isa<circt::handshake::EndOp>(op)) {
      hasEnd = true;
    }
    // (Temporary)
    // Inititialize all channels
    for (auto value : op->getOperands())
      if (!circuitState.channelMap[value].isValid) {
        circuitState.channelMap[value].isValid = false;
        circuitState.channelMap[value].isReady = false;
        circuitState.channelMap[value].data = std::nullopt;
      }
  });

  assert(
      hasEnd &&
      "At least one 'end' operation is required for the program to terminate.");

  // Initialize the value map for buffers with initial values.
  for (auto bufferOp : func.getOps<circt::handshake::BufferOp>()) {
    if (bufferOp.getInitValues().has_value()) {
      auto initValues = bufferOp.getInitValueArray();
      assert(initValues.size() == 1 &&
             "Handshake-runner only supports buffer initialization with a "
             "single buffer value.");
      Value bufferRes = bufferOp.getResult();
      APInt value = APInt(bufferRes.getType().getIntOrFloatBitWidth(),
                          initValues.front());
      circuitState.storeValue(bufferRes, value);
    }
  }

  // Main simulation initilisation
  StateManager manager;
  manager.currentCycle = 0;
  ExecutableData execData{circuitState, memoryMap,       store,
                          models,       internalDataMap, manager.currentCycle};
  // Main simulation loop
  while (true) {
    // 1 cycle
    do {
      manager.valueChangedThisCycle = false;

      for (Operation &op : entryBlock.getOperations()) {
        auto opName = op.getName().getStringRef().str();
        auto &execModel = models[opName];
        if (models.find(opName) == models.end()) {
          successFlag = false;
          return;
        }
        // Execute operation accordingly to its execution model
        if (execModel.get()->tryExecute(execData, op)) {
          // Update the cycle ; a value changed thus we need to keep cycling
          manager.valueChangedThisCycle = true;
          if (execModel.get()->isEndPoint()) {
            successFlag = true;
            return;
          }
        }
      } // for operations

      // Transfer data from rising edges to real data
      for (auto &[channel, data] : circuitState.bufferChannelMap)
        circuitState.storeValue(channel, data);

    } while (manager.valueChangedThisCycle);

    // Reset rising edges state so they can start at next cycle
    circuitState.edgeRisenOps.clear();
    ++manager.currentCycle;
  }
}

//===----------------------------------------------------------------------===//
// Simulator entry point
//===----------------------------------------------------------------------===//

LogicalResult simulate(StringRef toplevelFunction,
                       ArrayRef<std::string> inputArgs,
                       mlir::OwningOpRef<mlir::ModuleOp> &module,
                       mlir::MLIRContext &,
                       llvm::StringMap<std::string> &funcMap) {
  // The store associates each allocation in the program
  // (represented by a int) with a vector of values which can be
  // accessed by it.  Currently values are assumed to be an integer.
  std::vector<std::vector<Any>> store;

  // Handler for circuit states
  CircuitState circuitState;
  circuitState.channelMap = ChannelMap{};

  // We need three things in a function-type independent way.
  // The type signature of the function.
  mlir::FunctionType ftype;
  // The arguments of the entry block.
  mlir::Block::BlockArgListType blockArgs;

  // The number of inputs to the function in the IR.
  unsigned inputs;
  unsigned outputs;
  // The number of 'real' inputs.  This avoids the dummy input
  // associated with the handshake control logic for handshake
  // functions.
  unsigned realInputs;
  unsigned realOutputs;

  // Fill the model map with instancied model structures
  // initialiseMapp
  if (initialiseMap(funcMap, models).failed())
    return failure();

  if (circt::handshake::FuncOp toplevel =
          module->lookupSymbol<circt::handshake::FuncOp>(toplevelFunction)) {
    ftype = toplevel.getFunctionType();
    mlir::Block &entryBlock = toplevel.getBody().front();
    blockArgs = entryBlock.getArguments();

    // Get the primary inputs of toplevel off the command line.
    inputs = toplevel.getNumArguments();
    realInputs = inputs - 1;
    outputs = toplevel.getNumResults();
    realOutputs = outputs - 1;
    if (inputs == 0) {
      errs() << "Function " << toplevelFunction << " is expected to have "
             << "at least one dummy argument.\n";
      return failure();
    }
    if (outputs == 0) {
      errs() << "Function " << toplevelFunction << " is expected to have "
             << "at least one dummy result.\n";
      return failure();
    }
    // Implicit none argument
    APInt apnonearg(1, 0);
    circuitState.storeValue(blockArgs[blockArgs.size() - 1], apnonearg);
  } else
    llvm::report_fatal_error("Function '" + toplevelFunction +
                             "' not supported");

  if (inputArgs.size() != realInputs) {
    errs() << "Toplevel function " << toplevelFunction << " has " << realInputs
           << " actual arguments, but " << inputArgs.size()
           << " arguments were provided on the command line.\n";
    return failure();
  }

  for (unsigned i = 0; i < realInputs; ++i) {
    mlir::Type type = ftype.getInput(i);
    if (type.isa<mlir::MemRefType>()) {
      // We require this memref type to be fully specified.
      auto memreftype = type.dyn_cast<mlir::MemRefType>();
      std::vector<Any> nothing;
      std::string x;
      unsigned buffer;
      if (allocateMemRef(memreftype, nothing, store, buffer).failed())
        return failure();
      circuitState.storeValue(blockArgs[i], buffer);
      int64_t pos = 0;
      std::stringstream arg(inputArgs[i]);
      while (!arg.eof()) {
        getline(arg, x, ',');
        store[buffer][pos++] =
            readValueWithType(memreftype.getElementType(), x);
      }
    } else {
      Any value = readValueWithType(type, inputArgs[i]);
      circuitState.storeValue(blockArgs[i], value);
    }
  }

  std::vector<Any> results(realOutputs);
  bool succeeded = false;
  if (circt::handshake::FuncOp toplevel =
          module->lookupSymbol<circt::handshake::FuncOp>(toplevelFunction)) {
    succeeded = HandshakeExecuter(toplevel, circuitState, results, store,
                                  module, models)
                    .succeeded();

    outs() << "Finished execution\n";
  }

  if (!succeeded)
    return failure();

  // Go back through the arguments and output any memrefs.
  for (unsigned i = 0; i < realInputs; ++i) {
    mlir::Type type = ftype.getInput(i);
    if (type.isa<mlir::MemRefType>()) {
      // We require this memref type to be fully specified.
      auto memreftype = type.dyn_cast<mlir::MemRefType>();
      unsigned buffer =
          any_cast<unsigned>(*(circuitState.channelMap[blockArgs[i]].data));
      auto elementType = memreftype.getElementType();
      for (int j = 0; j < memreftype.getNumElements(); ++j) {
        if (j != 0)
          outs() << ",";
        printAnyValueWithType(outs(), elementType, store[buffer][j]);
      }
      outs() << " ";
    }
  }
  outs() << "\n";

  return success();
}

} // namespace experimental
} // namespace dynamatic
