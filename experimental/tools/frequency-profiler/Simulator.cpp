//===- Simulator.cpp - std-level simulator ----------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a simulator that executes a restricted form of the std dialect.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <list>

#define DEBUG_TYPE "simulator"

STATISTIC(instructionsExecuted, "Instructions Executed");

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {

class StdExecuter {
public:
  StdExecuter(mlir::func::FuncOp &toplevel,
              llvm::DenseMap<mlir::Value, Any> &valueMap,
              llvm::DenseMap<mlir::Value, double> &timeMap,
              std::vector<Any> &results, std::vector<double> &resultTimes,
              std::vector<std::vector<Any>> &store,
              std::vector<double> &storeTimes, StdProfiler &prof);

  LogicalResult succeeded() const {
    return successFlag ? success() : failure();
  }

private:
  /// Operation execution visitors
  LogicalResult execute(mlir::arith::ConstantIndexOp,
                        std::vector<Any> & /*inputs*/,
                        std::vector<Any> & /*outputs*/);
  LogicalResult execute(mlir::arith::ConstantIntOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::AddIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::LLVM::UndefOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::SelectOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::OrIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ShLIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ShRSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::TruncIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::AndIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::XOrIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::AddFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::CmpIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::CmpFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::SubIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::SubFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::MulIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::MulFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::DivSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::DivUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::DivFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::IndexCastOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ExtSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ExtUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(memref::LoadOp, std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(memref::StoreOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(memref::AllocOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::cf::BranchOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::cf::CondBranchOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(func::ReturnOp, std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(mlir::CallOpInterface, std::vector<Any> &,
                        std::vector<Any> &);

private:
  /// Execution context variables.
  llvm::DenseMap<mlir::Value, Any> &valueMap;
  llvm::DenseMap<mlir::Value, double> &timeMap;
  std::vector<Any> &results;
  std::vector<double> &resultTimes;
  std::vector<std::vector<Any>> &store;
  std::vector<double> &storeTimes;
  double time;

  /// Profiler to gather statistics.
  StdProfiler &prof;

  /// Flag indicating whether execution was successful.
  bool successFlag = true;

  /// An iterator which walks over the instructions.
  mlir::Block::iterator instIter;
};

} // namespace

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

template <typename T>
static void fatalValueError(StringRef reason, T &value) {
  std::string err;
  llvm::raw_string_ostream os(err);
  os << reason << " ('";
  // Explicitly use ::print instead of << due to possible operator resolution
  // error between i.e., mlir::Operation::<< and operator<<(OStream &&OS,
  // const T &Value)
  value.print(os);
  os << "')\n";
  llvm::report_fatal_error(err.c_str());
}

static void debugArg(const std::string &head, mlir::Value op,
                     const APInt &value, double time) {
  LLVM_DEBUG(dbgs() << "  " << head << ":  " << op << " = " << value
                    << " (APInt<" << value.getBitWidth() << ">) @" << time
                    << "\n");
}

static void debugArg(const std::string &head, mlir::Value op, const Any &value,
                     double time) {
  if (auto *val = any_cast<APInt>(&value)) {
    debugArg(head, op, *val, time);
  } else if (auto *val = any_cast<APFloat>(&value)) {
    debugArg(head, op, val, time);
  } else if (auto *val = any_cast<unsigned>(&value)) {
    // Represents an allocated buffer.
    LLVM_DEBUG(dbgs() << "  " << head << ":  " << op << " = Buffer " << *val
                      << "\n");
  } else {
    llvm_unreachable("unknown type");
  }
}

static Any readValueWithType(mlir::Type type, std::stringstream &arg) {
  if (type.isIndex()) {
    int64_t x;
    arg >> x;
    int64_t width = IndexType::kInternalStorageBitWidth;
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
  assert(false && "unknown argument type!");
  return {};
}

static Any readValueWithType(mlir::Type type, const std::string &in) {
  std::stringstream stream(in);
  return readValueWithType(type, stream);
}

// Allocate a new matrix with dimensions given by the type, in the
// given store.  Return the pseudo-pointer to the new matrix in the
// store (i.e. the first dimension index).
static unsigned allocateMemRef(mlir::MemRefType type, std::vector<Any> &in,
                               std::vector<std::vector<Any>> &store,
                               std::vector<double> &storeTimes) {
  ArrayRef<int64_t> shape = type.getShape();
  int64_t allocationSize = 1;
  unsigned count = 0;
  for (int64_t dim : shape) {
    if (dim > 0)
      allocationSize *= dim;
    else {
      assert(count < in.size());
      allocationSize *= any_cast<APInt>(in[count++]).getSExtValue();
    }
  }
  unsigned ptr = store.size();
  store.resize(ptr + 1);
  storeTimes.resize(ptr + 1);
  store[ptr].resize(allocationSize);
  storeTimes[ptr] = 0.0;
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
  return ptr;
}

//===----------------------------------------------------------------------===//
// Std executer
//===----------------------------------------------------------------------===//

LogicalResult StdExecuter::execute(mlir::arith::ConstantIndexOp op,
                                   std::vector<Any> &, std::vector<Any> &out) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue().sextOrTrunc(IndexType::kInternalStorageBitWidth);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::ConstantIntOp op,
                                   std::vector<Any> &, std::vector<Any> &out) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue();
  return success();
}

LogicalResult StdExecuter::execute(mlir::LLVM::UndefOp op, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  auto type = op.getRes().getType();
  if (isa<IndexType>(type)) {
    APInt val(IndexType::kInternalStorageBitWidth, 0);
    out[0] = val;
  } else if (isa<mlir::IntegerType>(type)) {
    APInt val(type.getIntOrFloatBitWidth(), 0);
    out[0] = val;
  } else if (isa<FloatType>(type)) {
    APFloat val(0.0);
    out[0] = val;
  } else
    return op.emitError() << "Unsupproted undef result type";
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::SelectOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) != 0 ? in[1] : in[2];
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::ShLIOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  auto toShift = any_cast<APInt>(in[0]).getSExtValue();
  auto shiftAmount = any_cast<APInt>(in[1]).getZExtValue();
  auto shifted =
      APInt(any_cast<APInt>(in[0]).getBitWidth(), toShift << shiftAmount);
  out[0] = shifted;
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::ShRSIOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  auto toShift = any_cast<APInt>(in[0]).getSExtValue();
  auto shiftAmount = any_cast<APInt>(in[1]).getZExtValue();
  auto shifted =
      APInt(any_cast<APInt>(in[0]).getBitWidth(), toShift >> shiftAmount);
  out[0] = shifted;
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::OrIOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) | any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::TruncIOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  auto width = dyn_cast<mlir::IntegerType>(op.getResult().getType()).getWidth();
  out[0] = any_cast<APInt>(in[0]).trunc(width);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::AndIOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) & any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::XOrIOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) ^ any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::AddIOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) + any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::AddFOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) + any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::CmpIOp op, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  APInt in0 = any_cast<APInt>(in[0]);
  APInt in1 = any_cast<APInt>(in[1]);
  APInt out0(1, mlir::arith::applyCmpPredicate(op.getPredicate(), in0, in1));
  out[0] = out0;
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::CmpFOp op, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  APFloat in0 = any_cast<APFloat>(in[0]);
  APFloat in1 = any_cast<APFloat>(in[1]);
  APInt out0(1, mlir::arith::applyCmpPredicate(op.getPredicate(), in0, in1));
  out[0] = out0;
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::SubIOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) - any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::SubFOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) + any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::MulIOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) * any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::MulFOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) * any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::DivSIOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  if (!any_cast<APInt>(in[1]).getZExtValue())
    return op.emitOpError() << "Division By Zero!";

  out[0] = any_cast<APInt>(in[0]).sdiv(any_cast<APInt>(in[1]));
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::DivUIOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  if (!any_cast<APInt>(in[1]).getZExtValue())
    return op.emitOpError() << "Division By Zero!";
  out[0] = any_cast<APInt>(in[0]).udiv(any_cast<APInt>(in[1]));
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::DivFOp, std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) / any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::IndexCastOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  mlir::Type outType = op.getOut().getType();
  APInt inValue = any_cast<APInt>(in[0]);
  APInt outValue;
  if (outType.isIndex())
    outValue =
        APInt(IndexType::kInternalStorageBitWidth, inValue.getZExtValue());
  else if (outType.isIntOrFloat())
    outValue = APInt(outType.getIntOrFloatBitWidth(), inValue.getZExtValue());
  else {
    return op.emitOpError() << "unhandled output type";
  }

  out[0] = outValue;
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::ExtSIOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  int64_t width = op.getType().getIntOrFloatBitWidth();
  out[0] = any_cast<APInt>(in[0]).sext(width);
  return success();
}

LogicalResult StdExecuter::execute(mlir::arith::ExtUIOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  int64_t width = op.getType().getIntOrFloatBitWidth();
  out[0] = any_cast<APInt>(in[0]).zext(width);
  return success();
}

LogicalResult StdExecuter::execute(mlir::memref::LoadOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  ArrayRef<int64_t> shape = op.getMemRefType().getShape();
  unsigned address = 0;
  for (unsigned i = 0; i < shape.size(); ++i) {
    address = address * shape[i] + any_cast<APInt>(in[i + 1]).getZExtValue();
  }
  unsigned ptr = any_cast<unsigned>(in[0]);
  if (ptr >= store.size())
    return op.emitOpError()
           << "Unknown memory identified by pointer '" << ptr << "'";

  auto &ref = store[ptr];
  if (address >= ref.size())
    return op.emitOpError()
           << "Out-of-bounds access to memory '" << ptr << "'. Memory has "
           << ref.size() << " elements but requested element " << address;

  Any result = ref[address];
  out[0] = result;

  double storeTime = storeTimes[ptr];
  LLVM_DEBUG(dbgs() << "STORE: " << storeTime << "\n");
  time = std::max(time, storeTime);
  storeTimes[ptr] = time;
  return success();
}

LogicalResult StdExecuter::execute(mlir::memref::StoreOp op,
                                   std::vector<Any> &in, std::vector<Any> &) {
  ArrayRef<int64_t> shape = op.getMemRefType().getShape();
  unsigned address = 0;
  for (unsigned i = 0; i < shape.size(); ++i) {
    address = address * shape[i] + any_cast<APInt>(in[i + 2]).getZExtValue();
  }
  unsigned ptr = any_cast<unsigned>(in[1]);
  if (ptr >= store.size())
    return op.emitOpError()
           << "Unknown memory identified by pointer '" << ptr << "'";
  auto &ref = store[ptr];
  if (address >= ref.size())
    return op.emitOpError()
           << "Out-of-bounds access to memory '" << ptr << "'. Memory has "
           << ref.size() << " elements but requested element " << address;
  ref[address] = in[0];

  double storeTime = storeTimes[ptr];
  LLVM_DEBUG(dbgs() << "STORE: " << storeTime << "\n");
  time = std::max(time, storeTime);
  storeTimes[ptr] = time;
  return success();
}

LogicalResult StdExecuter::execute(mlir::memref::AllocOp op,
                                   std::vector<Any> &in,
                                   std::vector<Any> &out) {
  out[0] = allocateMemRef(op.getType(), in, store, storeTimes);
  unsigned ptr = any_cast<unsigned>(out[0]);
  storeTimes[ptr] = time;
  return success();
}

LogicalResult StdExecuter::execute(mlir::cf::BranchOp branchOp,
                                   std::vector<Any> &in, std::vector<Any> &) {
  mlir::Block *dest = branchOp.getDest();
  for (auto out : enumerate(dest->getArguments())) {
    LLVM_DEBUG(debugArg("ARG", out.value(), in[out.index()], time));
    valueMap[out.value()] = in[out.index()];
    timeMap[out.value()] = time;
  }
  prof.transitions[std::make_pair(branchOp->getBlock(), dest)]++;
  instIter = dest->begin();
  return success();
}

LogicalResult StdExecuter::execute(mlir::cf::CondBranchOp condBranchOp,
                                   std::vector<Any> &in, std::vector<Any> &) {
  APInt condition = any_cast<APInt>(in[0]);
  mlir::Block *dest;
  std::vector<Any> inArgs;
  double time = 0.0;
  if (condition != 0) {
    dest = condBranchOp.getTrueDest();
    inArgs.resize(condBranchOp.getNumTrueOperands());
    for (auto in : enumerate(condBranchOp.getTrueOperands())) {
      inArgs[in.index()] = valueMap[in.value()];
      time = std::max(time, timeMap[in.value()]);
      LLVM_DEBUG(
          debugArg("IN", in.value(), inArgs[in.index()], timeMap[in.value()]));
    }
  } else {
    dest = condBranchOp.getFalseDest();
    inArgs.resize(condBranchOp.getNumFalseOperands());
    for (auto in : enumerate(condBranchOp.getFalseOperands())) {
      inArgs[in.index()] = valueMap[in.value()];
      time = std::max(time, timeMap[in.value()]);
      LLVM_DEBUG(
          debugArg("IN", in.value(), inArgs[in.index()], timeMap[in.value()]));
    }
  }
  for (auto out : enumerate(dest->getArguments())) {
    LLVM_DEBUG(debugArg("ARG", out.value(), inArgs[out.index()], time));
    valueMap[out.value()] = inArgs[out.index()];
    timeMap[out.value()] = time;
  }
  instIter = dest->begin();
  prof.transitions[std::make_pair(condBranchOp->getBlock(), dest)]++;
  return success();
}

LogicalResult StdExecuter::execute(func::ReturnOp op, std::vector<Any> &in,
                                   std::vector<Any> &) {
  for (unsigned i = 0; i < results.size(); ++i) {
    results[i] = in[i];
    resultTimes[i] = timeMap[op.getOperand(i)];
  }
  return success();
}

LogicalResult StdExecuter::execute(mlir::CallOpInterface callOp,
                                   std::vector<Any> &in, std::vector<Any> &) {
  // implement function calls.
  auto *op = callOp.getOperation();
  mlir::Operation *calledOp = callOp.resolveCallable();
  if (auto funcOp = dyn_cast<mlir::func::FuncOp>(calledOp)) {
    unsigned outputs = funcOp.getNumResults();
    llvm::DenseMap<mlir::Value, Any> newValueMap;
    llvm::DenseMap<mlir::Value, double> newTimeMap;
    std::vector<Any> results(outputs);
    std::vector<double> resultTimes(outputs);
    std::vector<std::vector<Any>> store;
    std::vector<double> storeTimes;
    mlir::Block &entryBlock = funcOp.getBody().front();
    mlir::Block::BlockArgListType blockArgs = entryBlock.getArguments();

    for (auto inIt : enumerate(in)) {
      newValueMap[blockArgs[inIt.index()]] = inIt.value();
      newTimeMap[blockArgs[inIt.index()]] =
          timeMap[op->getOperand(inIt.index())];
    }
    StdExecuter(funcOp, newValueMap, newTimeMap, results, resultTimes, store,
                storeTimes, prof);
    for (auto out : enumerate(op->getResults())) {
      valueMap[out.value()] = results[out.index()];
      timeMap[out.value()] = resultTimes[out.index()];
    }
    ++instIter;
  } else
    return op->emitOpError() << "Callable was not a function";

  return success();
}
enum ExecuteStrategy { Default = 1 << 0, Continue = 1 << 1, Return = 1 << 2 };

StdExecuter::StdExecuter(mlir::func::FuncOp &toplevel,
                         llvm::DenseMap<mlir::Value, Any> &valueMap,
                         llvm::DenseMap<mlir::Value, double> &timeMap,
                         std::vector<Any> &results,
                         std::vector<double> &resultTimes,
                         std::vector<std::vector<Any>> &store,
                         std::vector<double> &storeTimes, StdProfiler &prof)
    : valueMap(valueMap), timeMap(timeMap), results(results),
      resultTimes(resultTimes), store(store), storeTimes(storeTimes),
      prof(prof) {
  successFlag = true;
  mlir::Block &entryBlock = toplevel.getBody().front();
  instIter = entryBlock.begin();

  // Main executive loop.  Start at the first instruction of the entry
  // block.  Fetch and execute instructions until we hit a terminator.
  while (true) {
    mlir::Operation &op = *instIter;
    std::vector<Any> inValues(op.getNumOperands());
    std::vector<Any> outValues(op.getNumResults());
    LLVM_DEBUG(dbgs() << "OP:  " << op.getName() << "\n");
    time = 0.0;
    for (auto in : enumerate(op.getOperands())) {
      inValues[in.index()] = valueMap[in.value()];
      time = std::max(time, timeMap[in.value()]);
      LLVM_DEBUG(debugArg("IN", in.value(), inValues[in.index()],
                          timeMap[in.value()]));
    }

    unsigned strat = ExecuteStrategy::Default;
    auto res =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<mlir::arith::ConstantIndexOp, mlir::arith::ConstantIntOp,
                  mlir::arith::AddIOp, mlir::arith::AddFOp, mlir::arith::CmpIOp,
                  mlir::arith::CmpFOp, mlir::arith::SubIOp, mlir::arith::SubFOp,
                  mlir::arith::MulIOp, mlir::arith::MulFOp,
                  mlir::arith::DivSIOp, mlir::arith::DivUIOp,
                  mlir::arith::DivFOp, mlir::arith::IndexCastOp,
                  mlir::arith::TruncIOp, mlir::arith::AndIOp,
                  mlir::arith::OrIOp, mlir::arith::XOrIOp,
                  mlir::arith::SelectOp, mlir::LLVM::UndefOp,
                  mlir::arith::ShRSIOp, mlir::arith::ShLIOp,
                  mlir::arith::ExtSIOp, mlir::arith::ExtUIOp, memref::AllocOp,
                  memref::LoadOp, memref::StoreOp>([&](auto op) {
              strat = ExecuteStrategy::Default;
              return execute(op, inValues, outValues);
            })
            .Case<mlir::cf::BranchOp, mlir::cf::CondBranchOp,
                  mlir::CallOpInterface>([&](auto op) {
              strat = ExecuteStrategy::Continue;
              return execute(op, inValues, outValues);
            })
            .Case<func::ReturnOp>([&](auto op) {
              strat = ExecuteStrategy::Return;
              return execute(op, inValues, outValues);
            })
            .Default([](auto op) {
              return op->emitOpError() << "Unknown operation!";
            });

    if (res.failed()) {
      successFlag = false;
      return;
    }

    if (strat & ExecuteStrategy::Continue)
      continue;

    if (strat & ExecuteStrategy::Return)
      return;

    for (auto out : enumerate(op.getResults())) {
      LLVM_DEBUG(debugArg("OUT", out.value(), outValues[out.index()], time));
      valueMap[out.value()] = outValues[out.index()];
      timeMap[out.value()] = time + 1;
    }
    ++instIter;
    ++instructionsExecuted;
  }
}

LogicalResult simulate(func::FuncOp funcOp, ArrayRef<std::string> inputArgs,
                       StdProfiler &prof) {
  // The store associates each allocation in the program
  // (represented by a int) with a vector of values which can be
  // accessed by it.   Currently values are assumed to be an integer.
  std::vector<std::vector<Any>> store;
  std::vector<double> storeTimes;

  // The valueMap associates each SSA statement in the program
  // (represented by a Value*) with it's corresponding value.
  // Currently the value is assumed to be an integer.
  llvm::DenseMap<mlir::Value, Any> valueMap;

  // The timeMap associates each value with the time it was created.
  llvm::DenseMap<mlir::Value, double> timeMap;

  // The type signature of the function.
  mlir::FunctionType ftype;
  // The arguments of the entry block.
  mlir::Block::BlockArgListType blockArgs;

  unsigned numInputs = funcOp.getNumArguments();
  unsigned numOutputs = funcOp.getNumResults();

  ftype = funcOp.getFunctionType();
  mlir::Block &entryBlock = funcOp.getBody().front();
  blockArgs = entryBlock.getArguments();

  if (inputArgs.size() != numInputs) {
    errs() << "Toplevel function " << funcOp->getName() << " has " << numInputs
           << " actual arguments, but " << inputArgs.size()
           << " arguments were provided on the command line.\n";
    return failure();
  }

  for (unsigned i = 0; i < numInputs; ++i) {
    mlir::Type type = ftype.getInput(i);
    if (type.isa<mlir::MemRefType>()) {
      // We require this memref type to be fully specified.
      auto memreftype = type.dyn_cast<mlir::MemRefType>();
      std::vector<Any> nothing;
      std::string x;
      unsigned buffer = allocateMemRef(memreftype, nothing, store, storeTimes);
      valueMap[blockArgs[i]] = buffer;
      timeMap[blockArgs[i]] = 0.0;
      std::stringstream arg(inputArgs[i]);
      int64_t j = 0;
      while (!arg.eof()) {
        getline(arg, x, ',');
        store[buffer][j++] = readValueWithType(memreftype.getElementType(), x);
      }
    } else {
      Any value = readValueWithType(type, inputArgs[i]);
      valueMap[blockArgs[i]] = value;
      timeMap[blockArgs[i]] = 0.0;
    }
  }

  std::vector<Any> results(numOutputs);
  std::vector<double> resultTimes(numOutputs);

  return StdExecuter(funcOp, valueMap, timeMap, results, resultTimes, store,
                     storeTimes, prof)
      .succeeded();
}
