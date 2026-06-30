#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Any.h"
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

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
  LogicalResult execute(mlir::arith::ConstantOp, std::vector<Any> &,
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
  LogicalResult execute(mlir::arith::ShRUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::TruncIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::TruncFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::AndIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::XOrIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::AddFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::SIToFPOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::UIToFPOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::FPToSIOp, std::vector<Any> &,
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
  LogicalResult execute(mlir::arith::NegFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::RemFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::RemSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::RemUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::IndexCastOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ExtSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ExtUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ExtFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::CosOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::ExpOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::Exp2Op, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::LogOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::Log2Op, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::Log10Op, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::SinOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::SqrtOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::math::AbsFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::MaxUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::MaxSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::MinSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::MinUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(memref::LoadOp, std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(memref::StoreOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(memref::AllocOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(memref::AllocaOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(memref::GetGlobalOp, std::vector<Any> &,
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

mlir::LogicalResult simulate(mlir::func::FuncOp funcOp,
                             llvm::ArrayRef<std::string> inputArgs,
                             StdProfiler &prof);
