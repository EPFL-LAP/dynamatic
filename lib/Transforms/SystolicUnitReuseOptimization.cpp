#include "dynamatic/Transforms/SystolicUnitReuseOptimization.h"
#include "dynamatic/Analysis/NumericAnalysis.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <iterator>
#include <z3++.h>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

struct SystolicUnitReuseOptimizationPass
    : public dynamatic::impl::SystolicUnitReuseOptimizationBase<
          SystolicUnitReuseOptimizationPass> {

  void runDynamaticPass() override {

    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder builder(ctx);

    // Iterate over all functions in the module
    // and identify the memory loaded by systolic units
    // and remove the loads if there are multiple loads
    for (auto func : module.getOps<func::FuncOp>()) {
      if (failed(optimizeSystolicUnitLoads(func, builder))) {
        return signalPassFailure();
      }
    }
  };

private:
  // Function to optimize loads performed by systolic units
  LogicalResult optimizeSystolicUnitLoads(func::FuncOp func,
                                          OpBuilder &builder);
};
} // namespace

Value getWestInputSU(Operation *suOp) { return suOp->getOperand(0); }
Value getNorthInputSU(Operation *suOp) { return suOp->getOperand(1); }
Value setWestInputSU(Operation *suOp, Value v) {
  suOp->setOperand(0, v);
  return suOp->getOperand(0);
}
Value setNorthInputSU(Operation *suOp, Value v) {
  suOp->setOperand(1, v);
  return suOp->getOperand(1);
}
Value getEastOutputSU(Operation *suOp) { return suOp->getResult(1); }
Value getSouthOutputSU(Operation *suOp) { return suOp->getResult(2); }

// Function to recursively collect all predecessors of a value
void collectPredecessors(Value v, llvm::DenseSet<Value> &preds) {
  if (!v)
    return;

  // If block argument, loop induction variable or constant, stop
  if (v.isa<BlockArgument>() || v.getDefiningOp<arith::ConstantOp>() ||
      v.getDefiningOp<affine::AffineForOp>()) {
    preds.insert(v);
    return;
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return;

  for (Value opV : defOp->getOperands()) {
    collectPredecessors(opV, preds);
  }
}

// Function to create a Z3 expression from an MLIR value
LogicalResult createZ3Expr(z3::context &z3_ctx, Value v,
                           const llvm::DenseMap<Value, z3::expr> &z3_vars,
                           z3::expr &out_expr) {
  if (!v) {
    llvm::errs() << "[ERROR] Null value in createZ3Expr\n";
    return failure();
  }

  // If block argument or induction variable, return the corresponding Z3 var
  if (auto arg = v.dyn_cast<BlockArgument>()) {
    auto it = z3_vars.find(arg);
    if (it == z3_vars.end()) {
      llvm::errs() << "[ERROR] Block argument not found in Z3 vars\n";
      return failure();
    }
    out_expr = it->second;
    return success();
  }
  if (auto forOp = v.getDefiningOp<affine::AffineForOp>()) {
    auto it = z3_vars.find(forOp.getInductionVar());
    if (it == z3_vars.end()) {
      llvm::errs() << "[ERROR] Induction variable not found in Z3 vars\n";
      return failure();
    }
    out_expr = it->second;
    return success();
  }

  // If constant, return its integer value
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
      out_expr = z3_ctx.int_val(intAttr.getInt());
      return success();
    } else {
      llvm::errs() << "[ERROR] Unsupported constant type for Z3 expr\n";
      return failure();
    }
  }

  // If addition, subtraction, multiplication, or division, recursively create
  // Z3 exprs
  if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
    z3::expr lhs = z3_ctx.int_val(0);
    z3::expr rhs = z3_ctx.int_val(0);
    if (failed(createZ3Expr(z3_ctx, addOp.getLhs(), z3_vars, lhs)) ||
        failed(createZ3Expr(z3_ctx, addOp.getRhs(), z3_vars, rhs))) {
      return failure();
    }
    out_expr = lhs + rhs;
    return success();
  }

  // Unhandled operation which should be added support for
  llvm::errs() << "[ERROR] Unsupported operation in createZ3Expr: "
               << *v.getDefiningOp() << "\n";
  return failure();
}

// Create a Z3 integer variable for each BlockArgument and AffineFor
// induction variable in vars, and populate the map z3_vars
LogicalResult createZ3Vars(z3::context &z3_ctx,
                           const llvm::DenseSet<Value> &vars,
                           llvm::DenseMap<Value, z3::expr> &z3_vars) {
  int id_iv = 0;
  // Create a Z3 integer variable for each BlockArgument and AffineFor
  // induction variable
  for (Value v : vars) {
    std::string var_name;
    if (auto arg = v.dyn_cast<BlockArgument>()) {
      var_name = "arg" + std::to_string(arg.getArgNumber());
    } else if (auto forOp = v.getDefiningOp<affine::AffineForOp>()) {
      var_name = "iv" + std::to_string(id_iv++);
    } else {
      // Unsupported variable type
      llvm::errs() << "[ERROR] Unsupported variable type for Z3 var\n";
      return failure();
    }
    z3::expr z3_var = z3_ctx.int_const(var_name.c_str());
    z3_vars.insert({v, z3_var});
  }
  return success();
}

// Compare two MLIR values v1 and v2 for equivalence using an SMT solver
LogicalResult areEquivalentValues(mlir::Value v1, mlir::Value v2,
                                  MLIRContext *ctx, bool &equivalent) {
  if (v1 == v2) {
    equivalent = true;
    return success();
  }

  llvm::DenseSet<Value> preds1, preds2;
  collectPredecessors(v1, preds1);
  collectPredecessors(v2, preds2);
  // Collect BlockArguments, AffineFor induction variables for both and check
  // if they are the same
  llvm::DenseSet<Value> vars1, vars2;
  for (auto p : preds1) {
    if (p.isa<BlockArgument>() || p.getDefiningOp<affine::AffineForOp>())
      vars1.insert(p);
  }
  for (auto p : preds2) {
    if (p.isa<BlockArgument>() || p.getDefiningOp<affine::AffineForOp>())
      vars2.insert(p);
  }
  if (vars1 != vars2) {
    equivalent = false;
    return success();
  }

  // If they have the same predecessors, write the symbolic expressions
  // for both values and check if they are the same
  // First create a variable for each predecessor
  z3::context z3_ctx;
  llvm::DenseMap<Value, z3::expr> z3_vars;
  if (failed(createZ3Vars(z3_ctx, vars1, z3_vars))) {
    llvm::errs() << "[ERROR] Failed to create Z3 variables\n";
    return failure();
  }
  // Now create the symbolic expression for both values
  // by recursively substituting the operands with their expressions
  z3::expr expr1 = z3_ctx.int_val(0);
  z3::expr expr2 = z3_ctx.int_val(0);
  if (failed(createZ3Expr(z3_ctx, v1, z3_vars, expr1)) ||
      failed(createZ3Expr(z3_ctx, v2, z3_vars, expr2))) {
    llvm::errs() << "[ERROR] Failed to create Z3 expressions\n";
    return failure();
  }
  // Then, run Z3 SMT solver to check if expr1 == expr2
  // If they are equivalent, return true
  // Otherwise return false
  z3::solver solver(z3_ctx);
  solver.add(expr1 != expr2);
  if (solver.check() == z3::unsat) {
    equivalent = true;
    return success();
  }
  equivalent = false;
  return success();
}

// Compare two vector::TransferReadOps for equivalence using an SMT solver
LogicalResult areEquivalentVectorReads(mlir::vector::TransferReadOp r1,
                                       mlir::vector::TransferReadOp r2,
                                       bool &equivalent) {
  // Check memrefs are the same SSA value
  if (r1.getSource() != r2.getSource()) {
    equivalent = false;
    return success();
  }

  // Check element type and vector shape match
  if (r1.getVectorType() != r2.getVectorType()) {
    equivalent = false;
    return success();
  }

  // Check number of indices match
  auto idxs1 = r1.getIndices();
  auto idxs2 = r2.getIndices();
  if (idxs1.size() != idxs2.size()) {
    equivalent = false;
    return success();
  }

  // Compare index expressions using SMT solver
  for (auto [v1, v2] : llvm::zip(idxs1, idxs2)) {
    if (failed(areEquivalentValues(v1, v2, r1->getParentOp()->getContext(),
                                   equivalent))) {
      return failure();
    }
    if (!equivalent)
      return success();
  }

  equivalent = true;
  return success();
}

// Insert a new SU in the map if its load operation is not equivalent to
// existing loads in the map susByLoad. If it is equivalent, add the SU to the
// existing list of SUs. The list of SUs is ordered by the index of their other
// load operation (if applicable).
LogicalResult
insertNewLoad(Value new_mat_vec, Operation *suOp,
              DenseMap<Value, SmallVector<Operation *, 4>> &susByLoad) {
  bool newLoad = true;
  // Iterate over existing groups and see if we can find a match
  for (auto &[vec_map, sus] : susByLoad) {
    bool areEq = false;
    if (failed(areEquivalentVectorReads(
            vec_map.getDefiningOp<vector::TransferReadOp>(),
            new_mat_vec.getDefiningOp<vector::TransferReadOp>(), areEq))) {
      return failure();
    }
    if (areEq) {
      // We found a match, add this SU to the group and break
      susByLoad[vec_map].push_back(suOp);
      newLoad = false;
      break;
    }
  }
  // If we did not find a match, create a new group
  if (newLoad) {
    susByLoad[new_mat_vec] = {suOp};
  }
  return success();
}

// Reuse the output of an SU as input of the following SU in the list sus
LogicalResult reuseSUsOutputs(SmallVector<Operation *, 4> &sus, bool eastOutput,
                              OpBuilder &builder) {
  // Iterate over the SUs in the group and connect their outputs
  for (size_t i = 1; i < sus.size(); i++) {
    Operation *prev_su = sus[i - 1];
    Operation *curr_su = sus[i];
    if (eastOutput) {
      Value east_out = getEastOutputSU(prev_su);
      setWestInputSU(curr_su, east_out);
    } else {
      Value south_out = getSouthOutputSU(prev_su);
      setNorthInputSU(curr_su, south_out);
    }
  }
  return success();
}

LogicalResult SystolicUnitReuseOptimizationPass::optimizeSystolicUnitLoads(
    func::FuncOp func, OpBuilder &builder) {

  // Find all fpsa_su operations in the function
  SmallVector<Operation *, 16> suOps;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "fpsa_su") {
      suOps.push_back(op);
    }
  });

  // Map to track already loaded values
  DenseMap<std::pair<Value, Value>, Value> loadedValues;

  // List of all SUs with the same A_vec load
  DenseMap<Value, SmallVector<Operation *, 4>> susByNorthLoad;
  // List of all SUs with the same B_vec load
  DenseMap<Value, SmallVector<Operation *, 4>> susByWestLoad;
  // Iterate over all SUs and group them by their loads
  for (auto suOp : suOps) {
    // Get the operands of the fpsa_su operation
    Value west_vec = getWestInputSU(suOp);
    Value north_vec = getNorthInputSU(suOp);
    // Iterate over the existing groups and see if we can find a match
    if (failed(insertNewLoad(north_vec, suOp, susByNorthLoad))) {
      return failure();
    }
    // Apply the same logic for B_vec
    if (failed(insertNewLoad(west_vec, suOp, susByWestLoad))) {
      return failure();
    }
  }

  // Connect SUs with the same vector load to reuse the loaded value
  for (auto &[vec_map, sus] : susByNorthLoad) {
    // If there is only one SU in this group, nothing to do
    if (sus.size() <= 1)
      continue;
    if (failed(reuseSUsOutputs(sus, false, builder))) {
      return failure();
    }
  }
  // Apply the same logic for B_vec
  for (auto &[vec_map, sus] : susByWestLoad) {
    // If there is only one SU in this group, nothing to do
    if (sus.size() <= 1)
      continue;
    if (failed(reuseSUsOutputs(sus, true, builder))) {
      return failure();
    }
  }

  return success();
}
namespace dynamatic {
std::unique_ptr<dynamatic::DynamaticPass>
createSystolicUnitReuseOptimization() {
  return std::make_unique<SystolicUnitReuseOptimizationPass>();
}
} // namespace dynamatic
