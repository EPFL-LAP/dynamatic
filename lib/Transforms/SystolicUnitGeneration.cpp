#include "dynamatic/Transforms/SystolicUnitGeneration.h"
#include "dynamatic/Analysis/NumericAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/ScfRotateForLoops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <iterator>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

struct SystolicUnitGenerationPass
    : public dynamatic::impl::SystolicUnitGenerationBase<
          SystolicUnitGenerationPass> {

  void runDynamaticPass() override {

    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder builder(ctx);

    // Iterate over all functions in the module
    // and insert systolic array loops in each function
    // that calls the fpsaMatMulName function
    for (auto func : module.getOps<func::FuncOp>()) {
      if (failed(insertSystolicArrayLoop(func, module, builder))) {
        return signalPassFailure();
      }
    }
    // Remove the fpsaMatMulName function if it is unused
    llvm::SmallVector<func::FuncOp, 4> funcsToErase;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.getName() == fpsaMatMulName) {
        if (!func.use_empty()) {
          llvm::errs() << "[ERROR] " << fpsaMatMulName
                       << " function is still used but all its calls should "
                          "have been removed by this pass\n";
          return signalPassFailure();
        }
        funcsToErase.push_back(func);
      }
    }
    for (auto func : funcsToErase) {
      func.erase();
    }
  };

private:
  std::string fpsaMatMulName = "fpsa_mat_mul";
  std::string fpsaSuName = "fpsa_su";
  // Function to insert loops representing systolic array structure
  LogicalResult insertSystolicArrayLoop(func::FuncOp func, ModuleOp module,
                                        OpBuilder &builder);
  // Function to insert a single systolic unit
  LogicalResult insertSystolicUnit(OpBuilder builder, Location loc, Value A_row,
                                   Value B_col, Value OUT, int numCols,
                                   int SU_x, int SU_y, Value i_indvar,
                                   Value j_indvar, int ROWS_SU, int COLS_SU);

  // Function to extract parameters of matrix multiplication from a function
  // call
  LogicalResult extractMatMulParams(func::FuncOp func, func::CallOp fpsaCallOp,
                                    Value &A_mat, Value &B_mat, Value &OUT_mat,
                                    int &SA_ROWS, int &SA_COLS, int &A_rows,
                                    int &A_cols, int &B_cols);

  // Function to transpose a matrix if needed
  LogicalResult transposeMatrix(OpBuilder &builder, Location loc,
                                func::FuncOp funcOp, Value matrix,
                                Value &transposedMatrix);
};
} // namespace

LogicalResult SystolicUnitGenerationPass::insertSystolicArrayLoop(
    func::FuncOp func, ModuleOp module, OpBuilder &builder) {

  // Find a function call of `fpsa_mat_mul` to replace with systolic array loops
  // and to extract the parameters
  func::CallOp fpsaCallOp = nullptr;
  func.walk([&](func::CallOp callOp) {
    if (callOp.getCallee() == fpsaMatMulName) {
      fpsaCallOp = callOp;
    }
  });
  if (!fpsaCallOp)
    return success();

  // Get the block in which the function call fpsaCallOp resides
  Block *callBlock = fpsaCallOp->getBlock();
  if (!callBlock) {
    llvm::errs() << "[ERROR] Expected call operation to be in a block\n";
    return failure();
  }

  Value A_mat, B_mat, OUT_mat;
  int SA_ROWS, SA_COLS;
  int A_rows, A_cols, B_cols;
  // Extract the parameters of the matrix multiplication
  if (failed(extractMatMulParams(func, fpsaCallOp, A_mat, B_mat, OUT_mat,
                                 SA_ROWS, SA_COLS, A_rows, A_cols, B_cols))) {
    return failure();
  }
  // Compute ROWS_SU and COLS_SU
  int ROWS_SU = A_rows / SA_ROWS;
  int COLS_SU = B_cols / SA_COLS;

  if (A_rows % SA_ROWS != 0 || B_cols % SA_COLS != 0) {
    llvm::errs()
        << "[ERROR] Matrix dimensions are not divisible by systolic array "
           "dimensions\n";
    return failure();
  }

  // Insert the systolic array loops at the beginning of the function
  builder.setInsertionPointToStart(callBlock);
  Location loc = builder.getUnknownLoc();

  // === Outer loop (i) ===
  auto forI = builder.create<affine::AffineForOp>(loc, 0, ROWS_SU);
  builder.setInsertionPointToStart(forI.getBody());
  Value i = forI.getInductionVar();

  // === Inner loop (j) ===
  auto forJ = builder.create<affine::AffineForOp>(loc, 0, COLS_SU);
  builder.setInsertionPointToStart(forJ.getBody());
  Value j = forJ.getInductionVar();

  // Transpose B_mat to allow consecutive memory accesses for matrix B
  Value B_mat_transposed = nullptr;
  if (failed(transposeMatrix(builder, loc, func, B_mat, B_mat_transposed))) {
    return failure();
  }
  // Replace B_mat with B_mat_transposed if transposition was performed
  if (B_mat_transposed) {
    B_mat = B_mat_transposed;
  }

  for (int su_x = 0; su_x < SA_COLS; su_x++) {
    for (int su_y = 0; su_y < SA_ROWS; su_y++) {
      if (failed(insertSystolicUnit(builder, loc, A_mat, B_mat, OUT_mat, A_cols,
                                    su_x, su_y, i, j, ROWS_SU, COLS_SU))) {
        return failure();
      }
    }
  }

  // builder.create<affine::AffineYieldOp>(loc);

  return success();
}

static int uniqueSUId = 0;

LogicalResult SystolicUnitGenerationPass::insertSystolicUnit(
    OpBuilder builder, Location loc, Value A_mat, Value B_mat, Value OUT,
    int numCols, int SU_x, int SU_y, Value i_indvar, Value j_indvar,
    int ROWS_SU, int COLS_SU) {

  auto ctx = builder.getContext();
  auto i32Type = IntegerType::get(builder.getContext(), 32);
  auto i32VecType = VectorType::get({numCols}, i32Type);
  Value numColsConst = builder.create<arith::ConstantOp>(
      loc, i32Type, builder.getI32IntegerAttr(numCols));
  // R_SU_y = SU_y * ROWS_SU + i
  auto constR_SU_y =
      builder.create<arith::ConstantIndexOp>(loc, SU_y * ROWS_SU);
  Value R_SU_y = builder.create<arith::AddIOp>(loc, constR_SU_y, i_indvar);

  // C_SU_x = SU_x * COLS_SU + j
  auto constC_SU_x =
      builder.create<arith::ConstantIndexOp>(loc, SU_x * COLS_SU);
  Value C_SU_x = builder.create<arith::AddIOp>(loc, constC_SU_x, j_indvar);

  // SU_west = A_mat[R_SU_y][:] = A_mat[R_SU_y][0:numCols]
  // Column start = 0
  Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);

  // Padding value (used for OOB elements)
  Value padVal = builder.create<arith::ConstantOp>(
      loc, i32Type, builder.getI32IntegerAttr(0));

  // Perform vector read
  Value SU_west = builder.create<vector::TransferReadOp>(
      loc, i32VecType, A_mat, ValueRange{R_SU_y, zeroIdx}, padVal);

  // SU_north = B_mat[:][C_SU_x]
  Value SU_north = builder.create<vector::TransferReadOp>(
      loc, i32VecType, B_mat, ValueRange{C_SU_x, zeroIdx}, padVal);

  SmallVector<Value> operands = {SU_west, SU_north, numColsConst};
  SmallVector<Type> resultTypes = {i32Type, i32VecType, i32VecType};

  OperationState state(loc, fpsaSuName);
  state.addOperands(operands);
  state.addTypes(resultTypes);

  Operation *fpsaOp = builder.create(std::move(state));
  Value result = fpsaOp->getResult(0);

  // Add handshake.name attribute to the systolic unit
  fpsaOp->setAttr(
      "handshake.name",
      StringAttr::get(ctx, "systolic_unit" + std::to_string(uniqueSUId)));
  uniqueSUId++;

  // OUT[R_SU_y][C_SU_x] = result
  builder.create<memref::StoreOp>(loc, result, OUT, ValueRange{R_SU_y, C_SU_x});
  return success();
}

LogicalResult
SystolicUnitGenerationPass::transposeMatrix(OpBuilder &builder, Location loc,
                                            func::FuncOp funcOp, Value matrix,
                                            Value &transposedMatrix) {

  auto original_matrix_type = matrix.getType().dyn_cast<MemRefType>();
  if (!original_matrix_type) {
    llvm::errs() << "[ERROR] Expected matrix to be a memref type\n";
    return failure();
  }

  // Build the transposed memref type
  auto transposed_matrix_type = MemRefType::get(
      {original_matrix_type.getDimSize(1), original_matrix_type.getDimSize(0)},
      original_matrix_type.getElementType());

  // Find the corresponding argument in the function
  int argIndex = -1;
  for (auto &arg : funcOp.getArguments()) {
    if (arg == matrix) {
      argIndex = arg.getArgNumber();
      break;
    }
  }
  if (argIndex == -1) {
    llvm::errs() << "[ERROR] Could not find argument corresponding to matrix\n";
    return failure();
  }

  // Update the function signature
  auto oldFuncType = funcOp.getFunctionType();
  auto inputsFunc = llvm::to_vector<4>(oldFuncType.getInputs());
  inputsFunc[argIndex] = transposed_matrix_type;
  auto newFuncType = FunctionType::get(funcOp.getContext(), inputsFunc,
                                       oldFuncType.getResults());

  // Update the entry block argument type to match the new signature
  Block &entryBlock = funcOp.front();
  entryBlock.getArgument(argIndex).setType(transposed_matrix_type);

  // Set the new function type
  funcOp.setFunctionType(newFuncType);
  return success();
}

LogicalResult SystolicUnitGenerationPass::extractMatMulParams(
    func::FuncOp func, func::CallOp fpsaCallOp, Value &A_mat, Value &B_mat,
    Value &OUT_mat, int &SA_ROWS, int &SA_COLS, int &A_rows, int &A_cols,
    int &B_cols) {

  // Assert that the call has exactly 5 arguments
  if (fpsaCallOp.getNumOperands() != 5) {
    return fpsaCallOp.emitError()
           << "Expected fpsa_mat_mul to have exactly 5 arguments";
  }

  // Extract the arguments of the call A_row, B_col, OUT, SA_ROWS, SA_COLS
  A_mat = fpsaCallOp.getOperand(0);
  B_mat = fpsaCallOp.getOperand(1);
  OUT_mat = fpsaCallOp.getOperand(2);
  Value arg_SA_ROWS = fpsaCallOp.getOperand(3);
  Value arg_SA_COLS = fpsaCallOp.getOperand(4);
  // Check that A_mat, B_mat, and OUT_mat are memref types and 2D
  auto A_type = A_mat.getType().dyn_cast<MemRefType>();
  auto B_type = B_mat.getType().dyn_cast<MemRefType>();
  auto OUT_type = OUT_mat.getType().dyn_cast<MemRefType>();
  if (!A_type || !B_type || !OUT_type || A_type.getRank() != 2 ||
      B_type.getRank() != 2 || OUT_type.getRank() != 2) {
    llvm::errs() << "[ERROR] Expected A, B, and OUT to be 2D memref types\n";
    return failure();
  }
  // Extract row and column sizes
  A_rows = A_type.getDimSize(0);
  A_cols = A_type.getDimSize(1);
  int B_rows = B_type.getDimSize(0);
  B_cols = B_type.getDimSize(1);
  int OUT_rows = OUT_type.getDimSize(0);
  int OUT_cols = OUT_type.getDimSize(1);

  if (A_cols != B_rows || A_rows != OUT_rows || B_cols != OUT_cols) {
    llvm::errs()
        << "[ERROR] Matrix dimensions do not match for multiplication\n";
    return failure();
  }
  // Check that arg_SA_ROWS and arg_SA_COLS are constant integers
  auto constSA_ROWS = arg_SA_ROWS.getDefiningOp<arith::ConstantOp>();
  auto constSA_COLS = arg_SA_COLS.getDefiningOp<arith::ConstantOp>();
  if (!constSA_ROWS || !constSA_COLS) {
    llvm::errs()
        << "[ERROR] Expected SA_ROWS and SA_COLS to be constant integers\n";
    return failure();
  }
  SA_ROWS = constSA_ROWS.getValue().cast<IntegerAttr>().getInt();
  SA_COLS = constSA_COLS.getValue().cast<IntegerAttr>().getInt();
  // Remove the original call operation
  fpsaCallOp.erase();
  return success();
}

namespace dynamatic {
std::unique_ptr<dynamatic::DynamaticPass> createSystolicUnitGeneration() {
  return std::make_unique<SystolicUnitGenerationPass>();
}
} // namespace dynamatic
