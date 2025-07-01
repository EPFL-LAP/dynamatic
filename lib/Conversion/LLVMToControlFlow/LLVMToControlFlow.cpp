#include "dynamatic/Conversion/LLVMToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

std::string getCfgNodeLabel(Block &block) {
  std::string blockLabel;
  llvm::raw_string_ostream bl(blockLabel);
  bl << "\"";
  block.printAsOperand(bl);
  bl << "\"";
  bl.flush();
  return bl.str();
}

struct LLVMAddToArithAddPattern : public OpRewritePattern<LLVM::AddOp> {
  using OpRewritePattern<LLVM::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AddOp llvmAdd,
                                PatternRewriter &rewriter) const override {
    Location loc = llvmAdd.getLoc();
    Value lhs = llvmAdd.getLhs();
    Value rhs = llvmAdd.getRhs();

    // Check operands are integer type (LLVM dialect integer)
    auto lhsType = lhs.getType().dyn_cast<IntegerType>();
    auto rhsType = rhs.getType().dyn_cast<IntegerType>();
    if (!lhsType || !rhsType)
      return failure();

    // Convert LLVM integer operands to IntegerType (arith)
    // We need to convert LLVM::LLVMType to IntegerType for arith ops
    // Typically, LLVM integer types correspond to arith IntegerType with same
    // bitwidth
    unsigned bitWidth = lhsType.getWidth();
    auto intType = rewriter.getIntegerType(bitWidth);

    // Cast lhs and rhs from LLVM integer to arith integer type using
    // llvm.inttoptr or bitcast-like? Usually LLVM dialect values aren't
    // directly compatible with arith ops. So insert llvm-llvm -> arith cast or
    // do a Value conversion? Since direct cast is complex, here we assume
    // values are from LLVM dialect but convertible. You may need to add
    // conversion ops in a full pipeline.

    // For simplicity, just create arith.addi with the same operands (assuming
    // compatible types) Otherwise, you must handle type conversion explicitly
    // (usually in multi-stage conversion).

    // Create arith.addi op
    Value lhsCast = lhs;
    Value rhsCast = rhs;

    // If types don't match, you may want to insert conversions here (omitted
    // for brevity)

    Value addi = rewriter.create<arith::AddIOp>(loc, lhsCast, rhsCast);
    rewriter.replaceOp(llvmAdd, addi);
    return success();
  }
};

namespace {
struct LLVMToControlFlowPass
    : public dynamatic::impl::LLVMToControlFlowBase<LLVMToControlFlowPass> {

  void runDynamaticPass() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    // Setup conversion target
    ConversionTarget target(*context);

    // context->allowUnregisteredDialects();

    // target.addIllegalDialect<LLVM::LLVMDialect>();
    // target.addLegalDialect<arith::ArithDialect>();

    // Add arith-to-LLVM patterns
    RewritePatternSet patterns(context);

    // Iterate over all top-level operations in the module.
    for (Operation &op : module.getOps()) {
      // Check if the operation is an LLVM function.
      if (auto func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        llvm::outs() << "Function: " << func.getName() << "\n";
        int blockIndex = 0;
        for (Block &block : func.getBody()) {
          llvm::outs() << "  BasicBlock #" << blockIndex++;
          std::string label = getCfgNodeLabel(block);
          llvm::outs() << " (name: " << label << ")";
          llvm::outs() << "\n";
        }
      }
    }

    patterns.add<LLVMAddToArithAddPattern>(context);
  }
};
} // namespace

namespace dynamatic {
std::unique_ptr<DynamaticPass> createLLVMToControlFlowPass() {
  return std::make_unique<LLVMToControlFlowPass>();
}
} // namespace dynamatic
