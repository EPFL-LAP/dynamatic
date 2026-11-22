#ifndef DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBAOPS_H
#define DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBAOPS_H

#include "BanbaDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace banba {

/// The name under which the config attribute is stored on the module
inline constexpr StringLiteral configAttrName = "banba.config";

/// Builds the types of the operations' inputs and outputs
class Types {
public:
  // `explicit` stops an MLIRContext *
  // from being converted into a Types
  // without the code asking for it
  explicit Types(MLIRContext *context) : context(context) {}

  /// CIRCT's clock type
  Type Clock() const { return ClockType::get(context); }

  /// An integer of the given width
  Type Int(unsigned width) const { return IntegerType::get(context, width); }

  /// A CIRCT hardware array holding `length` integers of the given width
  Type Array(unsigned length, unsigned width) const {
    return circt::hw::ArrayType::get(IntegerType::get(context, width), length);
  }

private:
  MLIRContext *context;
};

} // namespace banba

// The declarations of the dialect's interfaces,
// generated from BanbaInterfaces.td
#include "BanbaInterfaces.h.inc"

namespace banba {
namespace OpTrait {

/// The check that MLIR's verifier runs on every operation
/// that has VerifyTypesByConfig in its declaration
template <typename ConcreteType>
class VerifyTypesByConfig
    : public TraitBase<ConcreteType, VerifyTypesByConfig> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) {
    // Find the config, stored on the module that contains the operation
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op->emitOpError() << "must be inside a module";
    auto config = module->getAttrOfType<ConfigAttr>(configAttrName);
    if (!config)
      return op->emitOpError()
             << "module is missing the " << configAttrName << " attribute";

    // Ask the operation for the types it expects.
    // cast asserts that the operation implements ExpectedTypes
    auto typed = cast<ExpectedTypes>(op);
    SmallVector<Type> expectedOperandTypes =
        typed.getExpectedOperandTypes(config);

    // Check the number of inputs, then the type of each
    TypeRange operandTypes = op->getOperandTypes();
    if (operandTypes.size() != expectedOperandTypes.size())
      return op->emitOpError()
             << "has " << operandTypes.size() << " operands, expected "
             << expectedOperandTypes.size();
    for (unsigned index = 0; index < expectedOperandTypes.size(); ++index)
      if (operandTypes[index] != expectedOperandTypes[index])
        return op->emitOpError()
               << "operand " << index << " has type " << operandTypes[index]
               << ", expected " << expectedOperandTypes[index];

    // The same for the outputs
    SmallVector<Type> expectedResultTypes =
        typed.getExpectedResultTypes(config);
    TypeRange resultTypes = op->getResultTypes();
    if (resultTypes.size() != expectedResultTypes.size())
      return op->emitOpError()
             << "has " << resultTypes.size() << " results, expected "
             << expectedResultTypes.size();
    for (unsigned index = 0; index < expectedResultTypes.size(); ++index)
      if (resultTypes[index] != expectedResultTypes[index])
        return op->emitOpError()
               << "result " << index << " has type " << resultTypes[index]
               << ", expected " << expectedResultTypes[index];

    // The inside of the operation hands back the values of its outputs
    // through its last operation. Check that they match the outputs
    if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
      Operation *terminator = op->getRegion(0).front().getTerminator();
      TypeRange outputTypes = terminator->getOperandTypes();
      if (outputTypes.size() != resultTypes.size())
        return op->emitOpError() << "region returns " << outputTypes.size()
                                 << " values, expected " << resultTypes.size();
      for (unsigned index = 0; index < resultTypes.size(); ++index)
        if (outputTypes[index] != resultTypes[index])
          return op->emitOpError()
                 << "region output " << index << " has type "
                 << outputTypes[index] << ", expected " << resultTypes[index];
    }

    return success();
  }
};

} // namespace OpTrait
} // namespace banba

// The declarations of the dialect's operations,
// generated from the Ops.td file of each folder
#define GET_OP_CLASSES
#include "Banba.h.inc"

#endif // DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBAOPS_H
