#ifndef DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBASUGAR_H
#define DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBASUGAR_H

#include "BanbaOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"

namespace banba {

/// A definition of constructRegion creates operations
/// inside the region of its operation, through an MLIR OpBuilder.
/// Declaring a BanbaSugarScope
/// allows the use of syntactic sugar: operations can be created
/// without explicitly passing an OpBuilder for every single one.
/// It also adds the block that holds those operations
/// to the operation's region,
/// containing only the terminator of that block
/// and the placeholders passed to it as operands.
/// These placeholders are later removed,
/// when the HasNamedResultDrivers trait is used to specify
/// which MLIR value drives which result
class BanbaSugarScope {
public:
  // `explicit` stops an Operation *
  // from being converted into a BanbaSugarScope
  // without the code asking for it
  explicit BanbaSugarScope(Operation *op)
      // Make a builder, store the operation's location,
      // and store the scope that was active until now
      : builder(op->getContext()), loc(op->getLoc()), previous(active) {
    // Create the block of the operation's region,
    // and point the builder at it
    Region &body = op->getRegion(0);
    assert(body.empty() && "region already constructed");
    builder.setInsertionPointToStart(&body.emplaceBlock());

    // Create a placeholder for each of the operation's results
    SmallVector<Value> placeholders;
    for (Type type : op->getResultTypes())
      placeholders.push_back(
          PlaceholderOp::create(builder, op->getLoc(), type));

    // Create the terminator of the region,
    // passing the placeholders as operands
    Operation *terminator =
        OutputOp::create(builder, op->getLoc(), placeholders);

    // Create all later operations above the terminator
    builder.setInsertionPoint(terminator);

    // Allow later syntactic sugar to find this scope
    // through the single static `active` pointer
    active = this;
  }

  // When this scope ends, point `active` back at the scope before it
  ~BanbaSugarScope() { active = previous; }

  /// Static builder getter,
  /// finds the current scope through the static `active` pointer
  static OpBuilder &getBuilder() {
    assert(active && "no BanbaSugarScope has been declared");
    return active->builder;
  }

  /// Static location getter,
  /// finds the current scope through the static `active` pointer
  static Location getLoc() {
    assert(active && "no BanbaSugarScope has been declared");
    return active->loc;
  }

private:
  /// Static pointer to the current scope
  inline static BanbaSugarScope *active = nullptr;

  /// Builder for the region of this scope's operation
  OpBuilder builder;

  /// Location of this scope's operation,
  /// given to every operation created inside its region
  Location loc;

  /// Pointer to the scope to restore when this scope ends
  BanbaSugarScope *previous;
};

/// Wraps an MLIR Value, to allow the use of syntactic sugar on it:
/// for example, `a & b` to create the AND of two values
class BanbaValue {
public:
  // Not `explicit`, so that an MLIR Value can be converted into a BanbaValue
  // wherever a BanbaValue is expected
  BanbaValue(Value value) : value(value) {}

  // Create a constant of the given width, as a CIRCT hw.constant operation
  BanbaValue(uint64_t constant, unsigned width)
      : value(circt::hw::ConstantOp::create(
            BanbaSugarScope::getBuilder(), BanbaSugarScope::getLoc(),
            BanbaSugarScope::getBuilder().getIntegerType(width), constant)) {}

  // Convert a BanbaValue back into an MLIR Value
  // wherever an MLIR Value is expected
  operator Value() const { return value; }

private:
  /// The wrapped MLIR Value
  Value value;
};

/// `a & b` creates the AND of two values, as a CIRCT comb.and operation
inline BanbaValue operator&(BanbaValue lhs, BanbaValue rhs) {
  return circt::comb::AndOp::create(BanbaSugarScope::getBuilder(),
                                    BanbaSugarScope::getLoc(), lhs, rhs)
      .getResult();
}

/// `a >= b` creates an unsigned greater-than-or-equal comparison
/// of two values, as a CIRCT comb.icmp operation
inline BanbaValue operator>=(BanbaValue lhs, BanbaValue rhs) {
  return circt::comb::ICmpOp::create(BanbaSugarScope::getBuilder(),
                                     BanbaSugarScope::getLoc(),
                                     circt::comb::ICmpPredicate::uge, lhs, rhs)
      .getResult();
}

// Syntactic sugar that is written as a function call.
// The namespace makes each call show that it is sugar,
// which creates its operations through the current BanbaSugarScope:
// `BanbaSugar::concatToArray(elements)`
namespace BanbaSugar {

/// Build a hw.array from a list of values
inline BanbaValue concatToArray(ArrayRef<Value> elements) {
  // Callers build their lists element 0 first,
  // but hw.array_create expects the highest-indexed element first,
  // so first reverse
  SmallVector<Value> operands = llvm::to_vector(llvm::reverse(elements));

  // And then make the array
  return circt::hw::ArrayCreateOp::create(BanbaSugarScope::getBuilder(),
                                          BanbaSugarScope::getLoc(), operands)
      .getResult();
}

} // namespace BanbaSugar

} // namespace banba

#endif // DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBASUGAR_H
