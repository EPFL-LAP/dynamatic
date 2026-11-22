#ifndef DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBASUGAR_H
#define DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBASUGAR_H

#include "BanbaOps.h"
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
      // Make a builder,
      // and store the scope that was active until now
      : builder(op->getContext()), previous(active) {
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

private:
  /// Static pointer to the current scope
  inline static BanbaSugarScope *active = nullptr;

  /// Builder for the region of this scope's operation
  OpBuilder builder;

  /// Pointer to the scope to restore when this scope ends
  BanbaSugarScope *previous;
};

} // namespace banba

#endif // DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBASUGAR_H
