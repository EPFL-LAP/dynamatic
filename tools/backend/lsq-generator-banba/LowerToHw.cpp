#include "LowerToHw.h"
#include "BanbaOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace banba;

/// Create the inputs and outputs of the hw.module,
/// based on the operation
static circt::hw::ModulePortInfo makePortInfo(Operation *op) {
  MLIRContext *context = op->getContext();

  // Get the names of the operation's operands and results
  auto portNames = cast<PortNames>(op);
  SmallVector<std::string> operandNames = portNames.getOperandNames();
  SmallVector<std::string> resultNames = portNames.getResultNames();

  // Vector of port infos
  SmallVector<circt::hw::PortInfo> ports;

  // Add the operands
  // (port names carry an _i suffix marking them as inputs,
  // except clk and rst, whose direction is unambiguous)
  for (auto [index, type] : llvm::enumerate(op->getOperandTypes())) {
    circt::hw::PortInfo port;
    bool isClkOrRst =
        operandNames[index] == "clk" || operandNames[index] == "rst";
    port.name = mlir::StringAttr::get(
        context, isClkOrRst ? operandNames[index] : operandNames[index] + "_i");
    port.type = type;
    port.dir = circt::hw::ModulePort::Direction::Input;
    port.argNum = index;
    ports.push_back(port);
  }

  // Add the results
  // (port names carry an _o suffix marking them as outputs)
  for (auto [index, type] : llvm::enumerate(op->getResultTypes())) {
    circt::hw::PortInfo port;
    port.name = mlir::StringAttr::get(context, resultNames[index] + "_o");
    port.type = type;
    port.dir = circt::hw::ModulePort::Direction::Output;
    port.argNum = index;
    ports.push_back(port);
  }
  return circt::hw::ModulePortInfo(ports);
}

/// The uniqued module name for an op: the base name is the op's mnemonic,
/// with its dots as underscores; the first of a kind keeps the base name,
/// repeats get a counter suffix.
///
/// Every name is prefixed with the LSQ's name, so several LSQs compiled into
/// one design do not collide on shared module names.
static std::string nameHWModule(Operation *op, StringRef lsqName,
                                llvm::StringMap<unsigned> &counts) {
  std::string baseName = op->getName().stripDialect().str();
  std::replace(baseName.begin(), baseName.end(), '.', '_');
  baseName = (lsqName + "_" + baseName).str();

  unsigned index = counts[baseName]++;
  if (index == 0) {
    return baseName;
  } else {
    return (baseName + "_" + llvm::Twine(index)).str();
  }
}

/// Create the hw.module for an op
static circt::hw::HWModuleOp createHWModule(OpBuilder &builder, ModuleOp modOp,
                                            ConstructsRegion op,
                                            StringRef moduleName) {
  // MLIR stuff
  Block *banbaBody = op.getBodyBlock();
  Location loc = op->getLoc();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(modOp.getBody());

  // Make the mod op
  circt::hw::HWModuleOp hwMod = circt::hw::HWModuleOp::create(
      builder, loc, builder.getStringAttr(moduleName), makePortInfo(op));
  // Get its body
  Block *hwModuleBody = hwMod.getBodyBlock();

  // Get both output operations
  Operation *banbaOutput = banbaBody->getTerminator();
  Operation *hwOutput = hwModuleBody->getTerminator();

  // Build the final hw.output from the banba.output operands,
  // then drop both old terminators
  builder.setInsertionPoint(hwOutput);
  circt::hw::OutputOp newOutput =
      circt::hw::OutputOp::create(builder, loc, banbaOutput->getOperands());
  hwOutput->erase();
  banbaOutput->erase();

  // Move the whole op body into the hw module,
  // inserting the moved operations above the output op
  hwModuleBody->getOperations().splice(newOutput->getIterator(),
                                       banbaBody->getOperations());

  // For each operand of the op
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    // Get the corresponding block arg
    mlir::BlockArgument moduleInput = hwModuleBody->getArgument(index);

    // For each use of that operand
    for (mlir::OpOperand &use : llvm::make_early_inc_range(operand.getUses())) {
      // If the use is inside the new hw module's body
      if (use.getOwner()->getBlock() == hwModuleBody) {
        // It should use the block arg instead
        use.set(moduleInput);
      }
    }
  }

  return hwMod;
}

/// Replace a banba op with a hw instance of its hw module,
/// erasing the banba op.
/// The instance name drops the module name's LSQ prefix: instance names
/// only need to be unique within the parent module, and the emitter
/// prefixes every instance-result wire with the instance name.
static void replaceWithInst(OpBuilder &builder, ConstructsRegion op,
                            circt::hw::HWModuleOp module, StringRef moduleName,
                            StringRef lsqName) {
  // Set up MLIR stuff
  Location loc = op->getLoc();
  builder.setInsertionPoint(op.getOperation());

  // Convert to vector
  SmallVector<Value> inputs(op->getOperands());

  // Strip the LSQ prefix off the module name
  StringRef instanceName = moduleName;
  instanceName.consume_front(lsqName);
  instanceName.consume_front("_");

  // Create the inst
  circt::hw::InstanceOp instance = circt::hw::InstanceOp::create(
      builder, loc, module, instanceName, inputs);

  // And have the inst replace the banba op
  op->replaceAllUsesWith(instance.getResults());
  op->erase();
}

/// Erase a banba op that sits directly in the builtin module,
/// together with the placeholders that are its operands.
/// No instance replaces it,
/// as there is no hw.module around it for an instance to be in
static void eraseTopLevelOp(ConstructsRegion op) {
  // Get the placeholders, before the op that uses them is erased
  SmallVector<Operation *> placeholders;
  for (Value operand : op->getOperands()) {
    Operation *placeholder = operand.getDefiningOp();
    assert(placeholder && isa<PlaceholderOp>(placeholder) &&
           "a top-level op's operands must be placeholders");
    placeholders.push_back(placeholder);
  }

  // Erase the op, and then the placeholders, which nothing uses any more
  op->erase();
  for (Operation *placeholder : placeholders)
    placeholder->erase();
}

/// Lower a banba op:
/// handle sub-ops first, then create its hw module,
/// then replace it with a hw instance
static void handleOp(OpBuilder &builder, ModuleOp modOp, ConstructsRegion op,
                     StringRef lsqName, llvm::StringMap<unsigned> &counts) {
  // Lower all sub-ops before lowering the parent
  for (Operation &subOp : llvm::make_early_inc_range(*op.getBodyBlock()))
    if (auto subOpToLower = llvm::dyn_cast<ConstructsRegion>(&subOp))
      handleOp(builder, modOp, subOpToLower, lsqName, counts);

  // Get the uniqued module name for this op
  std::string moduleName = nameHWModule(op.getOperation(), lsqName, counts);

  // Create the hw module
  circt::hw::HWModuleOp hwMod =
      createHWModule(builder, modOp, op, moduleName);

  if (op->getParentOp() == modOp.getOperation()) {
    // An op directly in the builtin module has no hw.module around it,
    // so it is erased
    eraseTopLevelOp(op);
  } else {
    // Otherwise replace the op with a hw inst
    replaceWithInst(builder, op, hwMod, moduleName, lsqName);
  }
}

void banba::lowerToHw(ModuleOp modOp) {
  OpBuilder builder(modOp.getContext());

  // Get the config
  auto config = modOp->getAttrOfType<ConfigAttr>(configAttrName);

  // Prefix modules with the lsq name,
  // after removing "handshake_lsq_" from it
  StringRef lsqName = config.getName();
  lsqName.consume_front("handshake_lsq_");

  // Call handleOp on the contents of the module
  llvm::StringMap<unsigned> counts;
  for (Operation &op : llvm::make_early_inc_range(*modOp.getBody()))
    if (auto banbaOp = llvm::dyn_cast<ConstructsRegion>(&op))
      handleOp(builder, modOp, banbaOp, lsqName, counts);
}
