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
/// It gives access to the config of the LSQ being generated,
/// which it gets from the module that contains the operation.
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

    // Get the config, stored on the module that contains the operation
    ModuleOp module = op->getParentOfType<ModuleOp>();
    assert(module && "the operation is not inside a module");
    config = module->getAttrOfType<ConfigAttr>(configAttrName);
    assert(config && "the module has no config");

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

  /// Static config getter,
  /// finds the current scope through the static `active` pointer
  static ConfigAttr getConfig() {
    assert(active && "no BanbaSugarScope has been declared");
    return active->config;
  }

private:
  /// Static pointer to the current scope
  inline static BanbaSugarScope *active = nullptr;

  /// Builder for the region of this scope's operation
  OpBuilder builder;

  /// Location of this scope's operation,
  /// given to every operation created inside its region
  Location loc;

  /// Config of the LSQ being generated
  ConfigAttr config;

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

  // Index with a constant:
  // on an integer, extract a single bit,
  // and on a hw.array, read an element
  BanbaValue operator[](unsigned index) {
    OpBuilder &builder = BanbaSugarScope::getBuilder();
    Location loc = BanbaSugarScope::getLoc();

    // On an integer, index out a single bit
    if (isa<IntegerType>(value.getType()))
      return circt::comb::ExtractOp::create(builder, loc, value, index, 1)
          .getResult();

    // Otherwise the value must be a hw.array, which cast asserts
    auto arrayType = cast<circt::hw::ArrayType>(value.getType());
    unsigned numElements = arrayType.getNumElements();

    // We use a bitcast on a one-entry array
    // to avoid dealing with indexing headaches
    if (numElements == 1) {
      // Only entry 0 exists, so the index must be 0
      assert(index == 0 && "non-zero constant index into a one-entry array");

      // And then just bitcast it to be non-array
      return circt::hw::BitcastOp::create(builder, loc,
                                          arrayType.getElementType(), value)
          .getResult();
    }

    // Otherwise convert the index to a constant in the IR
    BanbaValue indexConstant(index, Log2_32_Ceil(numElements));

    // And then return the indexing operation
    return circt::hw::ArrayGetOp::create(builder, loc, value, indexConstant)
        .getResult();
  }

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

/// `a - b` creates the subtraction of two values,
/// as a CIRCT comb.sub operation
inline BanbaValue operator-(BanbaValue lhs, BanbaValue rhs) {
  return circt::comb::SubOp::create(BanbaSugarScope::getBuilder(),
                                    BanbaSugarScope::getLoc(), lhs, rhs)
      .getResult();
}

/// `a + b` creates the addition of two values,
/// as a CIRCT comb.add operation
inline BanbaValue operator+(BanbaValue lhs, BanbaValue rhs) {
  return circt::comb::AddOp::create(BanbaSugarScope::getBuilder(),
                                    BanbaSugarScope::getLoc(), lhs, rhs)
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

/// Select between two values on a single bit,
/// as a CIRCT comb.mux operation
inline BanbaValue select(BanbaValue sel, BanbaValue trueValue,
                         BanbaValue falseValue) {
  return circt::comb::MuxOp::create(BanbaSugarScope::getBuilder(),
                                    BanbaSugarScope::getLoc(), sel, trueValue,
                                    falseValue)
      .getResult();
}

/// Build a hw.array constant from a list of values,
/// bitwidth bits per element.
/// numElements pads the array with zeros beyond the list
/// (0 means exactly the list length)
inline BanbaValue arrayConstant(ArrayRef<int32_t> values, unsigned bitwidth,
                                unsigned numElements = 0) {
  OpBuilder &builder = BanbaSugarScope::getBuilder();

  // If no number of elements is given, use the length of the list
  if (numElements == 0)
    numElements = values.size();

  // Check that the list fits in the array
  assert(values.size() <= (size_t)numElements &&
         "more constant values than array elements");

  // The type of an element of the array
  Type elementType = builder.getIntegerType(bitwidth);

  // For each element of the array,
  // make an attribute holding its constant value
  SmallVector<Attribute> elements;
  for (unsigned i = 0; i < numElements; ++i) {
    // Take the value from the list, or zero beyond the end of the list
    int32_t value = i < values.size() ? values[i] : 0;

    // Make the attribute, and add it to the elements
    elements.push_back(builder.getIntegerAttr(elementType, value));
  }

  // Callers build their lists element 0 first,
  // but hw.aggregate_constant expects the highest-indexed element first,
  // so first reverse
  std::reverse(elements.begin(), elements.end());

  // And then make the constant array, as a single operation
  return circt::hw::AggregateConstantOp::create(
             builder, BanbaSugarScope::getLoc(),
             circt::hw::ArrayType::get(elementType, numElements),
             builder.getArrayAttr(elements))
      .getResult();
}

/// Get a zero-valued value of the given type
inline BanbaValue zeroValue(Type type) {
  OpBuilder &builder = BanbaSugarScope::getBuilder();
  Location loc = BanbaSugarScope::getLoc();

  // Integer
  if (isa<IntegerType>(type))
    return circt::hw::ConstantOp::create(builder, loc, type, 0).getResult();

  // Else assert it is an array type
  auto arrayType = cast<circt::hw::ArrayType>(type);

  // Just make a super-wide zero integer,
  // so first we get the width
  unsigned width = arrayType.getNumElements() *
                   arrayType.getElementType().getIntOrFloatBitWidth();
  BanbaValue packed(0, width);

  // And then cast it to an array type
  return circt::hw::BitcastOp::create(builder, loc, type, packed).getResult();
}

/// Zero-extend an integer to the given width,
/// by concatenating zeros above it
inline BanbaValue zext(BanbaValue value, unsigned bitwidth) {
  Value inValue = value;
  unsigned inBitwidth = inValue.getType().getIntOrFloatBitWidth();
  assert(bitwidth >= inBitwidth && "zext target width smaller than input");

  // Nothing to extend if the value already has the target width
  if (bitwidth == inBitwidth)
    return value;

  BanbaValue zeros(0, bitwidth - inBitwidth);
  return circt::comb::ConcatOp::create(BanbaSugarScope::getBuilder(),
                                       BanbaSugarScope::getLoc(), zeros,
                                       inValue)
      .getResult();
}

/// Create an operation, and construct its region.
/// The OpBuilder and the location come from the current BanbaSugarScope,
/// and the arguments are passed on to the operation's own `create`
template <typename OpT, typename... Args>
OpT createOp(Args &&...args) {
  OpT op = OpT::create(BanbaSugarScope::getBuilder(), BanbaSugarScope::getLoc(),
                       std::forward<Args>(args)...);
  op.constructRegion();
  return op;
}

} // namespace BanbaSugar

} // namespace banba

#endif // DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBASUGAR_H
