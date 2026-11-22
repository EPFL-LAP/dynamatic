#include "ReduceCastsToIntegers.h"
#include "BanbaOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace banba;

/// Whether a type is an array of 1-bit values
static bool isArrayOf1BitValues(Type type) {
  auto arrayType = llvm::dyn_cast<circt::hw::ArrayType>(type);
  return arrayType && arrayType.getElementType().isInteger(1);
}

/// The integer type with one bit per element of an array of 1-bit values
static Type integerFor(Type arrayOf1BitValues) {
  auto arrayType = cast<circt::hw::ArrayType>(arrayOf1BitValues);
  return IntegerType::get(arrayOf1BitValues.getContext(),
                          arrayType.getNumElements());
}

/// The array type of 1-bit values with one element per bit of an integer
static Type arrayFor(Type integer) {
  unsigned numBits = integer.getIntOrFloatBitWidth();
  Type bit = IntegerType::get(integer.getContext(), 1);
  return circt::hw::ArrayType::get(bit, numBits);
}

/// Create a cast of a value to the given type, at the builder's position
static Value castTo(OpBuilder &builder, Value value, Type type) {
  return circt::hw::BitcastOp::create(builder, value.getLoc(), type, value);
}

/// Whether a consumer is a cast to an integer
static bool isCastToInteger(OpOperand *consumer) {
  auto cast = llvm::dyn_cast<circt::hw::BitcastOp>(consumer->getOwner());
  return cast && isa<IntegerType>(cast.getType());
}

/// Whether a producer is a cast from an integer
static bool isCastFromInteger(Value producer) {
  auto cast =
      llvm::dyn_cast_or_null<circt::hw::BitcastOp>(producer.getDefiningOp());
  return cast && isa<IntegerType>(cast.getInput().getType());
}

/// How the modules are connected to each other
struct Hierarchy {
  /// All the instances of each module
  llvm::DenseMap<circt::hw::HWModuleOp, SmallVector<circt::hw::InstanceOp>>
      instancesOf;

  /// The module that each instance instantiates
  llvm::DenseMap<circt::hw::InstanceOp, circt::hw::HWModuleOp> moduleOf;
};

/// Make a map of module -> all its instances,
/// and another map of instance -> its module
static Hierarchy findHierarchy(ModuleOp modOp) {
  Hierarchy hierarchy;

  // An instance refers to its module by name,
  // so first note every module under its name
  llvm::StringMap<circt::hw::HWModuleOp> moduleNamed;
  for (auto hwMod : modOp.getOps<circt::hw::HWModuleOp>())
    moduleNamed[hwMod.getName()] = hwMod;

  // For each instance in the design,
  // which means each instance inside each module
  for (auto parent : modOp.getOps<circt::hw::HWModuleOp>()) {
    for (auto instance :
         parent.getBodyBlock()->getOps<circt::hw::InstanceOp>()) {
      // Look up the module it instantiates
      circt::hw::HWModuleOp hwMod =
          moduleNamed.lookup(instance.getReferencedModuleName());

      // Add the pair to both maps
      hierarchy.moduleOf[instance] = hwMod;
      hierarchy.instancesOf[hwMod].push_back(instance);
    }
  }
  return hierarchy;
}

/// Find the values that produce a value, ignoring module boundaries.
/// If the value is a module's input,
/// follow it out to what each instance of the module connects to that input.
/// If the value is an instance's output,
/// follow it in to what the instantiated module sends to that output.
/// An input of the top-level module cannot be followed further,
/// so it is the producer
static void findRealProducers(Value value, const Hierarchy &hierarchy,
                              llvm::SetVector<Value> &producers) {
  if (auto input = llvm::dyn_cast<BlockArgument>(value)) {
    // The value is a module's input:
    // find the instances of the module that the input belongs to
    auto hwMod = cast<circt::hw::HWModuleOp>(input.getOwner()->getParentOp());
    auto instances = hierarchy.instancesOf.find(hwMod);

    if (instances == hierarchy.instancesOf.end()) {
      // The top-level module has no instances,
      // so its input is the producer
      producers.insert(input);
    } else {
      // Follow the input out to what each instance connects to it
      for (circt::hw::InstanceOp instance : instances->second) {
        Value connected = instance.getOperand(input.getArgNumber());
        findRealProducers(connected, hierarchy, producers);
      }
    }
  } else if (auto instance =
                 llvm::dyn_cast<circt::hw::InstanceOp>(value.getDefiningOp())) {
    // The value is an instance's output:
    // find the module that the instance instantiates
    circt::hw::HWModuleOp hwMod = hierarchy.moduleOf.lookup(instance);

    // Follow the output in to what that module sends to it.
    // hw.output, the last operation of the module, has one operand per output
    unsigned outputNumber = cast<OpResult>(value).getResultNumber();
    Operation *output = hwMod.getBodyBlock()->getTerminator();
    findRealProducers(output->getOperand(outputNumber), hierarchy, producers);
  } else {
    // The value is produced by an ordinary operation,
    // so it is the producer
    producers.insert(value);
  }
}

/// Find the places that consume a value, ignoring module boundaries.
/// If the value is consumed by an instance,
/// follow it in to what consumes that input inside the instantiated module.
/// If the value is consumed by hw.output,
/// follow it out to what consumes that output at each instance of the module.
/// An output of the top-level module cannot be followed further,
/// so its hw.output is the consumer
static void findRealConsumers(Value value, const Hierarchy &hierarchy,
                              SmallVector<OpOperand *> &consumers) {
  // For each place the value is used
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();

    if (auto instance = llvm::dyn_cast<circt::hw::InstanceOp>(user)) {
      // The value is consumed by an instance:
      // find the module that the instance instantiates
      circt::hw::HWModuleOp hwMod = hierarchy.moduleOf.lookup(instance);

      // Follow it in to what consumes that input inside the module
      Value input = hwMod.getBodyBlock()->getArgument(use.getOperandNumber());
      findRealConsumers(input, hierarchy, consumers);
    } else if (auto output = llvm::dyn_cast<circt::hw::OutputOp>(user)) {
      // The value is consumed by hw.output:
      // find the instances of the module that the hw.output belongs to
      auto hwMod = cast<circt::hw::HWModuleOp>(output->getParentOp());
      auto instances = hierarchy.instancesOf.find(hwMod);

      if (instances == hierarchy.instancesOf.end()) {
        // The top-level module has no instances,
        // so its hw.output is the consumer
        consumers.push_back(&use);
      } else {
        // Follow the output out to what consumes it at each instance
        for (circt::hw::InstanceOp instance : instances->second) {
          Value instanceOutput = instance.getResult(use.getOperandNumber());
          findRealConsumers(instanceOutput, hierarchy, consumers);
        }
      }
    } else {
      // The value is consumed by an ordinary operation,
      // so it is the consumer
      consumers.push_back(&use);
    }
  }
}

/// Whether the array that a cast reads is only ever consumed as an integer:
/// every consumer of every producer of the array is a cast to an integer.
/// Also gives the producers and the consumers that were found
static bool isOnlyConsumedAsInteger(circt::hw::BitcastOp cast,
                                    const Hierarchy &hierarchy,
                                    llvm::SetVector<Value> &producers,
                                    SmallVector<OpOperand *> &consumers) {
  // Find the producers of the array that the cast reads
  findRealProducers(cast.getInput(), hierarchy, producers);

  // Find the consumers of each of those producers
  for (Value producer : producers)
    findRealConsumers(producer, hierarchy, consumers);

  // Every one of them must be a cast to an integer
  return llvm::all_of(consumers, isCastToInteger);
}

/// Whether the array that a cast produces is only ever produced from an
/// integer: every producer of what every consumer of the array reads
/// is a cast from an integer.
/// Also gives the consumers and the producers that were found
static bool isOnlyProducedFromInteger(circt::hw::BitcastOp cast,
                                      const Hierarchy &hierarchy,
                                      SmallVector<OpOperand *> &consumers,
                                      llvm::SetVector<Value> &producers) {
  // Find the consumers of the array that the cast produces
  findRealConsumers(cast, hierarchy, consumers);

  // The cast itself produces the array.
  // It would not be found below if the array has no consumers,
  // as when a module never reads one of its input ports
  producers.insert(cast);

  // Find the producers of what each of those consumers reads
  for (OpOperand *consumer : consumers)
    findRealProducers(consumer->get(), hierarchy, producers);

  // Every one of them must be a cast from an integer
  return llvm::all_of(producers, isCastFromInteger);
}

/// The casts that are still to be examined.
/// A cast that is deleted must also be removed from here
using CastPool = llvm::SetVector<Operation *>;

/// Make a producer of an array of 1-bit values produce an integer.
/// The operation is changed into its form on integers if it has one,
/// and otherwise one cast to the integer is put right after it.
/// Returns the integer
static Value produceAsInteger(Value producer, CastPool &pool) {
  Type integerType = integerFor(producer.getType());

  // The operation that produces it.
  // It is null if the producer is an input of the top-level module
  Operation *op = producer.getDefiningOp();

  if (auto bitcast = llvm::dyn_cast_or_null<circt::hw::BitcastOp>(op);
      bitcast && bitcast.getInput().getType() == integerType) {
    // A cast from the integer: what it reads is already the integer
    return bitcast.getInput();
  } else if (auto create =
                 llvm::dyn_cast_or_null<circt::hw::ArrayCreateOp>(op)) {
    // hw.array_create of bits: comb.concat of the same bits in the same order,
    // as both take the highest index first
    OpBuilder builder(op);
    return circt::comb::ConcatOp::create(builder, op->getLoc(),
                                         create.getInputs());
  } else if (auto constant =
                 llvm::dyn_cast_or_null<circt::hw::AggregateConstantOp>(op)) {
    // hw.aggregate_constant of bits: hw.constant with the same bits

    // Get the attribute which contains the value of the aggregate array
    ArrayAttr elements = constant.getFields();
    unsigned numBits = elements.size();

    // If an AggregateConstantOp was used as an integer,
    // it must have N 1-bit values: replace it with an N-bit integer
    APInt bits(numBits, 0);

    // For each element of the array
    for (unsigned i = 0; i < numBits; ++i) {
      // The constant lists its elements with the highest index first
      auto element = cast<IntegerAttr>(elements[numBits - 1 - i]);
      assert(element.getValue().getBitWidth() == 1 &&
             "expected an array of 1-bit values");

      // Set bit i of the integer to element i of the array
      bits.setBitVal(i, element.getValue().isOne());
    }

    // Create the hw.constant with those bits, where the aggregate constant is
    OpBuilder builder(op);
    return circt::hw::ConstantOp::create(builder, op->getLoc(), bits);
  } else if (auto mux = llvm::dyn_cast_or_null<circt::comb::MuxOp>(op)) {
    // comb.mux between two arrays of 1-bit values: comb.mux between two
    // integers

    // Put a cast to the integer on each of the two arrays
    OpBuilder builder(op);
    Value trueValue = castTo(builder, mux.getTrueValue(), integerType);
    Value falseValue = castTo(builder, mux.getFalseValue(), integerType);

    // Add the two casts to the pool, to be examined in turn
    pool.insert(trueValue.getDefiningOp());
    pool.insert(falseValue.getDefiningOp());

    // Create the mux between the two integers, with the same select
    return circt::comb::MuxOp::create(builder, op->getLoc(), mux.getCond(),
                                      trueValue, falseValue);
  } else if (auto wire = llvm::dyn_cast_or_null<circt::hw::WireOp>(op)) {
    // hw.wire of an array of 1-bit values: hw.wire of an integer

    // Put a cast to the integer on the array, and add it to the pool
    OpBuilder builder(op);
    Value input = castTo(builder, wire.getInput(), integerType);
    pool.insert(input.getDefiningOp());

    // Create the wire of the integer, with the same name
    return circt::hw::WireOp::create(builder, op->getLoc(), input,
                                     wire.getNameAttr());
  } else if (auto concat =
                 llvm::dyn_cast_or_null<circt::hw::ArrayConcatOp>(op)) {
    // hw.array_concat of arrays of 1-bit values: comb.concat of integers,
    // in the same order, as both take the highest index first
    OpBuilder builder(op);
    SmallVector<Value> inputs;

    // For each array that is concatenated
    for (Value array : concat.getInputs()) {
      // Put a cast to its integer on it, and add the cast to the pool
      Value input = castTo(builder, array, integerFor(array.getType()));
      pool.insert(input.getDefiningOp());
      inputs.push_back(input);
    }

    // Create the comb.concat of the integers
    return circt::comb::ConcatOp::create(builder, op->getLoc(), inputs);
  } else if (auto slice = llvm::dyn_cast_or_null<circt::hw::ArraySliceOp>(op);
             slice &&
             slice.getLowIndex().getDefiningOp<circt::hw::ConstantOp>()) {
    // hw.array_slice of an array of 1-bit values, starting at a constant index:
    // comb.extract of the same range of bits
    auto lowIndex = slice.getLowIndex().getDefiningOp<circt::hw::ConstantOp>();
    unsigned lowBit = lowIndex.getValue().getZExtValue();

    // Put a cast to the integer on the array, and add it to the pool
    OpBuilder builder(op);
    Value array = slice.getInput();
    Value input = castTo(builder, array, integerFor(array.getType()));
    pool.insert(input.getDefiningOp());

    // Create the comb.extract, starting at the same bit,
    // of as many bits as the slice has elements
    unsigned numBits = integerType.getIntOrFloatBitWidth();
    return circt::comb::ExtractOp::create(builder, op->getLoc(), input, lowBit,
                                          numBits);
  } else {
    // Anything else cannot be changed:
    // put one cast to the integer right after it
    OpBuilder builder(producer.getContext());
    if (op) {
      builder.setInsertionPointAfter(op);
    } else {
      // An input of the top-level module has no operation to go after,
      // so the cast goes at the top of the module
      Block *body = cast<BlockArgument>(producer).getOwner();
      builder.setInsertionPointToStart(body);
    }
    return castTo(builder, producer, integerType);
  }
}

/// Change the type of one port of a module, which is held in its module type
static void setPortType(circt::hw::HWModuleOp hwMod, unsigned portNumber,
                        Type type) {
  // Copy the ports out of the module type, and change the one
  SmallVector<circt::hw::ModulePort> ports(hwMod.getModuleType().getPorts());
  ports[portNumber].type = type;

  // Give the module a new module type, made from the ports
  hwMod.setModuleType(circt::hw::ModuleType::get(hwMod.getContext(), ports));
}

/// A producer now produces an integer in place of an array of 1-bit values.
/// Update everything from it to its consumers to match:
/// each port that the value passes through becomes an integer port,
/// and each consumer's cast to an integer is removed
static void consumeAsInteger(Value integer, const Hierarchy &hierarchy,
                             CastPool &pool) {
  // For each place the integer is used.
  // A use can be removed while iterating, hence the early-increment range
  for (OpOperand &use : llvm::make_early_inc_range(integer.getUses())) {
    Operation *user = use.getOwner();
    Type integerType = integer.getType();

    if (auto instance = llvm::dyn_cast<circt::hw::InstanceOp>(user)) {
      // The integer is connected to an input port of an instance:
      // find the module that the instance instantiates,
      // and the block argument of the module's matching input port
      circt::hw::HWModuleOp hwMod = hierarchy.moduleOf.lookup(instance);
      unsigned inputNumber = use.getOperandNumber();
      BlockArgument argument = hwMod.getBodyBlock()->getArgument(inputNumber);

      // Only if the module's input port is not yet an integer
      if (argument.getType() != integerType) {
        // Make the module's input port an integer in its module type
        unsigned portNumber =
            hwMod.getModuleType().getPortIdForInputId(inputNumber);
        setPortType(hwMod, portNumber, integerType);

        // Make the block argument of the module's input port an integer
        argument.setType(integerType);

        // Continue with what consumes the block argument inside the module
        consumeAsInteger(argument, hierarchy, pool);
      }
    } else if (auto output = llvm::dyn_cast<circt::hw::OutputOp>(user)) {
      // The integer is connected to an output port of its own module,
      // by hw.output: find the module, and the number of that output port
      auto hwMod = cast<circt::hw::HWModuleOp>(output->getParentOp());
      unsigned outputNumber = use.getOperandNumber();
      unsigned portNumber =
          hwMod.getModuleType().getPortIdForOutputId(outputNumber);

      // Only if the module's output port is not yet an integer,
      // and has consumers: they are at the instances of the module,
      // of which the top-level module has none
      Type portType = hwMod.getModuleType().getPorts()[portNumber].type;
      bool hasInstances = hierarchy.instancesOf.count(hwMod);
      if (portType != integerType && hasInstances) {
        // Make the module's output port an integer in its module type
        setPortType(hwMod, portNumber, integerType);

        // For each instance of the module
        for (circt::hw::InstanceOp instance :
             hierarchy.instancesOf.lookup(hwMod)) {
          // Make the result of the instance's matching output port an integer
          Value result = instance.getResult(outputNumber);
          result.setType(integerType);

          // Continue with what consumes that result
          consumeAsInteger(result, hierarchy, pool);
        }
      }
    } else if (auto bitcast = llvm::dyn_cast<circt::hw::BitcastOp>(user);
               bitcast && bitcast.getType() == integerType) {
      // The integer is consumed by a cast to the type it already has,
      // which is a cast that does nothing.
      // What consumed the cast consumes the integer directly
      bitcast.replaceAllUsesWith(integer);

      // Delete the cast, and remove it from the pool
      pool.remove(bitcast);
      bitcast.erase();
    }
  }
}

/// Get a value as an integer: the value itself if it is one already,
/// and otherwise a cast of it to the integer, which is added to the pool
static Value asInteger(OpBuilder &builder, Value value, CastPool &pool) {
  if (isa<IntegerType>(value.getType())) {
    return value;
  } else {
    Value integer = castTo(builder, value, integerFor(value.getType()));
    pool.insert(integer.getDefiningOp());
    return integer;
  }
}

/// Make an operation consume integers,
/// in the operands where it consumed arrays of 1-bit values.
/// The operation is changed into its form on integers if it has one,
/// and otherwise one cast to the array is put right before it
/// for each of those operands
static void consumeIntegers(Operation *op, ArrayRef<OpOperand *> operands,
                            CastPool &pool) {
  OpBuilder builder(op);

  if (auto get = llvm::dyn_cast<circt::hw::ArrayGetOp>(op);
      get && get.getIndex().getDefiningOp<circt::hw::ConstantOp>()) {
    // hw.array_get at a constant index: comb.extract of that one bit
    auto index = get.getIndex().getDefiningOp<circt::hw::ConstantOp>();
    unsigned bit = index.getValue().getZExtValue();
    Value extracted = circt::comb::ExtractOp::create(builder, op->getLoc(),
                                                     get.getInput(), bit, 1);

    // What consumed the hw.array_get consumes the comb.extract
    get.replaceAllUsesWith(extracted);
    op->erase();
  } else if (auto slice = llvm::dyn_cast<circt::hw::ArraySliceOp>(op);
             slice &&
             slice.getLowIndex().getDefiningOp<circt::hw::ConstantOp>()) {
    // hw.array_slice at a constant index: comb.extract of that range of bits,
    // of as many bits as the slice has elements
    auto lowIndex = slice.getLowIndex().getDefiningOp<circt::hw::ConstantOp>();
    unsigned lowBit = lowIndex.getValue().getZExtValue();
    unsigned numBits = integerFor(slice.getType()).getIntOrFloatBitWidth();
    Value extracted = circt::comb::ExtractOp::create(
        builder, op->getLoc(), slice.getInput(), lowBit, numBits);

    // The slice produced an array, so put a cast to the array
    // after the comb.extract, and add the cast to the pool
    Value array = castTo(builder, extracted, slice.getType());
    pool.insert(array.getDefiningOp());

    // What consumed the hw.array_slice consumes the cast
    slice.replaceAllUsesWith(array);
    op->erase();
  } else if (auto mux = llvm::dyn_cast<circt::comb::MuxOp>(op)) {
    // comb.mux between two arrays of 1-bit values:
    // comb.mux between two integers, with the same select
    Value trueValue = asInteger(builder, mux.getTrueValue(), pool);
    Value falseValue = asInteger(builder, mux.getFalseValue(), pool);
    Value integer = circt::comb::MuxOp::create(
        builder, op->getLoc(), mux.getCond(), trueValue, falseValue);

    // The mux produced an array, so put a cast to the array
    // after the new mux, and add the cast to the pool
    Value array = castTo(builder, integer, mux.getType());
    pool.insert(array.getDefiningOp());

    // What consumed the old mux consumes the cast
    mux.replaceAllUsesWith(array);
    op->erase();
  } else if (auto wire = llvm::dyn_cast<circt::hw::WireOp>(op)) {
    // hw.wire of an array of 1-bit values:
    // hw.wire of an integer, with the same name
    Value input = asInteger(builder, wire.getInput(), pool);
    Value integer = circt::hw::WireOp::create(builder, op->getLoc(), input,
                                              wire.getNameAttr());

    // The wire produced an array, so put a cast to the array
    // after the new wire, and add the cast to the pool
    Value array = castTo(builder, integer, wire.getType());
    pool.insert(array.getDefiningOp());

    // What consumed the old wire consumes the cast
    wire.replaceAllUsesWith(array);
    op->erase();
  } else if (auto concat = llvm::dyn_cast<circt::hw::ArrayConcatOp>(op)) {
    // hw.array_concat of arrays of 1-bit values: comb.concat of integers,
    // in the same order, as both take the highest index first
    SmallVector<Value> inputs;
    for (Value input : concat.getInputs())
      inputs.push_back(asInteger(builder, input, pool));
    Value integer =
        circt::comb::ConcatOp::create(builder, op->getLoc(), inputs);

    // The concat produced an array, so put a cast to the array
    // after the comb.concat, and add the cast to the pool
    Value array = castTo(builder, integer, concat.getType());
    pool.insert(array.getDefiningOp());

    // What consumed the hw.array_concat consumes the cast
    concat.replaceAllUsesWith(array);
    op->erase();
  } else {
    // Anything else cannot be changed:
    // put one cast to the array right before it, for each of the operands
    for (OpOperand *operand : operands) {
      Value integer = operand->get();
      operand->set(castTo(builder, integer, arrayFor(integer.getType())));
    }
  }
}

/// Examine one cast from an array of 1-bit values to an integer.
/// If the array is only ever consumed as an integer,
/// make whatever produces the array produce an integer,
/// and update everything from there to the consumers to match
static void examineCastToInteger(circt::hw::BitcastOp cast,
                                 const Hierarchy &hierarchy, CastPool &pool) {
  // Find what produces the array that the cast reads, and what consumes it.
  // Only continue if the array is only ever consumed as an integer
  llvm::SetVector<Value> producers;
  SmallVector<OpOperand *> consumers;
  if (isOnlyConsumedAsInteger(cast, hierarchy, producers, consumers)) {
    // For each value that produces the array
    for (Value producer : producers) {
      // Make it produce an integer
      Value integer = produceAsInteger(producer, pool);

      // Everything that used the array uses the integer.
      // The exception is the cast that produceAsInteger puts after a producer
      // it cannot change, which has to keep reading the array
      producer.replaceAllUsesExcept(integer, integer.getDefiningOp());

      // Update everything from the integer to its consumers to match
      consumeAsInteger(integer, hierarchy, pool);

      // Delete the operation that produced the array if nothing reads it now,
      // and remove it from the pool, in case it is a cast waiting there
      Operation *op = producer.getDefiningOp();
      if (op && op->use_empty()) {
        pool.remove(op);
        op->erase();
      }
    }
  }
}

/// Examine one cast from an integer to an array of 1-bit values.
/// If the array is only ever produced from an integer,
/// remove the casts that produce it,
/// update everything from there to the consumers to match,
/// and make each consumer consume an integer
static void examineCastToArray(circt::hw::BitcastOp cast,
                               const Hierarchy &hierarchy, CastPool &pool) {
  // Find what consumes the array that the cast produces, and what produces it.
  // Only continue if the array is only ever produced from an integer
  SmallVector<OpOperand *> consumers;
  llvm::SetVector<Value> producers;
  if (isOnlyProducedFromInteger(cast, hierarchy, consumers, producers)) {
    // Group the consumers by operation.
    // An operation is changed only once,
    // even if several of its operands are consumers, like both values of a mux.
    // A consumer that is a cast to an integer is left out,
    // as consumeAsInteger removes those casts below
    llvm::MapVector<Operation *, SmallVector<OpOperand *>> operandsOf;
    for (OpOperand *consumer : consumers)
      if (!isCastToInteger(consumer))
        operandsOf[consumer->getOwner()].push_back(consumer);

    // For each cast that produces the array
    for (Value producer : producers) {
      // Everything that used the array uses the integer that the cast reads
      auto producerCast =
          llvm::cast<circt::hw::BitcastOp>(producer.getDefiningOp());
      Value integer = producerCast.getInput();
      producer.replaceAllUsesWith(integer);

      // Delete the cast, and remove it from the pool
      pool.remove(producerCast);
      producerCast.erase();

      // Update everything from the integer to its consumers to match
      consumeAsInteger(integer, hierarchy, pool);
    }

    // Make each consuming operation consume integers
    for (auto &[op, operands] : operandsOf)
      consumeIntegers(op, operands, pool);
  }
}

/// Reduce the casts between arrays of 1-bit values and integers
/// in a design that is lowered to hw modules
void banba::reduceCastsToIntegers(ModuleOp modOp) {
  Hierarchy hierarchy = findHierarchy(modOp);

  // The pool starts as every cast in the design
  // between an array of 1-bit values and an integer, in either direction
  CastPool pool;
  for (auto hwMod : modOp.getOps<circt::hw::HWModuleOp>()) {
    for (auto cast : hwMod.getBodyBlock()->getOps<circt::hw::BitcastOp>()) {
      Type from = cast.getInput().getType();
      Type to = cast.getType();
      bool toInteger = isArrayOf1BitValues(from) && isa<IntegerType>(to);
      bool toArray = isa<IntegerType>(from) && isArrayOf1BitValues(to);
      if (toInteger || toArray)
        pool.insert(cast);
    }
  }

  // Examine one cast at a time until the pool is empty.
  // Examining a cast can add new casts to the pool
  while (!pool.empty()) {
    auto cast = llvm::cast<circt::hw::BitcastOp>(pool.pop_back_val());
    if (isa<IntegerType>(cast.getType())) {
      examineCastToInteger(cast, hierarchy, pool);
    } else {
      examineCastToArray(cast, hierarchy, pool);
    }
  }
}
