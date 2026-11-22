#include "BanbaOps.h"
#include "BanbaSugar.h"

namespace banba {

void OrReduceOp::constructRegion() {
  BanbaSugarScope scope(getOperation());

  // Sanity assert
  assert(!getInputs().empty() && "cannot reduce an empty list");

  // If nothing to OR, return the single input as it is
  if (getInputs().size() == 1) {
    driveReduced(getInputs().front());
    return;
  }

  // The list of values to build a set of ORs for.
  // That set of ORs will then produce a list of values
  // which is half as long
  SmallVector<Value> inputs;
  for (Value input : getInputs())
    inputs.push_back(BanbaSugar::castToInt(input));

  // The first layer of the OR tree has a number of outputs
  // equal to the largest power-of-two value that is strictly smaller
  // than the number of values to OR-reduce
  //
  // bit_floor finds "largest power of two ≤ x"
  // so we do that on the number-of-values - 1
  unsigned numOutputs = llvm::bit_floor(inputs.size() - 1);

  // Loop until there is 1 value
  while (true) {
    // List of outputs
    SmallVector<Value> outputs(numOutputs);

    // If num inputs is not double num outputs
    // then there are less ORs,
    // and some inputs don't get combined this round
    // e.g. 14 -> 8, there is 6 ORs, and 2 value passed directly
    unsigned numOrs = inputs.size() - numOutputs;

    // Combine as many inputs as possible
    for (unsigned i = 0; i < numOrs; ++i)
      outputs[i] = circt::comb::OrOp::create(BanbaSugarScope::getBuilder(),
                                             BanbaSugarScope::getLoc(),
                                             inputs[i], inputs[i + numOutputs]);

    // Pass the rest through
    for (unsigned i = numOrs; i < numOutputs; ++i)
      outputs[i] = inputs[i];

    // If this was the final round,
    // drive the result with the output
    if (numOutputs == 1) {
      driveReduced(BanbaSugar::castTo(outputs.front(), getReduced().getType()));
      return;
    }

    // Set the outputs of this round to be the inputs of the next round
    inputs = outputs;

    // And set the next number of outputs
    // to be half as many
    numOutputs /= 2;
  }
}

LogicalResult OneHotMuxOp::verify() {
  auto inputsType = cast<circt::hw::ArrayType>(getInputs().getType());

  // Check that there is one select bit per input
  int64_t numSelectBits = circt::hw::getBitWidth(getOneHotSel().getType());
  if (numSelectBits != (int64_t)inputsType.getNumElements())
    return emitOpError() << "has " << numSelectBits << " select bits but "
                         << inputsType.getNumElements() << " inputs";

  // Check that the result has the type of an input
  if (getSelected().getType() != inputsType.getElementType())
    return emitOpError() << "result type must match the type of an input";

  return success();
}

void OneHotMuxOp::constructRegion() {
  BanbaSugarScope scope(getOperation());

  BanbaValue oneHotSel = getOneHotSel();
  BanbaValue inputs = getInputs();
  auto inputsType = cast<circt::hw::ArrayType>(getInputs().getType());

  // Create a zero of the element type of `inputs`
  Value zero = BanbaSugar::zeroValue(inputsType.getElementType());

  // For each element of `inputs`,
  // select between the element and zero on its select bit
  SmallVector<Value> terms;
  for (unsigned i = 0; i < inputsType.getNumElements(); ++i) {
    Value selBit = oneHotSel[i];
    Value element = inputs[i];
    terms.push_back(circt::comb::MuxOp::create(BanbaSugarScope::getBuilder(),
                                               BanbaSugarScope::getLoc(),
                                               selBit, element, zero));
  }

  // OR-reduce the terms, with an OR reduction operation
  OrReduceOp orReduce =
      BanbaSugar::createOp<OrReduceOp>(inputsType.getElementType(), terms);

  // Results
  driveSelected(orReduce.getReduced());
}

// Verify a, b, and difference are clog2(modulo) bits
LogicalResult WrapSubOp::verify() {
  // Modulo 1 doesn't make any sense
  if (getModulo() < 2)
    return emitOpError() << "modulo must be at least 2, got " << getModulo();

  // Get binary width of the modulo
  unsigned expected = Log2_32_Ceil(getModulo());
  if (cast<IntegerType>(getA().getType()).getWidth() != expected)
    return emitOpError() << "a must be clog2(modulo = " << getModulo()
                         << ") = " << expected << " bits";
  if (cast<IntegerType>(getB().getType()).getWidth() != expected)
    return emitOpError() << "b must be clog2(modulo = " << getModulo()
                         << ") = " << expected << " bits";
  if (cast<IntegerType>(getDifference().getType()).getWidth() != expected)
    return emitOpError() << "difference must be clog2(modulo = " << getModulo()
                         << ") = " << expected << " bits";
  return success();
}

// WrapSub: a - b modulo num_entries,
// for circular pointers of clog2(num_entries) bits.
// A power-of-two size wraps for free;
// otherwise we pick a - b when a >= b
// and num_entries - b + a when it would borrow.
void WrapSubOp::constructRegion() {
  BanbaSugarScope scope(getOperation());

  // Get inputs and the modulo
  BanbaValue a = getA(), b = getB();
  unsigned modulo = getModulo();

  // RTL subtraction
  BanbaValue diff = a - b;

  if (llvm::isPowerOf2_32(modulo)) {
    // The subtraction wraps automatically if
    // modulo is a power of 2
    driveDifference(diff);
  } else {
    // Otherwise need extra logic:
    // if a >= b: out = a - b; else: out = (a + modulo) - b

    // RTL: a >= b
    BanbaValue pred = a >= b;

    // Get the modulo as an RTL constant
    BanbaValue moduloConst(modulo, Log2_32_Ceil(modulo));

    // RTL: (a + modulo) - b
    BanbaValue borrowed = (moduloConst + a) - b;

    // Mux based on the predicate
    driveDifference(BanbaSugar::select(pred, diff, borrowed));
  }
}

// Verify amount is clog2(element count of data) bits
LogicalResult CyclicLeftShiftOp::verify() {
  // Gets the binary width of the amount to shift by
  unsigned amountWidth = cast<IntegerType>(getAmount().getType()).getWidth();

  // Get the number of elements in the array
  // also asserts the input type is an array
  unsigned numElements =
      cast<circt::hw::ArrayType>(getData().getType()).getNumElements();

  // Should only be called on the load and store queue
  // so we should never be shifting a 1-element queue
  if (numElements < 2)
    return emitOpError() << "data must have at least 2 elements, got "
                         << numElements;

  // Get the binary width required to fully shift a queue of numElements
  unsigned expected = Log2_32_Ceil(numElements);

  // Verify that the widths match
  if (amountWidth != expected)
    return emitOpError() << "amount is " << amountWidth
                         << " bits, expected clog2(" << numElements
                         << " elements) = " << expected;

  // Verify that the shifted output type is the same
  // as the to-shift input type
  if (getShifted().getType() != getData().getType())
    return emitOpError() << "shifted type must match data type";

  return success();
}

// Cyclic left rotate
void CyclicLeftShiftOp::constructRegion() {
  BanbaSugarScope scope(getOperation());

  // Get inputs
  BanbaValue data = getData();
  BanbaValue amount = getAmount();

  // Get the array info
  auto arrayType = cast<circt::hw::ArrayType>(getData().getType());
  unsigned numElements = arrayType.getNumElements();
  unsigned amountWidth = cast<IntegerType>(getAmount().getType()).getWidth();

  // Convert from one hw.array value
  // to a C++ list holding the value of each of its elements,
  // as the initial set of values to shift
  SmallVector<Value> elements;
  for (unsigned i = 0; i < numElements; ++i)
    elements.push_back(data[i]);

  // For each bit starting with the MSB
  // build a shift
  for (int bit = amountWidth - 1; bit >= 0; --bit) {
    // Potentially shift by 2^n
    unsigned shift = (1u << bit);

    // Get bit n of the amount to shift
    BanbaValue selector = amount[bit];

    // Make an output vector of the result of this shift
    SmallVector<Value> shifted(numElements);

    // For each output
    for (unsigned i = 0; i < numElements; ++i) {
      // Annoying indexing from the python version:
      // we calculate the reverse of the shift by 2^n
      // and then index into the input at that point
      unsigned source = (i + numElements - shift) % numElements;

      // Use a multiplexer to select the shifted input or unshifted input
      // based on whether bit n of amount is 1
      shifted[i] = BanbaSugar::select(selector, elements[source], elements[i]);
    }

    // Set up the next shifter to use the output of the previous shifter
    elements = shifted;
  }

  // Pack into an array type, and drive the result
  driveShifted(BanbaSugar::concatToArray(elements));
}

} // namespace banba
