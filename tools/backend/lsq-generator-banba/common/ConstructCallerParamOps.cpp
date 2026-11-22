#include "BanbaOps.h"
#include "BanbaSugar.h"

namespace banba {

void OrReduceOp::constructRegion() {
  BanbaSugarScope scope(getOperation());

  // The list of values to build a set of ORs for.
  // That set of ORs will then produce a list of values
  // which is half as long
  SmallVector<Value> inputs(getInputs().begin(), getInputs().end());

  // Sanity assert
  assert(!inputs.empty() && "cannot reduce an empty list");

  // If nothing to OR, return
  if (inputs.size() == 1) {
    driveReduced(inputs.front());
    return;
  }

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
      driveReduced(outputs.front());
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
  OrReduceOp orReduce = BanbaSugar::createOp<OrReduceOp>(terms);

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

} // namespace banba
