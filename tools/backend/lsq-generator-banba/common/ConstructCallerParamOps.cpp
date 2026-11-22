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

} // namespace banba
