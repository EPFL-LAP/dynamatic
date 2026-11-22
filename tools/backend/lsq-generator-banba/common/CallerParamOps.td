#ifndef BANBA_CALLER_PARAM_OPS_TD
#define BANBA_CALLER_PARAM_OPS_TD

// OR Reduction:
// OR-reduces a list of values into one, as a balanced binary tree.
// On hw.array values the OR is elementwise:
// each is cast to a flat integer, they are OR-reduced once,
// and the result is cast back to the array type
// (hardware-identical to per-element gates)
def OrReduceOp : CallerParamComponent<"or_reduce",
    [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<AnyTypeOf<[Int, ArrayInt]>>:$inputs);
  let results = (outs AnyTypeOf<[Int, ArrayInt]>:$reduced);
}

// The result `selected` has the element type of the hw.array `inputs`.
// MLIR checks this, and uses it to work out the result type
// when the operation is created without one
def SelectedHasElementTypeOfInputs : TypesMatchWith<
    "selected has the element type of inputs", "inputs", "selected",
    "::llvm::cast<::circt::hw::ArrayType>($_self).getElementType()">;

// One-Hot Mux:
// one-hot selection between the elements of a hw.array:
// each element is selected against zero on its select bit,
// and the terms OR-reduced (a one-hot and-or mux).
// The select is one bit per input, as an array of bits or an integer
def OneHotMuxOp : CallerParamComponent<"one_hot_mux",
    [SelectedHasElementTypeOfInputs]> {
  let arguments = (ins
      AnyTypeOf<[Int, ArrayI1]>:$one_hot_sel,
      ArrayInt:$inputs);
  let results = (outs Int:$selected);

  // Verify that there is one select bit per input
  let hasVerifier = 1;
}

// WrapSub: a - b modulo num_entries.
// We need to store the modulus on the op, as the constructRegion function
// needs to know it in order to build the correct RTL logic
def WrapSubOp : CallerParamComponent<"wrap_sub",
    [SameOperandsAndResultType]> {
  let arguments = (ins
      // Inputs
      Int:$a,
      Int:$b,
      // Modulo as meta data
      I32Attr:$modulo);

  // Output
  let results = (outs Int:$difference);

  // Verify a, b, and difference are clog2(modulo) bits
  let hasVerifier = 1;
}

// Rotate the elements of an array left by `amount`, cyclically, as a barrel
def CyclicLeftShiftOp : CallerParamComponent<"cyclic_left_shift",
    [AllTypesMatch<["data", "shifted"]>]> {
  let arguments = (ins
      // Inputs: array and shift amount
      ArrayInt:$data,
      Int:$amount);

  // Output
  let results = (outs ArrayInt:$shifted);

  let extraClassDeclaration = [{
    /// The comment printed above the operation's module in the SystemVerilog
    static constexpr StringRef ModComment[] = {
        "CyclicLeftShift: rotate an array left by the shift amount"};
  }];

  // Verify amount is clog2(element count of data) bits
  let hasVerifier = 1;
}

#endif // BANBA_CALLER_PARAM_OPS_TD
