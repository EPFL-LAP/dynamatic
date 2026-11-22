#ifndef BANBA_CALLER_PARAM_OPS_TD
#define BANBA_CALLER_PARAM_OPS_TD

// OR Reduction:
// OR-reduces a list of values into one, as a balanced binary tree
def OrReduceOp : CallerParamComponent<"or_reduce",
    [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<Int>:$inputs);
  let results = (outs Int:$reduced);
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

#endif // BANBA_CALLER_PARAM_OPS_TD
