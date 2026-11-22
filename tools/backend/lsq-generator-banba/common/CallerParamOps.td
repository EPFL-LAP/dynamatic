#ifndef BANBA_CALLER_PARAM_OPS_TD
#define BANBA_CALLER_PARAM_OPS_TD

// OR Reduction:
// OR-reduces a list of values into one, as a balanced binary tree
def OrReduceOp : CallerParamComponent<"or_reduce",
    [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<Int>:$inputs);
  let results = (outs Int:$reduced);
}

// One-Hot Mux:
// one-hot selection between the elements of a hw.array:
// each element is selected against zero on its select bit,
// and the terms OR-reduced (a one-hot and-or mux).
// The select is one bit per input, as an array of bits or an integer
def OneHotMuxOp : CallerParamComponent<"one_hot_mux"> {
  let arguments = (ins
      AnyTypeOf<[Int, ArrayI1]>:$one_hot_sel,
      ArrayInt:$inputs);
  let results = (outs Int:$selected);

  // Verify that there is one select bit per input,
  // and that the result has the type of an input
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

#endif // BANBA_CALLER_PARAM_OPS_TD
