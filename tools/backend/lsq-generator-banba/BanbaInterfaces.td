#ifndef BANBA_INTERFACES_TD
#define BANBA_INTERFACES_TD

include "mlir/IR/OpBase.td"

// Lets an operation state the exact type
// that each of its inputs and outputs must have for a given config.
// The types are checked against it when the IR is verified
def ExpectedTypes : OpInterface<"ExpectedTypes"> {
  let cppNamespace = "::banba";
  let methods = [
    StaticInterfaceMethod<
        "The types that the operation's inputs must have, in order.",
        "SmallVector<Type>", "getExpectedOperandTypes",
        (ins "ConfigAttr":$config)>,
    StaticInterfaceMethod<
        "The types that the operation's outputs must have, in order.",
        "SmallVector<Type>", "getExpectedResultTypes",
        (ins "ConfigAttr":$config)>,
  ];
}

#endif // BANBA_INTERFACES_TD
