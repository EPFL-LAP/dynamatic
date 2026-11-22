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

// Banba operations correspond to sub-modules
// each operation is responsible for adding its own internal sub-operations
// and wiring them together
// as well as possibly adding some direct RTL logic
// (or purely RTL logic if an operation has no sub-modules)
//
// This interface declares that an operation contains code to handle
// building its level of the hierarchy
//
// We call the constructRegion function on an operation to have that
// code run
def ConstructsRegion : OpInterface<"ConstructsRegion"> {
  let cppNamespace = "::banba";
  let methods = [
    InterfaceMethod<"Populate this operation's internal region "
                    "with its nested region ops, any comb/seq implementation, "
                    "and wire them together correctly.",
                    "void", "constructRegion">,
    InterfaceMethod<"The single block of the op's region.",
                    "::mlir::Block *", "getBodyBlock", (ins),
                    /*methodBody=*/[{}],
                    /*defaultImpl=*/[{ return &$_op->getRegion(0).front(); }]>,
  ];
}

// Lets an operation give the names of its inputs and outputs,
// in the order in which its declaration lists them.
// The lowering to hardware uses them as the names of the module's ports.
// The functions are generated per operation by banba-tblgen
def PortNames : OpInterface<"PortNames"> {
  let cppNamespace = "::banba";
  let methods = [
    InterfaceMethod<
        "The names of the operation's inputs, in order.",
        "SmallVector<std::string>", "getOperandNames">,
    InterfaceMethod<
        "The names of the operation's outputs, in order.",
        "SmallVector<std::string>", "getResultNames">,
  ];
}

#endif // BANBA_INTERFACES_TD
