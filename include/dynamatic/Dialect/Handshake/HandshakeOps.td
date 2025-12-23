//===- HandshakeOps.td - Handshake operation definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file define Handshake operations in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_OPS_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_OPS_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "dynamatic/Dialect/Handshake/HandshakeAttributes.td"
include "dynamatic/Dialect/Handshake/HandshakeTypes.td"

/// Base class for Handshake dialect operations.
class Handshake_Op<string mnemonic, list<Trait> traits = []>
    : Op<Handshake_Dialect, mnemonic,
         traits #[HasParent<"handshake::FuncOp">,
         DeclareOpInterfaceMethods<NamedIOInterface>,
         DeclareOpInterfaceMethods<ControlInterface>]> {
}

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.  Also, it defines a Dominance-Free Scope
def FuncOp : Op<Handshake_Dialect, "func", [
   IsolatedFromAbove,
   FunctionOpInterface,
   Symbol,
   RegionKindInterface,
   OpAsmOpInterface,
   HasClock
]> {
  let summary = "Handshake dialect function.";
  let description = [{
    The func operation represents a handshaked function.
    This is almost exactly like a standard FuncOp, except that it has
    some extra verification conditions. In particular, each Value must
    only have a single use.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region AnyRegion : $body);

  let skipDefaultBuilders = 1;

  let builders =
       [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() {
      return &getBody();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the names of the arguments to this function.
    ArrayAttr getArgNames() {
      return (*this)->getAttrOfType<ArrayAttr>("argNames");
    }

    /// Returns the names of the results from this function.
    ArrayAttr getResNames() {
      return (*this)->getAttrOfType<ArrayAttr>("resNames");
    }

    /// Returns the argument name at the given index.
    StringAttr getArgName(unsigned idx) {
      return getArgNames()[idx].cast<StringAttr>();
    }

    /// Returns the result name at the given index.
    StringAttr getResName(unsigned idx) {
      return getResNames()[idx].cast<StringAttr>();
    }

    /// Hook for FunctionOpInterface, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError(
            "requires '" + getFunctionTypeAttrName().getValue() +
            "' attribute of function type");
      return success();
    }

    /// Returns the body block of the function.
    Block* getBodyBlock() {
      return &getBody().front();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "handshake";
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// InstanceOp
def InstanceOp : Handshake_Op<"instance", [
    CallOpInterface,
    HasClock,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "module instantiate operation";
  let description = [{
    The `instance` operation represents the instantiation of a module.  This
    is similar to a function call, except that different instances of the
    same module are guaranteed to have their own distinct state.
    The instantiated module is encoded as a symbol reference attribute named
    "module". An instance operation takes a control input as its last argument
    and returns a control output as its last result.

    Example:

    ```mlir
    %2:2 = handshake.instance @my_add(%0, %1, %ctrl) : (f32, f32, !handshake.control) -> (f32, !handshake.control)
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$module, Variadic<AnyType>:$opOperands);
  let results = (outs Variadic<AnyType>);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    (ins "FuncOp":$module, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", SymbolRefAttr::get(module));
      $_state.addTypes(module.getResultTypes());
  }]>, OpBuilder<
    (ins "SymbolRefAttr":$module, "TypeRange":$results,
     CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", module);
      $_state.addTypes(results);
  }]>, OpBuilder<
    (ins "StringRef":$module, "TypeRange":$results,
     CArg<"ValueRange", "{}">:$operands), [{
       build($_builder, $_state,
             SymbolRefAttr::get($_builder.getContext(), module), results,
             operands);
  }]>];

  let extraClassDeclaration = [{
    FunctionType getModuleType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the module of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("module");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getModuleAttrName(), callee.get<mlir::SymbolRefAttr>());
    }

    /// Get the control operand of this instance op
    Value getControl() {
      return getOperands().back();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOpOperandsMutable();
    }

  }];

  let assemblyFormat = [{
    $module `(` $opOperands `)` attr-dict `:` functional-type($opOperands, results)
  }];
}

def BufferOp : Handshake_Op<"buffer", [
  HasClock, SameOperandsAndResultType,
  BufferLikeOpInterface 
]> {
  let summary = "buffer operation";
  let description = [{
    Represents an arbitrary buffer whose properties (e.g., type, latency, and number of
    slots) are dictated by HW parameters.
  }];

  let arguments = (ins HandshakeType:$operand,
                        Handshake_BufferTypeAttr:$bufferType,
                        ConfinedAttr<I64Attr, [IntMinValue<1>]>:$numSlots
                        );

  let results = (outs HandshakeType:$result);

  let extraClassDeclaration = [{
    // can a data token bypass the buffer
    bool isBypassDV(){
      switch (getBufferType()) {
        case BufferType::ONE_SLOT_BREAK_DV:
        case BufferType::FIFO_BREAK_DV:
        case BufferType::SHIFT_REG_BREAK_DV:
        case BufferType::ONE_SLOT_BREAK_DVR:
          return false;
        case BufferType::ONE_SLOT_BREAK_R:
        case BufferType::FIFO_BREAK_NONE:
          return true;
      }
      llvm_unreachable("Unknown BufferType in isBypassDV()");
    }

    int getLatencyDV(){
      switch (getBufferType()) {
        case BufferType::ONE_SLOT_BREAK_R:
        case BufferType::FIFO_BREAK_NONE:
          return 0;
        case BufferType::ONE_SLOT_BREAK_DV:
        case BufferType::FIFO_BREAK_DV:
        case BufferType::ONE_SLOT_BREAK_DVR:
          return 1;
        case BufferType::SHIFT_REG_BREAK_DV:
          return getNumSlots();
      }
      llvm_unreachable("Unknown BufferType in getLatencyDV()");
    }
  }];


  let builders = [
    OpBuilder<(
      ins
        "Value":$operand,
        "int":$numSlots,
        "::dynamatic::handshake::BufferType":$bufferType
    ), [{
      $_state.addOperands(operand);
      $_state.addTypes(operand.getType());
      $_state.addAttribute("bufferType", ::dynamatic::handshake::BufferTypeAttr::get($_builder.getContext(), bufferType));
      $_state.addAttribute("numSlots", $_builder.getI64IntegerAttr(numSlots));
    }]>
  ];


  let assemblyFormat = [{
    $operand `,` `bufferType` `=` $bufferType `,` `numSlots` `=` $numSlots attr-dict `:` custom<HandshakeType>(type($operand))
  }];

  let hasVerifier = 1;
}

def InitOp : Handshake_Op<"init", [
  HasClock, SameOperandsAndResultType
]> {
  let summary = "init operation";
  let description = [{
    A single-slot buffer which contains a token in its reset state.
    Similar to a buffer, its timing behavior is configurable.
  }];

  let arguments = (ins HandshakeType:$operand);
  let results = (outs HandshakeType:$result);

  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral INIT_TOKEN_ATTR_NAME = "INIT_TOKEN",
                                           TIMING_ATTR_NAME = "TIMING";
  }];

  let assemblyFormat = [{
    $operand attr-dict `:` custom<HandshakeType>(type($operand))
  }];
}

def NDWireOp : Handshake_Op<"ndwire", [
  HasClock, SameOperandsAndResultType
]> {
  let summary = "non-deterministic wire operation";
  let description = [{
    Represents a wire that non-determinastically stalls the propagation of a signal.
  }];

  let arguments = (ins HandshakeType:$operand);
  let results = (outs HandshakeType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<HandshakeType>(type($operand))
  }];
}

class Handshake_ForkOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Op<mnemonic, traits # [
    Pure, HasClock, SameOperandsAndResultType
]> {
  let arguments = (ins HandshakeType:$operand);
  let results = (outs Variadic<HandshakeType>:$result);

  let builders = [OpBuilder<
    (ins "Value":$operand, "unsigned":$size), [{
      $_state.addOperands(operand);
      SmallVector<Type> resultTypes {size, operand.getType()};
      $_state.addTypes(resultTypes);
    }]>
  ];

  let assemblyFormat = [{
    custom<SingleTypedHandshakeOp>($operand, attr-dict,
                                   type($operand), type($result))
  }];
}

def ForkOp : Handshake_ForkOp<"fork", [
  EagerForkLikeOpInterface 
]> {
  let summary = "eager fork operation";
  let description = [{
    A single input is replicated to N outputs and distributed to each
    output as soon as the corresponding successor is available.

    Example:

    ```mlir
    %1:2 = fork [2] %0 : !handshake.channel<i32>
    ```
  }];
  let extraClassDeclaration = [{
    int getNumEagerOutputs() {
      return getNumResults();
    }
  }];
}

def LazyForkOp : Handshake_ForkOp<"lazy_fork"> {
  let summary = "lazy fork operation";
  let description = [{
    A single input is replicated to N outputs and distributed to each
    output when all successors are available.

    Example:

    ```mlir
    %1:2 = lazy_fork [2] %0 : !handshake.channel<i32>
    ```
  }];
}

def MergeOp : Handshake_Op<"merge", [
  Pure,
  // SameOperandsAndResultsType is a built-in variant of AllTypesMatch with useful
  // builder behavior, supporting variadic operands.
  // MergeOp imposes this constraint to enforce that all the data types and
  // extra signals of operands and results match, while MuxOp and CMergeOp only
  // require that the data types match.
  SameOperandsAndResultType,
  VariadicHasElement<"dataOperands">,
  DeclareOpInterfaceMethods<MergeLikeOpInterface, ["getDataResult"]>
]> {
  let summary = "merge operation";
  let description = [{
    The merge operation represents a (nondeterministic) merge operation. Any
    input is propagated to the single output. The number of inputs corresponds
    to the number of predecessor blocks.

    Example:

    ```mlir
    %0 = merge %a, %b, %c : !handshake.channel<i32>
    ```
  }];

  let arguments = (ins Variadic<HandshakeType>:$dataOperands);
  let results = (outs HandshakeType:$result);

  let assemblyFormat = [{
    $dataOperands attr-dict `:` custom<HandshakeType>(type($result))
  }];
}

def MuxOp : Handshake_Op<"mux", [
  Pure,
  VariadicHasElement<"dataOperands">,
  AllDataTypesMatchWithVariadic<"dataOperands", ["result"]>,
  AllExtraSignalsMatchWithVariadic<"dataOperands", ["result", "selectOperand"]>,
  // Interface declarations
  DeclareOpInterfaceMethods<MergeLikeOpInterface, ["getDataResult"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>
]> {
  let summary = "mux operation";
  let description = [{
    The mux operation represents a(deterministic) merge operation.
    Operands: select, data0, data1, data2, ...

    The 'select' operand is received from ControlMerge of the same
    block and it represents the index of the data operand that the mux
    should propagate to its single output.  The number of data inputs
    corresponds to the number of predecessor blocks.

    The mux operation is intended solely for control+dataflow selection.
    For purely dataflow selection, use the 'select' operation instead.

    Example:

    ```mlir
    %res = mux %select [%data0, %data1, %data2] :
      !handshake.channel<i32>, !handshake.channel<i32>
    ```
  }];
  let arguments = (ins ChannelType:$selectOperand,
                       Variadic<HandshakeType>:$dataOperands);
  let results = (outs HandshakeType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ControlMergeOp : Handshake_Op<"control_merge", [
  Pure, HasClock,
  VariadicHasElement<"dataOperands">,
  AllDataTypesMatchWithVariadic<"dataOperands", ["result"]>,
  AllExtraSignalsMatchWithVariadic<"dataOperands", ["result", "index"]>,
  // Interface declarations
  BufferLikeOpInterface,
  EagerForkLikeOpInterface,
  DeclareOpInterfaceMethods<MergeLikeOpInterface, ["getDataResult"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "control merge operation";
  let description = [{
    The control_merge operation represents a
    (nondeterministic) control merge.  Any input is propagated to the
    first output and the index of the propagated input is sent to the
    second output.  The number of inputs corresponds to the number of
    predecessor blocks.

    Example:
    ```
    %res, %idx = control_merge %a, %b, %c :
      !handshake.channel<i32>, !handshake.channel<i32>
    ```
  }];
  let extraClassDeclaration = [{
    int getNumSlots() {
      return 1;
    }
    int getNumEagerOutputs() {
      return 2;
    }
  }];

  let arguments = (ins Variadic<HandshakeType>:$dataOperands);
  let results = (outs HandshakeType:$result, ChannelType:$index);

  let builders = [OpBuilder<
    (ins "ValueRange":$operands), [{
      assert(!operands.empty() && "cmerge needs at least one operand");
      $_state.addOperands(operands);

      // Optimize the size of the index result based on the number of operands
      auto dataType = ::dynamatic::handshake::getOptimizedIndexValType(
          $_builder, operands.size());
      auto idxType = ::dynamatic::handshake::ChannelType::get(dataType);
      $_state.addTypes({operands[0].getType(), idxType});
  }]>];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def BranchOp : Handshake_Op<"br", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "branch operation";
  let description = [{
    The branch operation represents an unconditional
    branch.  The single data input is propagated to the single
    successor.  The input must be triggered by some predecessor to
    avoid continous triggering of a successor block.

    Example:

    ```mlir
    %1 = br %0 : !handshake.channel<i32>
    ```
  }];
  let arguments = (ins HandshakeType:$operand);
  let results = (outs HandshakeType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<HandshakeType>(type($result))
  }];
}

def ConditionalBranchOp : Handshake_Op<"cond_br", [
  AllTypesMatch<["dataOperand", "trueResult", "falseResult"]>,
  AllExtraSignalsMatch<["conditionOperand", "dataOperand", "trueResult", "falseResult"]>,
  IsIntSizedChannel<1, "conditionOperand">,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = "conditional branch operation";
  let description = [{
    The cbranch operation represents a conditional
    branch.  The data input is propagated to one of the two outputs
    based on the condition input.

    Example:

    ```mlir
    %true, %false = cond_br %cond, %data : !handshake.channel<i1>,
      !handshake.channel<i32>
    ```
  }];

  let arguments = (ins ChannelType:$conditionOperand,
                       HandshakeType:$dataOperand);
  let results = (outs HandshakeType:$trueResult,
                      HandshakeType:$falseResult);

  let assemblyFormat = [{
    $conditionOperand `,` $dataOperand attr-dict
      `:` type($conditionOperand) `,` custom<HandshakeType>(type($dataOperand))
  }];
  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };
  }];
}

def SinkOp : Handshake_Op<"sink"> {
  let summary = "sink operation";
  let description = [{
    The sink operation discards any data that arrives at its
    input. The sink has no successors and it can continuously consume data.

    Example:

    ```mlir
    sink %data : !handshake.channel<i32>
    ```
  }];

  let arguments = (ins HandshakeType:$operand);
  let assemblyFormat = [{
    $operand attr-dict `:` custom<HandshakeType>(type($operand))
  }];
}

def SourceOp : Handshake_Op<"source", [Pure]> {
  let summary = "source operation";
  let description = [{
    The source operation represents a continuous control-only-token source. The
    source continously sets a 'valid' signal which the successor can consume at
    any point in time.

    Example:

    ```mlir
    %ctrl = source
    ```
  }];

  let results = (outs ControlType:$result);

  let builders = [
    // Unless otherwise specified, output type doesn't have any extra signals.
    OpBuilder<(ins), [{
      $_state.addTypes(ControlType::get($_builder.getContext(), /*extraSignals=*/{}));
    }]>
  ];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def JoinOp : Handshake_Op<"join", [
  SameOperandsAndResultType,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>
]> {
  let summary = "join operation";
  let description = [{
    A control-only synchronizer.  Produces a valid output when all
    inputs become available.

    Example:

    ```mlir
    %0 = join %a, %b, %c : !handshake.control
    ```
  }];

  let arguments = (ins Variadic<ControlType>:$data);
  let results = (outs ControlType:$result);
  let assemblyFormat = "$data attr-dict `:` type($result)";
}

// TODO: Split the input into two parts and explicitly mark the forwarded input,
// once the backend supports variadic channels with zero or one item correctly.
def BlockerOp : Handshake_Op<"blocker", [
  SameOperandsAndResultType
]> {
  let summary = "blocker operation";
  let description = [{
    A data synchronizer. Blocks handshakes until all inputs are valid,
    then forwards the first input.

    Example:

    ```mlir
    %0 = blocker %a, %b, %c : !handshake.channel<i32>
    ```
  }];

  let arguments = (ins Variadic<ChannelType>:$data);
  let results = (outs ChannelType:$result);
  let assemblyFormat = "$data attr-dict `:` type($result)";
}

def NotOp : Handshake_Op<"not", [Pure, SameOperandsAndResultType]> {
  let summary = "Logical negation";
  let description = [{
    Bitwise logical negation.

    Example:

    ```mlir
    %neg = not %value : !handshake.channel<i32>
    ```
  }];
  let arguments = (ins ChannelType:$operand);
  let results = (outs ChannelType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Memory interfaces
//===----------------------------------------------------------------------===//

def MemoryControllerOp : Handshake_Op<"mem_controller", [
  // The memory controller does not accept ControlTypes with extra signals.
  // Handling extra signals is the responsibility of MemPortOps (Load/Store).
  // See the issue #214.
  IsSimpleHandshake<"memStart">,
  IsSimpleHandshake<"ctrlEnd">,
  IsSimpleHandshake<"memEnd">,
  IsSimpleHandshakeVariadic<"inputs">,
  IsSimpleHandshakeVariadic<"outputs">,
  DeclareOpInterfaceMethods<MemoryOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "memory controller (dynamatic)";
  let description = [{
    Each `MemoryControllerOp` represents an interface to an externally defined
    unidimensional memory (i.e., it interfaces a memref input to a Handshake
    function). It receives control signals from each basic block containing
    store operations referencing the wrapped memref; the formers are fed to the
    operation through constants indicating the number of stores the basic block
    will make to the referenced memory region. It also receives load (address
    value) and store (address value + data value) memory accesses, optionally
    fed throuhg an LSQ. It returns a data value for each load request as well as
    a single control signal to indicate basic block completion to the enclosing
    function's end operation.

    Additionally, the `connectedBlocks` attribute contains the ordered list of
    basic blocks (referenced by their unique IDs) that communicate with the
    controller. Every load/store operation is required to be tagged with the ID
    of a basic block present in this list.

    Optionally, a memory controller may act as a middle-person between an LSQ
    and the external memory.

    The order of memory inputs is
    1. For each basic block:
      a. If there is at least one store in the block, a control signal fed
         through a constant indicating the number of store operations in the
         block.
      b. Load/Store access requests from within the block, in program order.
    2. If an LSQ references the same memory region:
      a. Control signals from basic blocks containing store ports to the LSQ but
         not directly to the MC. These have the same semantics as the control
         signals described above.
      b. A single address operand for load accesses coming through the LSQ and
         a single pair of address/data operands for store accesses coming
         through the LSQ.

    The order of memory outputs is
    1. Load results, in program order (i.e., in load accesses order).
    2. If an LSQ references the same memory region, a data result for load
       accesses.
    3. Control signal indicating completion.
    ```
  }];

  let arguments = (ins AnyMemRef:$memRef, ControlType:$memStart,
                       Variadic<ChannelType>:$inputs, ControlType:$ctrlEnd,
                       I32ArrayAttr:$connectedBlocks);
  let results = (outs Variadic<ChannelType>:$outputs, ControlType:$memEnd);

  let builders = [OpBuilder<(ins "Value":$memRef, "Value":$memStart,
                                 "ValueRange":$inputs, "Value":$ctrlEnd,
                                 "ArrayRef<unsigned>":$blocks,
                                 "unsigned":$numLoads)>];
  let hasVerifier = 1;

  // Dispatch SimpleControl signals to custom print and parse
  let assemblyFormat = [{
    `[` $memRef `:` type($memRef) `]` $memStart ` ` `(` $inputs `)` $ctrlEnd
    attr-dict `:` custom<SimpleControl>(type($memStart)) custom<SimpleControl>(type($memEnd)) custom<SimpleControl>(type($ctrlEnd)) functional-type($inputs, $outputs)
  }];

  let extraClassDeclaration = [{
    /// Returns the list of basic block IDs the MC is connected to.
    mlir::SmallVector<unsigned> getMCBlocks() {
      mlir::SmallVector<unsigned> blocks;
      for (mlir::Attribute attr : getConnectedBlocks())
        blocks.push_back(cast<IntegerAttr>(attr).getValue().getZExtValue());
      return blocks;
    }

    /// Returns a convenient data-structure to go over the controls and memory
    /// accesses that are connected to the memory controller.
    dynamatic::MCPorts getPorts();
  }];
}

def LSQOp : Handshake_Op<"lsq", [
  DeclareOpInterfaceMethods<MemoryOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "load-store queue  (dynamatic)";
  let description = [{
    Each `LSQOp` represents an interface to an externally defined unidimensional
    memory. In terms of circuit connectivity and semantics it is quite similar
    to the `MemoryControllerOp` operation (see the latter's documentation), with
    some important differences outlined below.
    - LSQs work on a group basis instead of on a block basis. Basic blocks can
      be considered groups but the opposite is not generally true (i.e., groups
      are more general than blocks). A group must satisfy a "linear dominance"
      constraint, meaning that all the memory accesses it contains must be
      clearly program-orderable, and stored in the LSQ's inputs in that order.
      Additionally, two groups connected to the same LSQ cannot share the same
      control signal, which are determined by the program location of the first
      memory access of each group).
    - Control signals are not fed through a constant and are instead data-less
      signals simply indicating that a new group has started.
    - To ensure proper identification of groups when analyzing the operation's
      operands, the `groupSizes` attribute contains the numberf of load/store
      ports in each LSQ group, in operand order.

    In the absence of an MC on the same memory region, the LSQ talks directly to
    the external memory and its first memory input will be the Handhsake
    function's memref argument that it interfaces in the circuit. In the
    presence of an MC on the same memory region, all load/store requests to the
    LSQ are forwarded to the MC (which talks to the external memory) and the
    first memory input is a control signal.

    The order of memory inputs is
    1. Wrapped memref (optional, if not connected to an MC).
    2. For each LSQ group:
      a. A control signal indicating that the group of accesses has started.
      b. Load/Store access requests from within the block, in program order.
    3. If an MC references the same memory region, a single data operand for
       load accesses coming back from an external memory through the MC.

    The order of results is
    1. Load results, in program order (i.e., in load accesses order).
    2. If an MC references the same memory region, a single address operand for
       load accesses and a single pair of address/data operands for store
       accesses going to the external memory through the MC.
    3. Control signal indicating completion.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs, I32ArrayAttr:$groupSizes);
  let results = (outs Variadic<HandshakeType>:$outputs);

  let builders = [
    OpBuilder<(ins "Value":$memref, "Value":$memStart, "ValueRange":$inputs,
                   "Value":$ctrlEnd, "ArrayRef<unsigned>":$groupSizes, "unsigned":$numLoads)>,
    OpBuilder<(ins "::dynamatic::handshake::MemoryControllerOp":$mcOp,
                   "ValueRange":$inputs, "ArrayRef<unsigned>":$groupSizes,
                   "unsigned":$numLoads)>,
  ];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of sizes of all LSQ groups.
    mlir::SmallVector<unsigned> getLSQGroupSizes() {
      mlir::SmallVector<unsigned> sizes;
      for (mlir::Attribute attr : getGroupSizes())
        sizes.push_back(cast<IntegerAttr>(attr).getValue().getZExtValue());
      return sizes;
    }

    /// Returns a convenient data-structure to go over the controls and memory
    /// accesses that are connected to the LSQ.
    dynamatic::LSQPorts getPorts();

    /// Determines whether the LSQ is connected to an MC.
    bool isConnectedToMC() {
      return !isa<mlir::MemRefType>(getInputs().front().getType());
    }

    /// Returns the MC connected to the LSQ, if it exists.
    handshake::MemoryControllerOp getConnectedMC();

    /// Identifies the subset of the control operation's results that are part
    /// of the control path to the LSQ interface. The control operations'
    /// results that are not of type `handshake::ControlType` are ignored and
    /// will never be part of the returned vector. Typically, one would call
    /// this function on a (lazy-)fork directly providing a group allocation
    /// signal to the LSQ to inquire about other fork results that would trigger
    /// other group allocations. As such the method assumed that the IR is
    /// materialized when called. The returned values are guaranteed to be in
    /// the same order as the control operation's results.
    SmallVector<Value> getControlPaths(Operation *ctrlOp);
  }];
}

//===----------------------------------------------------------------------===//
// Memory ports
//===----------------------------------------------------------------------===//

class Handshake_MemPortOp<
    string mnemonic, list<Trait> customTraits, list<OpBuilder> customBuilders> :
  Handshake_Op<mnemonic, customTraits # [
    MemPortOpInterface,
    AllDataTypesMatch<["address", "addressResult"]>,
    AllDataTypesMatch<["data", "dataResult"]>,
    DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>,
    IsIntChannel<"address">,
    IsIntChannel<"addressResult">
  ]
> {
  let arguments = (ins ChannelType:$address, ChannelType:$data);
  let results = (outs ChannelType:$addressResult, ChannelType:$dataResult);

  let builders = customBuilders # [
    // Unless otherwise specified, addressResult/dataResult type is the same as
    // address/data type.
    OpBuilder<(ins "::mlir::Value":$address, "::mlir::Value":$data), [{
      $_state.addOperands({address, data});
      // Specify {addressResult type, dataResult type}
      $_state.addTypes({address.getType(), data.getType()});
    }]>,
  ];

  let assemblyFormat = [{
    `[` $address `]` $data attr-dict `:` type($address)`,` type($data) `,`
    type($addressResult) `,` type($dataResult)
  }];
}

def LoadOp : Handshake_MemPortOp<"load", [
  AllExtraSignalsMatch<["address", "dataResult"]>,
  // In LoadOp, addressResult and data are connected to a memory controller.
  IsSimpleHandshake<"addressResult">,
  IsSimpleHandshake<"data">,
  BufferLikeOpInterface,
], [
  // TODO: Please add comments on why this builder is necessary
  OpBuilder<(ins "MemRefType":$memrefType, "Value":$address), [{
    // Address (data operand needs to be added later)
    // TODO: Please describe to us when and where the data operand is added
    $_state.addOperands(address);

    // Data and address results
    $_state.types.push_back(address.getType());
    $_state.types.push_back(::dynamatic::handshake::ChannelType::get(
      memrefType.getElementType()));
  }]>
]> {
  let summary = "load port for memory interface";
  let description = [{
    Represents a load memory port which sends load requests to a memory
    interface. It receives an address from a dataflow predecessor and a data
    value from the memory interface that eventually holds the loaded data. It
    returns an address that is sent to the memory interface as well as a data
    value that is sent to its dataflow successor.

    The order of operands is
    1. address (from predecessor)
    2. data value (from memory interface)

    The order of results is
    1. address (to memory interface).
    2. data value (to successor)

    Example:

    ```mlir
    %addrOut, %dataOut = load [%addrIn] %dataFromMem : <i32>, <i32>
    ```
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getNumOperands() && "index too high");
      return (idx == 0) ? "addrIn" : "dataFromMem";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < getNumResults() && "index too high");
      return (idx == 0) ? "addrOut" : "dataOut";
    }
  }];

  let extraClassDeclaration = [{
    int getNumSlots() {
      return 2;
    }
  }];

}

def StoreOp : Handshake_MemPortOp<"store", [
  AllExtraSignalsMatch<["address", "data"]>,
  // In StoreOp, addressResult and dataResult are connected to a memory controller.
  IsSimpleHandshake<"addressResult">,
  IsSimpleHandshake<"dataResult">
], []> {
  let summary = "store operation for memory controller (MC)";
  let description = [{
    Represents a store memory port which sends store requests to a memory
    interface. It receives an address and a data value to store in memory from
    its dataflow predecessors. It returns the value to store and an address,
    both of which are sent to the memory interface.

    The order of operands is
    1. address (from predecessor)
    2. data value (from predecessor)

    The order of results is
    1. address (to memory interface).
    2. data value (to memory interface)

    Example:

    ```mlir
    %addrOut, %dataToMem = store [%addrIn] %dataIn : <i32>, <i32>
    ```
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned int idx) {
      return (idx == 0) ? "addrIn" : "dataIn";
    }

    std::string $cppClass::getResultName(unsigned int idx) {
      return (idx == 0) ? "addrOut" : "dataToMem";
    }
  }];
}

//===----------------------------------------------------------------------===//
// Memory hierarchy
//===----------------------------------------------------------------------===//

def RAMOp : Handshake_Op<"ram", []>  {
  let arguments = (ins ElementsAttr:$initialValue);

  let description = [{
    A placeholder operation for an RAM or ROM instantiated inside the handshake
    function. It holds the shape and the type of the stored elements and also
    has an "ElementsAttr" attribute that stores the initial (reset) content of
    the memory.

    This operation is converted from get_global ops or alloca ops in the
    built-in memref dialect (see "CfToHandshake" pass).

    Assumptions:
    - Currently, if the global op does not have an initial value, we initialize
      the RAM content with all zeros (and 0.0 if the elements are FPs).

    Limitations:
    - Currently this RAMOp always gets mapped into a RAM with 1 read port and 1
      write port. 

    TODO: 
    - Enum attribute to indicate the architecture of the memory op.
    - ROM support.
    - Multiple (at most 2) reads and writes within the same clock cycle.
    - Map ROM to BRAM or wide MUX.

    Example: MLIR format

    ```mlir
    %1 = handshake.ram() {handshake.name = "ram0", initialValue = ...} : () -> memref<100xi32>
    ```

    Example: Using handshake.RAMOp in the C++ API.
    ```c++
    // Building a new RAMOp
    builder.create<handshake::RAMOp>(type, elementAttr);

    // Access its initialValue
    std::optional<ElementsAttr> initialValue = ramOp.getInitialValue();
    ```
  }];

  let results = (outs AnyStaticShapeMemRef:$result);
}

//===----------------------------------------------------------------------===//
// Terminator
//===----------------------------------------------------------------------===//

def EndOp : Handshake_Op<"end", [
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  Terminator
]> {
  let summary = "function endpoint (dynamatic)";
  let description = [{
    Terminator of every Handshake function. Its operands represent the parent
    Handshake function's results and must thefore match their types.

    Example:

    ```mlir
    end %res1, %res2, %res3 : <i32>, <i64>, <>
    ```
  }];

  let arguments = (ins Variadic<HandshakeType>:$inputs);

  let assemblyFormat = [{
    attr-dict ($inputs^ `:` custom<HandshakeTypes>(type($inputs)))?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Speculation
//===----------------------------------------------------------------------===//

def SpeculatorOp : Handshake_Op<"speculator", [
  // Expect a spec tag for dataIn, since Speculator is used in a loop.
  HasValidSpecTag<"dataIn">,
  HasValidSpecTag<"dataOut">,
  AllTypesMatch<["dataIn", "dataOut"]>,
  IsSimpleHandshake<"saveCtrl">,
  IsIntSizedChannel<1, "saveCtrl">,
  IsSimpleHandshake<"commitCtrl">,
  IsIntSizedChannel<1, "commitCtrl">,
  IsSimpleHandshake<"SCSaveCtrl">,
  IsIntSizedChannel<3, "SCSaveCtrl">,
  IsSimpleHandshake<"SCCommitCtrl">,
  IsIntSizedChannel<3, "SCCommitCtrl">,
  IsSimpleHandshake<"SCIsMisspec">,
  IsIntSizedChannel<1, "SCIsMisspec">,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = "Central control unit of the speculative circuit.";
  let description = [{
    The speculator produces speculative tokens when real data has
    not yet arrived. The speculator is also responsible for deciding
    the correctness of its speculation.

    It needs to send control tokens to save units and commit units
    ($saveCtrl, $commitCtrl). For save-commit units, three control
    signals are needed. $SCSaveCtrl is connected directly, while
    $SCCommitCtrl replicates the branches that $dataOut
    follows and might not reach a save-commit. $SCIsMisspec is needed
    in the replicated branches to discard the latter signal.

    Example:

    ```mlir
    %dataOut, %saveCtrl, %commitCtrl, %SCSaveCtrl, %SCCommitCtrl, %SCIsMisspec
      = speculator[trigger] %dataIn : <[spec: i1]>, <i1, [spec: i1]>,
        <i1, [spec: i1]>, <i1>, <i1>, <i3>, <i3>, <i1>
    ```
  }];

  let arguments = (ins ChannelType:$dataIn,
                       ControlType:$trigger,
                       UI32Attr:$fifoDepth);
  let results = (outs ChannelType:$dataOut,
                      ChannelType:$saveCtrl, ChannelType:$commitCtrl,
                      ChannelType:$SCSaveCtrl,
                      ChannelType:$SCCommitCtrl,
                      ChannelType:$SCIsMisspec);

  let assemblyFormat = [{
    `[` $trigger `]` $dataIn attr-dict `:` type($trigger) `,` type($dataIn) `,`
    type($dataOut) `,` type($saveCtrl) `,` type($commitCtrl) `,`
    type($SCSaveCtrl) `,` type($SCCommitCtrl) `,` type($SCIsMisspec)
  }];

  // Infer the type of the control signals
  let builders = [OpBuilder<(ins "Type":$dataOutType, "Value":$dataIn, "Value":$trigger, "unsigned":$fifoDepth), [{
    $_state.addOperands({dataIn, trigger});
    $_state.addAttribute("fifoDepth", $_builder.getUI32IntegerAttr(fifoDepth));

    ChannelType ctrlType = ChannelType::get($_builder.getIntegerType(1));
    ChannelType wideControlType = ChannelType::get($_builder.getIntegerType(3));

    $_state.addTypes({dataOutType, ctrlType, ctrlType, wideControlType,
      wideControlType, ctrlType});
  }]>];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getNumOperands() && "index too high");
      return idx == 0 ? "ins" : "trigger";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < getNumResults() && "index too high");
      switch (idx) {
      case 0:
        return "outs";
      case 1:
        return "ctrl_save";
      case 2:
        return "ctrl_commit";
      case 3:
        return "ctrl_sc_save";
      case 4:
        return "ctrl_sc_commit";
      default:
        return "ctrl_sc_branch";
      }
    }
  }];
}

def SpecSaveOp : Handshake_Op<"spec_save", [
  AllDataTypesMatch<["dataIn", "dataOut"]>,
  AllExtraSignalsMatchExcept<"spec", ["dataIn", "dataOut"]>,
  LacksSpecTag<"dataIn">,
  HasValidSpecTag<"dataOut">,
  IsSimpleHandshake<"ctrl">,
  IsIntSizedChannel<1, "ctrl">,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = "Saves data tokens that interact in the speculative region.";
  let description = [{
    Save units are used to mark the beginning of the speculation region.
    Whenever a speculative token can interact with a non-speculative token,
    the non-speculative token needs to be saved. In case of correct speculation
    or no speculation, the saved token can be dropped.

    On the other hand, if the speculation was incorrect, the saved tokens
    need to be reinserted into the circuit to repeat the previously
    miscalculated computations.

    Example:

    ```mlir
    %dataOut = spec_save[%ctrl] %dataIn : !handshake.channel<i32>, !handshake.channel<i32, [spec: i1]>, !handshake.channel<i1>
    ```
  }];

  let arguments = (ins HandshakeType:$dataIn, ChannelType:$ctrl);
  let results = (outs HandshakeType:$dataOut);
  let assemblyFormat = [{
    `[` $ctrl `]` $dataIn attr-dict `:` type($dataIn) `,` type($dataOut) `,` type($ctrl)
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getOperation()->getNumOperands() && "index too high");
      return idx == 0 ? "ins" : "ctrl";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < getOperation()->getNumResults() && "index too high");
      return "outs";
    }
  }];
}

def SpecCommitOp : Handshake_Op<"spec_commit", [
  AllDataTypesMatch<["dataIn", "dataOut"]>,
  AllExtraSignalsMatchExcept<"spec", ["dataIn", "dataOut"]>,
  HasValidSpecTag<"dataIn">,
  LacksSpecTag<"dataOut">,
  IsSimpleHandshake<"ctrl">,
  IsIntSizedChannel<1, "ctrl">,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = "Stall speculative data tokens until they are resolved.";
  let description = [{
    Commit units are used to mark the end of the speculation region.
    Commit units are used to stall speculative tokens until they receive
    the decision from the speculator. In case the speculation is correct,
    the speculative tokens are converted into non-speculative tokens and
    passed on to the rest of the circuit.
    Otherwise, the speculative tokens are discarded.

    Any non-speculative token can pass through the commit units
    without any stall.

    Example:

    ```mlir
    %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, !handshake.channel<i1>
    ```
  }];

  let arguments = (ins HandshakeType:$dataIn, ChannelType:$ctrl);
  let results = (outs HandshakeType:$dataOut);

  let assemblyFormat = [{
    `[` $ctrl `]` $dataIn attr-dict `:` type($dataIn) `,` type($dataOut) `,` type($ctrl)
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getOperation()->getNumOperands() && "index too high");
      return idx == 0 ? "ins" : "ctrl";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < getOperation()->getNumResults() && "index too high");
      return "outs";
    }
  }];
}

def SpecSaveCommitOp : Handshake_Op<"spec_save_commit", [
  HasValidSpecTag<"dataIn">,
  HasValidSpecTag<"dataOut">,
  AllTypesMatch<["dataIn", "dataOut"]>,
  IsSimpleHandshake<"ctrl">,
  IsIntSizedChannel<3, "ctrl">,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = "Lets all tokens pass and saves a copy of them.";
  let description = [{
    Save-Commits are used when speculation occurs in a loop.
    To increase loop parallelism, the save-commit unit will let both
    speculative and non-speculative tokens pass as well as save
    a copy of them.

    Example:

    ```mlir
    %dataOut = spec_save_commit[%ctrl] %dataIn : i11
    ```
  }];
  let arguments = (ins HandshakeType:$dataIn,
                       ChannelType:$ctrl,
                       UI32Attr:$fifoDepth);
  let results = (outs HandshakeType:$dataOut);
  let assemblyFormat = [{
    `[` $ctrl `]` $dataIn attr-dict `:` type($dataIn) `,` type($ctrl)
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getOperation()->getNumOperands() && "index too high");
      return idx == 0 ? "ins" : "ctrl";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < getOperation()->getNumResults() && "index too high");
      return "outs";
    }
  }];
}

def SpeculatingBranchOp : Handshake_Op<"speculating_branch", [
  AllDataTypesMatch<["dataOperand", "trueResult", "falseResult"]>,
  AllExtraSignalsMatchExcept<"spec", ["tagFromOperand", "dataOperand", "trueResult", "falseResult"]>,
  HasValidSpecTag<"tagFromOperand">,
  HasValidSpecTag<"dataOperand">,
  LacksSpecTag<"trueResult">,
  LacksSpecTag<"falseResult">,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = "speculating branch operation";
  let description = [{
    The speculating branch operation represents a conditional
    branch that decides the condition based on if a token Value is
    speculative or not.

    The $tagFromOperand field is the condition. Depending on wether the data
    is speculative or not, $dataOperand will be send to $trueResult or
    $falseResult. The speculative tag is a fictious attribute.

    Example:
    ```mlir
    %trueResult, %falseResult = speculating_branch[%tagFromOperand]
      %dataOperand : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32, [spec: i1]>,
      !handshake.channel<i32>, !handshake.channel<i32>
    ```
  }];

  let arguments = (ins HandshakeType:$tagFromOperand, HandshakeType:$dataOperand);
  let results = (outs HandshakeType:$trueResult, HandshakeType:$falseResult);
  let assemblyFormat = [{
    `[` $tagFromOperand `]` $dataOperand attr-dict `:`
      type($tagFromOperand) `,` type($dataOperand) `,`
      type($trueResult) `,` type($falseResult)
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getNumOperands() && "index too high");
      return idx == 0 ? "spec_tag_data" : "data";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < getNumResults() && "index too high");
      return idx == 0 ? "trueOut" : "falseOut";
    }
  }];
}

def NonSpecOp : Handshake_Op<"non_spec", [
  AllDataTypesMatch<["dataIn", "dataOut"]>,
  AllExtraSignalsMatchExcept<"spec", ["dataIn", "dataOut"]>,
  LacksSpecTag<"dataIn">,
  HasValidSpecTag<"dataOut">,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = "Adds a non-speculative spec bit.";
  let description = [{
    The non-spec operation has an input channel without a spec bit and an output
    channel with a spec bit. Tokens on the output channel are marked as non-spec.
    It's typically used before a MuxOp or CMergeOp to satisfy their type
    requirements and to indicate the beginning of a speculative region.

    Example:

    ```mlir
    %dataOut = non_spec %dataIn : !handshake.channel<i32> to !handshake.channel<i32, [spec: i1]>
    ```
  }];

  let arguments = (ins HandshakeType:$dataIn);
  let results = (outs HandshakeType:$dataOut);
  let assemblyFormat = [{
    $dataIn attr-dict `:` type($dataIn) `to` type($dataOut)
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getOperation()->getNumOperands() && "index too high");
      return "dataIn";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < getOperation()->getNumResults() && "index too high");
      return "dataOut";
    }
  }];
}

//===----------------------------------------------------------------------===//
// Resource sharing
//===----------------------------------------------------------------------===//

def SharingWrapperOp : Handshake_Op<"sharing_wrapper", [
  SameOperandsAndResultType,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>,
]> {

  let summary = "sharing wrapper operation";
  let description = [{
    The sharing_wrapper operation represents the selecting & distributing logic
    that manages access to a shared unit.

    The sharing wrapper interfaces with the shared unit and the rest of the
    circuit.

    Attributes:
    - $credits: an array of integers that indicates, each integer indicates the
      maximum number of in-flight accesses in the shared resource of one
      operation that share the unit.
    - $numSharedOperands: the number of operands of the shared unit.
    - $latency: the execution latency of the shared unit.

    The convention for the input operands:
    - the first "numSharedOperands * len(credits)" operands are the operands
      from the circuit
    - the last operand is the result computed by the shared unit

    The convention for the output results:
    - the first "len(credits)" results are results dispatched to the rest of the
      circuit
    - the last "numSharedOperands" result is the selected operands that will be
      delivered to the shared unit.

    Future improvement:
    - Add the number of operations that share the unit (now it is assumed to be
      the size of the DenseI64ArrayAttr credits).
    - More verbose separations between the operands/results that interface with
      the rest of the circuit or interface with the shared unit (i.e., add a
      custom interface printing for different operand/result types).
    }];

  let arguments = (ins Variadic<ChannelType> : $dataOperands,
    ChannelType : $sharedOpResult,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$credits,
    ConfinedAttr<I32Attr, [IntMinValue<2>]>:$numSharedOperations,
    ConfinedAttr<I32Attr, [IntMinValue<1>]>:$numSharedOperands,
    ConfinedAttr<I32Attr, [IntMinValue<1>]>:$latency);
  let results = (outs Variadic<ChannelType> : $dataOut);

  let assemblyFormat = [{
    `[` $dataOperands `]` `,` `[` $sharedOpResult `]` attr-dict `:`
      functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Type manipulation
//===----------------------------------------------------------------------===//

def BundleOp : Handshake_Op<"bundle", [Pure]> {
  let summary = [{
    Bundles individual signals into a `handshake::ChannelType` or
    `handshake::ControlType`.
  }];
  let description = [{
    Combines individual signals into a channel-like value, producing upstream
    signals as results and taking downstream signals as arguments. Note
    that combining individual signals into a `handshake::ChannelType` is a
    two-step process.
    1. First, bundle a `i1` value, which produces a `!handshake.control` and
       `i1` (representing the upstream ready signal) as results.
    2. Then, bundle the `!handshake.control` along with a value representing the
       data signal (of a compatible signal type e.g., `i32`), which produces
       a `!handshake.channel<i32>`.

    Example:

    ```mlir
    // Bundling into a channel with a downstream extra signal
    %channel = bundle %ctrl, %data, %extra : _ to <i32, [extra: i1]>

    // -----

    // Bundling into a channel with an upstream extra signal
    %channel, %extra = bundle %ctrl, %data : _ to <i32, [extra: i1 (U)]>

    // -----

    // Bundling into a control-only channel
    %ctrl, %ready = unbundle %valid : (i1) -> _ to <>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$signals);
  let results = (outs HandshakeType:$channelLike,
                      Variadic<SignalType>: $upstreams);

  let skipDefaultBuilders = 1;
  let builders = [
    // Builder for bundling ControlType
    OpBuilder<(ins "::mlir::Value":$valid), [{
      mlir::Type validReadyType = $_builder.getIntegerType(1);
      assert(valid.getType() == validReadyType && "incorrect valid signal");

      $_state.addOperands(valid);
      $_state.addTypes({
        ::dynamatic::handshake::ControlType::get($_builder.getContext()),
        validReadyType
      });
    }]>,
    // Builder for bundling ChannelType
    OpBuilder<(ins "::mlir::Value":$ctrl, "::mlir::Value":$data,
                   "::mlir::ValueRange":$downstreams,
                   "::dynamatic::handshake::ChannelType":$channelType)
    >
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def UnbundleOp : Handshake_Op<"unbundle", [
  Pure, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let summary = [{
    Unbundles a `handshake::ChannelType` or `handshake::ControlType` into
    individual signals.
  }];
  let description = [{
    Splits a channel-like value into its individual signals, producing
    downstream signals as results and taking upstream signals as arguments. Note
    that getting the individual valid and ready signals from a
    `handshake::ChannelType` is a two-step process.
    1. First, unbundle the `!handshake.channel`, which produces a
       `!handshake.control` as result (i.e., the control is considered a
       "downstream bundle").
    2. Then, unbundle the `!handshake.control` along with an `i1` value
       representing the ready signal, which produces an `i1` result representing
       the valid signal.

    Example:

    ```mlir
    // Unbundling a channel with a downstream extra signal
    %ctrl, %data, %extra = unbundle %channel : <i32, [extra: i2]> to _

    // -----

    // Unbundling a channel with an upstream extra signal
    %ctrl, %data = unbundle %channel [%extra] : <i32, [extra: i2 (U)]> to _

    // -----

    // Unbundling a control-only channel
    %valid = unbundle %ctrl [%ready] : <> to _
    ```
  }];

  let arguments = (ins ChannelOrSimpleControl:$channelLike,
                       Variadic<SignalType>: $upstreams);
  let results = (outs Variadic<AnyType>:$signals);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$channel), [{
      $_state.addOperands(channel);

      // Infer return types
      ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
      if (::mlir::succeeded(UnbundleOp::inferReturnTypes($_builder.getContext(),
                    $_state.location, $_state.operands,
                    $_state.attributes.getDictionary($_state.getContext()),
                    $_state.getRawProperties(),
                    $_state.regions, inferredReturnTypes)))
        $_state.addTypes(inferredReturnTypes);
      else
        ::llvm::report_fatal_error("Failed to infer result type(s).");
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ReadyRemoverOp : Handshake_Op<"ready_remover",[
  SameOperandsAndResultType,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = [{
    Rigidifies a channel to simplify the handshake logic.
  }];
  let description = [{
    Rigidifies a channel by setting the output ready signal to 1 and input ready
    signal to a open. This allows the simplification of redundant handshake logic.
  }];

  let arguments = (ins HandshakeType:$channelIn);
  let results = (outs HandshakeType:$channelOut);

  let assemblyFormat = [{
    $channelIn attr-dict `:` custom<HandshakeType>(type($channelIn))
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx = 0) {
      return "ins";
    }

    std::string $cppClass::getResultName(unsigned idx = 0) {
      return "outs";
    }
  }];

}

def ValidMergerOp : Handshake_Op<"valid_merger",[
  AllTypesMatch<["lhsIn", "lhsOut"]>,
  AllTypesMatch<["rhsIn", "rhsOut"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
]> {
  let summary = [{
    Merges the valid signals of two channels.
  }];
  let description = [{
    Merges the valid signal of two channels by propagating only one input
    valid signal to both output valid signals. In this way the valid generation
    of the second input signal can be simplified.
  }];

  let arguments = (ins HandshakeType:$lhsIn, HandshakeType:$rhsIn);
  let results = (outs HandshakeType:$lhsOut, HandshakeType:$rhsOut);

  let assemblyFormat = [{
    $lhsIn `,` $rhsIn attr-dict `:` custom<HandshakeType>(type($lhsIn)) `,` custom<HandshakeType>(type($rhsIn))
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
    assert(idx < getNumOperands() && "index too high");
    return (idx == 0) ? "lhs_ins" : "rhs_ins";
    }

    std::string $cppClass::getResultName(unsigned idx) {
    assert(idx < getNumOperands() && "index too high");
    return (idx == 0) ? "lhs_outs" : "rhs_outs";
    }
  }];

}


#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_OPS_TD
