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
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

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
  DeclareOpInterfaceMethods<ReshapableChannelsInterface, ["getReshapableChannelType"]>,
]> {
  let summary = "buffer operation";
  let description = [{
    Represents an arbitrary buffer whose properties (e.g., latencies, number of
    slots) are dictated by HW parameters. 
  }];
  
  let arguments = (ins HandshakeType:$operand);
  let results = (outs HandshakeType:$result);

  let builders = [OpBuilder<
    (ins "Value":$operand, "const ::dynamatic::handshake::TimingInfo &":$timing,
         "std::optional<unsigned>":$numSlots)>
  ];

  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral NUM_SLOTS_ATTR_NAME = "NUM_SLOTS",
                                           TIMING_ATTR_NAME = "TIMING";
  }];

  let assemblyFormat = [{
    $operand attr-dict `:` custom<HandshakeType>(type($operand))
  }];
}

class Handshake_ForkOp<string mnemonic, list<Trait> traits = []> : 
  Handshake_Op<mnemonic, traits # [
    Pure, HasClock, SameOperandsAndResultType,
    DeclareOpInterfaceMethods<ReshapableChannelsInterface, ["getReshapableChannelType"]>
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

  let extraClassDefinition = [{
    std::pair<handshake::ChannelType, bool>
    $cppClass::getReshapableChannelType() {
      return {mlir::dyn_cast<handshake::ChannelType>(getOperand().getType()),
              true};
    }
  }];
}

def ForkOp : Handshake_ForkOp<"fork"> {
  let summary = "eager fork operation";
  let description = [{
    A single input is replicated to N outputs and distributed to each
    output as soon as the corresponding successor is available.

    Example:

    ```mlir
    %1:2 = fork [2] %0 : !handshake.channel<i32>
    ```
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
  Pure, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<MergeLikeOpInterface, ["getDataResult"]>,
  DeclareOpInterfaceMethods<ReshapableChannelsInterface, ["getReshapableChannelType"]>
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
  DeclareOpInterfaceMethods<MergeLikeOpInterface, ["getDataResult"]>,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<SameExtraSignalsInterface, ["getChannelsWithSameExtraSignals"]>,
  DeclareOpInterfaceMethods<ReshapableChannelsInterface, ["getReshapableChannelType"]>
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
  let arguments = (ins IntChannelType:$selectOperand,
                       Variadic<HandshakeType>:$dataOperands);
  let results = (outs HandshakeType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ControlMergeOp : Handshake_Op<"control_merge", [
  Pure, HasClock,
  DeclareOpInterfaceMethods<MergeLikeOpInterface, ["getDataResult"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>,
  DeclareOpInterfaceMethods<SameExtraSignalsInterface, ["getChannelsWithSameExtraSignals"]>,
  DeclareOpInterfaceMethods<ReshapableChannelsInterface, ["getReshapableChannelType"]>
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
  Pure, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<ReshapableChannelsInterface, ["getReshapableChannelType"]>
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
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>,
  DeclareOpInterfaceMethods<ReshapableChannelsInterface, ["getReshapableChannelType"]>
]> {
  let summary = "conditional branch operation";
  let description = [{
    The cbranch operation represents a conditional
    branch.  The data input is propagated to one of the two outputs
    based on the condition input.

    Example:

    ```mlir
    %true, %false = cond_br %cond, %data : !handshake.control<i1>, 
      !handshake.channel<i32>
    ```
  }];

  let arguments = (ins BoolChannel:$conditionOperand,
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
  
  let assemblyFormat = "attr-dict";
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

def NotOp : Handshake_Op<"not", [Pure, SameOperandsAndResultType,
                                 ReshapableChannelsInterface]> {
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
  let assemblyFormat = [{
    `[` $memRef `:` type($memRef) `]` $memStart ` ` `(` $inputs `)` $ctrlEnd
    attr-dict `:` functional-type($inputs, $outputs)
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

class Handshake_MemPortOp<string mnemonic, Trait PortInterface> :
  Handshake_Op<mnemonic, [PortInterface] # [
    AllTypesMatch<["address", "addressResult"]>,
    AllTypesMatch<["data", "dataResult"]>,
    DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
  ]
> {
  let arguments = (ins IntChannelType:$address, ChannelType:$data);
  let results = (outs IntChannelType:$addressResult, ChannelType:$dataResult);
  let assemblyFormat = [{
    `[` $address `]` $data attr-dict `:` type($address) `,` type($data)
  }];
}

class Handshake_LoadOp<string mnemonic> :
  Handshake_MemPortOp<mnemonic, LoadOpInterface> {
  let builders = [OpBuilder<(ins "MemRefType":$memrefType, "Value":$address)>];

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
}

class Handshake_StoreOp<string mnemonic> :
  Handshake_MemPortOp<mnemonic, StoreOpInterface> {
  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned int idx) {
      return (idx == 0) ? "addrIn" : "dataIn";
    }

    std::string $cppClass::getResultName(unsigned int idx) {
      return (idx == 0) ? "addrOut" : "dataToMem";
    }
  }];
}

def MCLoadOp : Handshake_LoadOp<"mc_load"> {
  let summary = "load operation for memory controller (MC)";
  let description = [{
    Represents a load memory port which sends load requests to a memory
    controller (`dynamatic::handshake::MemoryControllerOp`). It receives an
    address from a dataflow predecessor and a data value from the memory
    interface that eventually holds the loaded data. It returns an address that
    is sent to the memory interface as well as a data value that is sent to its
    dataflow successor.

    The order of operands is
    1. address (from predecessor)
    2. data value (from memory interface)

    The order of results is
    1. address (to memory interface).
    2. data value (to successor)

    Example:

    ```mlir
    %addrOut, %dataOut = mc_load [%addrIn] %dataFromMem : <i32>, <i32>
    ```
  }];
}

def LSQLoadOp : Handshake_LoadOp<"lsq_load"> {
  let summary = "load operation for load-store queue (LSQ)";
  let description = [{
    Represents a load memory port which sends load requests to an LSQ
    (`dynamatic::handshake::LSQOp`). It receives an address from a dataflow
    predecessor and a data value from the memory interface that eventually holds
    the loaded data. It returns an address that is sent to the memory interface
    as well as a data value that is sent to its dataflow successor.

    The order of operands is
    1. address (from predecessor)
    2. data value (from memory interface)

    The order of results is
    1. address (to memory interface).
    2. data value (to successor)

    Example:

    ```mlir
    %addrOut, %dataOut = lsq_load [%addrIn] %dataFromMem : <i32>, <i32>
    ```
  }];
}

def MCStoreOp : Handshake_StoreOp<"mc_store"> {
  let summary = "store operation for memory controller (MC)";
  let description = [{
    Represents a store memory port which sends store requests to a memory
    controller (`dynamatic::handshake::MemoryControllerOp`). It receives an
    address and a data value to store in memory from its dataflow predecessors.
    It returns the value to store and an address, both of which are sent to the
    memory interface.

    The order of operands is
    1. address (from predecessor)
    2. data value (from predecessor)

    The order of results is
    1. address (to memory interface).
    2. data value (to memory interface)

    Example:

    ```mlir
    %addrOut, %dataToMem = mc_load [%addrIn] %dataIn : <i32>, <i32>
    ```
  }];
}

def LSQStoreOp : Handshake_StoreOp<"lsq_store"> {
  let summary = "store operation for load-store queue (LSQ)";
  let description = [{
    Represents a store memory port which sends store requests to an LSQ
    (`dynamatic::handshake::LSQOp`). It receives an address and a data value to
    store in memory from its dataflow predecessors. It returns the value to
    store and an address, both of which are sent to the memory interface.

    The order of operands is
    1. address (from predecessor)
    2. data value (from predecessor)

    The order of results is
    1. address (to memory interface).
    2. data value (to memory interface)

    Example:

    ```mlir
    %addrOut, %dataToMem = lsq_load [%addrIn] %dataIn : <i32>, <i32>
    ```
  }];
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
  SpeculationOpInterface,
  AllTypesMatch<["dataIn", "dataOut"]>,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
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
    follows and might not reach a save-commit. $SCBranchCtrl is needed
    in the replicated branches to discard the latter signal.

    Example:

    ```mlir
    %dataOut, %saveCtrl, %commitCtrl, %SCSaveCtrl, %SCCommitCtrl, %SCBranchCtrl
      = speculator[enable] %dataIn : !handshake.channel<i1>
    ```
  }];

  let arguments = (ins HandshakeType:$dataIn, ControlType:$enable);
  let results = (outs HandshakeType:$dataOut,
                      BoolChannel:$saveCtrl, BoolChannel:$commitCtrl, 
                      IntSizedChannel<3>:$SCSaveCtrl,
                      IntSizedChannel<3>:$SCCommitCtrl, 
                      BoolChannel:$SCBranchCtrl);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

class Handshake_SpecOp<string mnemonic> : Handshake_Op<mnemonic, [
  SpeculationOpInterface, AllTypesMatch<["dataIn", "dataOut"]>
]> {
  let arguments = (ins HandshakeType:$dataIn, BoolChannel:$ctrl);
  let results = (outs HandshakeType:$dataOut);
  let assemblyFormat = [{
    `[` $ctrl `]` $dataIn attr-dict `:` type($dataIn) `,` type($ctrl)
  }];
}

def SpecSaveOp : Handshake_SpecOp<"spec_save"> {
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
    %dataOut = spec_save[%ctrl] %dataIn : !handshake.channel<i11>
    ```
  }];
}

def SpecCommitOp : Handshake_SpecOp<"spec_commit"> {
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
    %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i11>
    ```
  }];

}

def SpecSaveCommitOp : Handshake_SpecOp<"spec_save_commit"> {
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
  let arguments = (ins HandshakeType:$dataIn, IntSizedChannel<3>:$ctrl);
}

def SpeculatingBranchOp : Handshake_Op<"speculating_branch", [
  SpeculationOpInterface,
  AllTypesMatch<["dataOperand", "trueResult", "falseResult"]>
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
      %dataOperand : !handshake.channel<i1>, !handshake.channel<i32>
    ```
  }];

  let arguments = (ins HandshakeType:$tagFromOperand, HandshakeType:$dataOperand);
  let results = (outs HandshakeType:$trueResult, HandshakeType:$falseResult);
  let assemblyFormat = [{
    `[` $tagFromOperand `]` $dataOperand attr-dict `:` 
      type($tagFromOperand) `,` type($dataOperand)
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
    OpBuilder<(ins "::mlir::Value":$valid), [{
      mlir::Type validReadyType = $_builder.getIntegerType(1);
      assert(valid.getType() == validReadyType && "incorrect valid signal");

      $_state.addOperands(valid);
      $_state.addTypes({
        ::dynamatic::handshake::ControlType::get($_builder.getContext()),
        validReadyType
      });
    }]>,
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

  let arguments = (ins HandshakeType:$channelLike,
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

def ChannelReshapeTypeAttr : I32EnumAttr<
    "ChannelReshapeType", "",
    [
      I32EnumAttrCase<"MergeData", 0>,
      I32EnumAttrCase<"SplitData", 1>,
      I32EnumAttrCase<"MergeExtra", 2>,
      I32EnumAttrCase<"SplitExtra", 3>
    ]> {
  let cppNamespace = "::dynamatic::handshake";
}

def ReshapeOp : Handshake_Op<"reshape", [Pure]> {
  let summary = "Reshapes the individual signals of a `handshake::ChannelType`.";
  let description = [{
    Reshapes a `ChannelType`'d value to a different form without modifying the
    channel's actual payload, only rerouting bits to different extra signals or
    to the channel's data signal. There are currently four reshaping types that
    match two-by-two to perform reverse reshapings (each reshaping type pair has
    a dedicated operation builder).

    - `ChannelReshapeType::MergeData` merges all downstream extra signals into
      the data signal, which becomes an `i<X>` where X is the sum of the
      bitwidths of all downstream extra signals plus the data signal's bitwidth.
      If there are no downstream extra signals, the data type is unchanged. All
      upstream extra signals (if any) are merged into a single one named
      `MERGED_UP_NAME` with type `i<Y>`, where Y is the sum of the bitwidths
      of all upstream extra signals. `ChannelReshapeType::SplitData` performs
      the reverse transformation.
      
    - `ChannelReshapeType::MergeExtra` behaves identically to
      `ChannelReshapeType::MergeData` for extra upstream signals. However,
      all downstream extra signals (if any) are merged into a single one named
      `MERGED_DOWN_NAME` with type `i<Z>`, where Z is the sum of the bitwidths
      of all downstream extra signals. The data type always remain unchanged.
      `ChannelReshapeType::SplitExtra` performs the reverse transformation.  

    Example:

    ```mlir
    // Merging into data
    %reshaped = reshape [MergeData] %channel :
      <f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>
      -> <i42, [mergedUp: i8 (U)]>
    
    // -----
    
    // Merging into extra
    %reshaped = reshape [MergeExtra] %channel :
      <f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>
      -> <f32, [mergedDowm: i10, mergedUp: i8 (U)]>
    ```
  }];

  let arguments = (ins ChannelReshapeTypeAttr:$reshapeType, ChannelType:$channel);
  let results = (outs ChannelType:$reshaped);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins 
      "::mlir::TypedValue<::dynamatic::handshake::ChannelType>":$channel,
      "bool":$mergeDownstreamIntoData)>,
    OpBuilder<(ins
      "::mlir::TypedValue<::dynamatic::handshake::ChannelType>":$channel,
      "bool":$splitDownstreamFromData, "::mlir::Type":$reshapedType)>,
  ];

  let assemblyFormat = [{
    `[` $reshapeType `]` $channel attr-dict `:`
      type($channel) `->` type($reshaped)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
  
  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral
      MERGED_DOWN_NAME = ::llvm::StringLiteral("mergedDown"),
      MERGED_UP_NAME = ::llvm::StringLiteral("mergedUp");
  }];
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_OPS_TD
