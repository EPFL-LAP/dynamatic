//===- HandshakeAttributes.td - Attributes for Handshake ---*- tablegen -*-===//
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
// This file defines Handshake dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ATTRIBUTES_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ATTRIBUTES_TD

include "dynamatic/Dialect/Handshake/Handshake.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Attribute classes 
//===----------------------------------------------------------------------===//

/// Base class for all Handshake attributes.
class Handshake_Attr< string name, string attrMnemonic, list<Trait> traits = [],
                      string baseCppClass = "::mlir::Attribute"> 
  : AttrDef<Handshake_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

/// Helper base class for attributes that just act as containers around an
/// attribute type semantically attached to operands.
///
/// Implementors only need to define the `extraClassDeclaration` member with a
/// type alias named `OperandAttr` that points to the operand attribute the
/// container attribute "wraps". For example:
/// 
/// let extraClassDeclaration = [{
///   using OperandAttr = ::dynamatic::handshake::ChannelBufPropsAttr;
/// }];
class OperandContainerAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : Handshake_Attr<name, attrMnemonic, traits> {

  /// A dictionnary from operand index (encoded as string) to an operand
  /// attribute of a specific type
  let parameters = (ins "mlir::DictionaryAttr":$operandAttributes);

  let builders = [
    AttrBuilder<(ins 
      "mlir::SmallVector<mlir::NamedAttribute>&":$operandAttributes), [{
      return $_get(context, 
        mlir::DictionaryAttr::get(context, operandAttributes));   
    }]>
  ];

  /// Automatically implemented.
  let genVerifyDecl = 1;
  /// Automatically implemented.
  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    /// Attempts to convert the string attribute to an unsigned number. Returns it
    /// on success, or std::nullopt on failure.
    static inline std::optional<size_t> toIdx(std::string str) {
      if (std::any_of(str.begin(), str.end(),
                      [](char c) { return !std::isdigit(c); }))
        return {};
      return stoi(str);
    }

    /// Automatically implemented print method.
    void $cppClass::print(mlir::AsmPrinter &odsPrinter) const {
      mlir::DictionaryAttr operandAttrs = getOperandAttributes();
      if (operandAttrs.empty()) {
        odsPrinter << "{}";
        return;
      }

      odsPrinter << "{";
      size_t numProps = operandAttrs.size();
      for (auto [idx, attr] : llvm::enumerate(operandAttrs)) {
        odsPrinter << attr.getName() << ": ";
        attr.getValue().cast<$cppClass::OperandAttr>().print(odsPrinter);
        if (idx != numProps - 1)
          odsPrinter << ", ";
      }
      odsPrinter << "}";
    }

    /// Automatically implemented parse function.
    mlir::Attribute $cppClass::parse(mlir::AsmParser &odsParser, mlir::Type odsType) {
      mlir::SmallVector<mlir::NamedAttribute> operandAttrs;
      mlir::MLIRContext *ctx = odsParser.getContext();

      auto parseChannelProps = [&]() -> mlir::ParseResult {
        // Parse the channel index
        std::string channelStr;
        if (odsParser.parseString(&channelStr) || !toIdx(channelStr).has_value() ||
            odsParser.parseColon())
          return failure();

        // Parse the operand attribute
        mlir::Attribute channel = $cppClass::OperandAttr::parse(odsParser, odsType);
        if (!channel)
          return failure();

        // Accumulate in the list
        operandAttrs.push_back(
            mlir::NamedAttribute(StringAttr::get(ctx, channelStr), channel));
        return success();
      };

      // The attribute is printed as a comma-separated list of named channel
      // properties surrounded by brackets
      if (odsParser.parseLBrace() ||
          odsParser.parseCommaSeparatedList(parseChannelProps) ||
          odsParser.parseRBrace())
        return nullptr;

      return $cppClass::get(ctx, operandAttrs);
    }

    /// Automatically implemented verify method.
    LogicalResult
    $cppClass::verify(function_ref<InFlightDiagnostic()> emitError,
                      mlir::DictionaryAttr operandAttrs) {
      for (const mlir::NamedAttribute &attr : operandAttrs) {
        // Name must represent a number
        std::optional<size_t> idx = toIdx(attr.getName().str());
        if (!idx.has_value())
          return emitError() << "map keys must represent indices, but got "
                            << attr.getName();

        // Value must be channel buffering properties
        if (!attr.getValue().isa<$cppClass::OperandAttr>())
          return emitError() << "map values are not of the correct type";
      }
      return success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Memory annotations
//===----------------------------------------------------------------------===//

def MemInterfaceAttr : Handshake_Attr<"MemInterface", "mem_interface"> {
  let summary = [{
    Conveys information about the memory interface a memory port should
    eventually connect to.
  }];
  let description = [{
    When attached to an operation representing some kind of memory access, this
    instructs the cf-to-handshake conversion pass to connect the memory port to
    a MC (no LSQ group provided) or to a particular LSQ group (LSQ group
    provided). Usually, this would be set by a memory analysis pass that would
    have determined dependency relations between this access and  others on the
    same memory region.
  }];
  let parameters = (ins "::std::optional<unsigned>":$lsqGroup);
  let builders = [AttrBuilder<(ins), "return $_get(context, std::nullopt);">];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Determines whether the memory port the attribite is attached to should
    /// connect to a load-store queue (LSQ).
    inline bool connectsToLSQ() {
      return getLsqGroup().has_value();
    }

    /// Determines whether the memory port the attribite is attached to should
    /// connect to a memory controller (MC).
    inline bool connectsToMC() {
      return !connectsToLSQ();
    }
  }];
}

def MemDependenceAttr : Handshake_Attr<"MemDependence", "dep"> {
  let summary = "A memory dependence with respect to another memory access";
  let description = [{
    Represents a memory dependence between a source operation (the one this
    attribute is attached to) and a destination operation referenced through
    the `dstAccess` parameter. For example, if the source operation is a
    memref::StoreOp and the destination operation is a memref::LoadOp, this
    represents a RAW memory dependency. 
    
    The dependency is furthermore characterized by the loop depth at which the
    dependency is (`loopDepth`).
  }];
  
  let parameters = (ins 
    "::mlir::StringAttr":$dstAccess,
    "unsigned":$loopDepth
  );

  let assemblyFormat = "`[` $dstAccess `,` $loopDepth `]`";
}

def MemDependenceArrayAttr : Handshake_Attr<"MemDependenceArray", "deps"> {
  let summary = "A list of memory dependencies with respect to other memory operations";
  let description = [{
    Represents a list of memory dependencies between the operation this
    attribute is attached to and other named memory accesses in the same
    function.
  }];

  let parameters = (ins ArrayRefParameter<"::dynamatic::handshake::MemDependenceAttr">:$dependencies);
  let assemblyFormat = "`[` $dependencies `]`";
}

//===----------------------------------------------------------------------===//
// Buffer annotations
//===----------------------------------------------------------------------===//


def Handshake_BufferTypeAttr : I64EnumAttr<
    "BufferType", "",
    [
      /// ONE_SLOT_BREAK_DV: This buffer breaks the D and V signal paths.
      /// Previously known as a slot of OEHB (Opaque Elastic Half-Buffer),
      /// it introduces one cycle of latency on the D and V paths.
      /// It does not break the R signal path and adds no latency on R.
      I64EnumAttrCase<"ONE_SLOT_BREAK_DV", 0>,
          
      /// ONE_SLOT_BREAK_R: This buffer breaks the R signal path.
      /// Previously known as a slot of TEHB (Transparent Elastic Half-Buffer),
      /// it introduces one cycle of latency on the R path.
      /// It does not break the D and V signal path and adds no latency on D and V.
      I64EnumAttrCase<"ONE_SLOT_BREAK_R", 1>,

      /// FIFO_BREAK_NONE: Previously known as a 'tfifo' (Transparent FIFO),
      /// this is a FIFO_BREAK_DV with a bypass, adding no latency to any signal paths.
      /// Its only purpose is to hold tokens.
      I64EnumAttrCase<"FIFO_BREAK_NONE", 2>,

      /// FIFO_BREAK_DV: This buffer breaks the D and V paths.
      /// It has multiple slots but, unlike a chain of ONE_SLOT_BREAK_DV buffers,
      /// its structure cannot be split. It introduces one cycle of latency on the
      /// D and V paths regardless of the number of slots and has no latency on R.
      /// It was previously called an 'elastic_fifo_inner'.
      I64EnumAttrCase<"FIFO_BREAK_DV", 3>,

      /// ONE_SLOT_BREAK_DVR: Each slot of this buffer breaks the D, V, and R signal
      /// paths and introduces one cycle of latency on all of them.
      I64EnumAttrCase<"ONE_SLOT_BREAK_DVR", 4>,

      /// SHIFT_REG_BREAK_DV: Breaks D and V paths. Multiple slots share the same
      /// handshake control signals, causing them to accept or stall inputs
      /// simultaneously.
      I64EnumAttrCase<"SHIFT_REG_BREAK_DV", 5>
    ]> {
  let cppNamespace = "::dynamatic::handshake";
}

def ChannelBufPropsAttr : Handshake_Attr<"ChannelBufProps", "channelBufProps"> {
  let summary = "Buffering properties of a single channel";
  let description = [{
    Holds a set of buffering properties for a channel (i.e, an SSA value used
    only once) that are meant to be read by the buffer placement pass and
    constrain it in what kind of buffers it can place on the channel these
    properties are attached to. These properties are (1) the minimum and maximum
    number of transparent buffer slots placable on the channel, (2) the minimum
    and maximum number of opaque buffer slots placable on the channel, and (3)
    the channel's combinational delay expressed as a pair of pre-buffer
    (`inDelay`) and post-buffer (`outDelay`) delays as well as the total channel
    delay (`delay`) when it is unbuffered, all in ns.

    Users of this attribute may wish to interact with it through the
    dynamatic::handshake::ChannelBufProps struct, which represents the same information and
    is easier to manipulate than the raw MLIR attribute.
  }];
  
  let parameters = (ins "unsigned":$minTrans,
                        "std::optional<unsigned>":$maxTrans,
                        "unsigned":$minOpaque,
                        "std::optional<unsigned>":$maxOpaque,
                        "unsigned":$minSlots,
                        "mlir::FloatAttr":$inDelay,
                        "mlir::FloatAttr":$outDelay,
                        "mlir::FloatAttr":$delay
                        );

  let builders = [
    AttrBuilder<(ins "dynamatic::handshake::ChannelBufProps&":$props), [{
      return $_get(context, props.minTrans, props.maxTrans, props.minOpaque, 
        props.maxOpaque, props.minSlots,
        FloatAttr::get(mlir::Float32Type::get(context), props.inDelay), 
        FloatAttr::get(mlir::Float32Type::get(context), props.outDelay),
        FloatAttr::get(mlir::Float32Type::get(context), props.delay));
    }]>
  ];

  let hasCustomAssemblyFormat = 1;  

  let extraClassDeclaration = [{
    /// Container attribute corresponding to this operand attribute
    using ContainerAttr = ::dynamatic::handshake::ChannelBufPropsContainerAttr;

    /// Gets a ChannelBufProps instance that holds the same information as this
    /// attribute for easier data manipulation.
    inline dynamatic::handshake::ChannelBufProps getProps() {
      return dynamatic::handshake::ChannelBufProps(getMinTrans(), getMaxTrans(),
        getMinOpaque(), getMaxOpaque(), getMinSlots(), getInDelay().getValueAsDouble(),
        getOutDelay().getValueAsDouble(), getDelay().getValueAsDouble());
    }
  }];
}

def ChannelBufPropsContainerAttr : OperandContainerAttr<
  "ChannelBufPropsContainer", "bufProps"
> {
  let summary = "Container around the `ChannelBufPropsAttr` operand attribute.";
  let description = [{
    Maps operands of an operation to their `ChannelBufPropsAttr` attribute, if
    it exists. Never really needs to be interacted with by user code beside the 
    `dynamatic/Support/OperandAttribute.h` header.  
  }];

  let extraClassDeclaration = [{
    /// Operand attribute corresponding to this container attribute
    using OperandAttr = ::dynamatic::handshake::ChannelBufPropsAttr;
  }];
}

//===----------------------------------------------------------------------===//
// CFDFC Extraction Results
//===----------------------------------------------------------------------===//
def CFDFCToBBListAttr : Handshake_Attr<
  "CFDFCToBBList", "cfdfcToBBList"
> {
  let summary = "Maps CFDFC index to the list of BBs";
  let description = [{
    Holds a map from CFDFC index to the corresponding list of BB indices. For example
    `{"0" : [1, 2], "1" : [3, 4, 5]}`, means CFDFC 0 contains BB 1 and BB 2 and CFDFC 1 contains BB 3,
    BB 4 and BB 5.

    For constructing this attribute, user can directly provide 
    an llvm::MapVector<int, std::vector<int>>.
  }];

  let parameters = (ins "mlir::DictionaryAttr":$cfdfcMap);

  let assemblyFormat = " $cfdfcMap ";

  let builders = [
    // Builder for llvm::MapVector<size_t, std::vector<unsigned>>
    AttrBuilder<(ins "llvm::MapVector<size_t, std::vector<unsigned>>":$mapVector), [{
      llvm::SmallVector<mlir::NamedAttribute, 10> attrs;
      mlir::Builder builder(context);
      for (const auto &pair: mapVector) {
        llvm::SmallVector<mlir::Attribute, 10> bbList;
        for (int val : pair.second) {
          bbList.push_back(builder.getUI32IntegerAttr(val));
        }
        auto bbListAttr = builder.getArrayAttr(bbList);
        attrs.push_back({builder.getStringAttr(std::to_string(pair.first)), bbListAttr});
      }
      return $_get(context, mlir::DictionaryAttr::get(context, attrs));
    }]>
  ];
}

def CFDFCThroughputAttr : Handshake_Attr<
  "CFDFCThroughput", "cfdfcThroughput"
> {
  let summary = "Maps CFDFC index to the throughput value";
  let description = [{
    Holds a map from CFDFC index to the corresponding throughput value (floating value). For example,
    {"0" : 0.5, "1": 0.25}. This means CFDFC 0 has a throughput of 0.5 and CFDFC 1 has 
    a throughput of 0.25.

    For constructing this attribute, user can directly provide an
    llvm::MapVector<int, double>
  }];

  let parameters = (ins "mlir::DictionaryAttr":$throughputMap);

  let assemblyFormat = " $throughputMap ";

  let builders = [
    // Builder for llvm::MapVector<size_t, double>
    AttrBuilder<(ins "llvm::MapVector<size_t, double>":$cfdfcThroughputMap), [{
      llvm::SmallVector<mlir::NamedAttribute, 10> attrs;
      mlir::Builder builder(context);
      for (const auto &pair: cfdfcThroughputMap) {
        mlir::Type doubleType = mlir::FloatType::getF64(context);
        auto tmpCFDFCThroughputAttr = builder.getFloatAttr(doubleType, pair.second);
        attrs.push_back({builder.getStringAttr(std::to_string(pair.first)), tmpCFDFCThroughputAttr});
      }
      return $_get(context, mlir::DictionaryAttr::get(context, attrs));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// LSQ Sizing Results
//===----------------------------------------------------------------------===//
def LSQDepthAttr : Handshake_Attr<"LSQDepth", "lsqDepth"> {
  let summary = "Load-Queue and Store-Queue depths";
  let description = [{
    Holds the Load-Queue and Store-Queue depth for a LSQ.
  }];

  let parameters = (ins "unsigned":$loadQueueDepth, "unsigned":$storeQueueDepth);
  let assemblyFormat = "`[` $loadQueueDepth `,` $storeQueueDepth `]`";
}


//===----------------------------------------------------------------------===//
// FPU Implementation
//===----------------------------------------------------------------------===//

// Enum Tablegen declaration structure taken from the test enum in TestOps.td
// in https://reviews.llvm.org/D115181, which is the PR which introduced
// the version of enums we use in Dynamatic
//
// this weird structure has been removed in up-to-date MLIR
//
// I64EnumAttrs can be used directly,
// but they serialize as integers when in the attribute dictionary
// which makes the IR difficult for humans to interpret.

// Declare the underlying C++ enum and accompanying utility functions
// by declaring an I64EnumAttr with "let genSpecializedAttr = 0;"
// which tells tablegen not to actually generate the attribute
def FPUImplEnum : I64EnumAttr<"FPUImpl", "", [
  // C++ value is FPUImpl::FLOPOCO, string version is "flopoco"
  I64EnumAttrCase<"FLOPOCO", 0, "flopoco">,
  // C++ value is FPUImpl::VIVADO, string version is "vivado"
  I64EnumAttrCase<"VIVADO", 1, "vivado">
]> {
  let cppNamespace = "::dynamatic::handshake";
  let genSpecializedAttr = 0;
}
// and then we pass that "fake" attribute to this, which is the real attribute
def FPUImplEnumAttr : EnumAttr<Handshake_Dialect, FPUImplEnum, "fpu_impl">;

//===----------------------------------------------------------------------===//
// MemInitialValueAttr
//===----------------------------------------------------------------------===//

// Used to represent the constant/initial value of the ROM/RAM
def MemoryInitialValueAttr : Handshake_Attr<"MemoryInitialValue", "init_value"> {
  let parameters = (ins "DenseElementsAttr":$constant);
  let assemblyFormat = "`[` $constant `]`";
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ATTRIBUTES_TD
