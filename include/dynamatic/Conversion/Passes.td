//===-- Passes.td - Conversion passes definition ----------------*- C++ -*-===//
//
// This file contains definitions for all dialect conversion and export passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_PASSES_TD
#define DYNAMATIC_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AffineToScf
//===----------------------------------------------------------------------===//

def AffineToScf : Pass<"lower-affine-to-scf", "mlir::ModuleOp"> {
  let summary = "Lower affine operations to a mix of SCF and standard operations";
  let description = [{
    Very close analog to the AffineToStandard pass from MLIR that preserves the
    attributes of AffineLoadOp's and AffineStoreOp's during lowering. Keeping
    these attributes is required for our memory analysis pass to pass down
    dependence information between memory accesses to lower IR levels, in
    particular to the std-to-handshake lowering pass. 
  }];
  let constructor = "dynamatic::createAffineToScfPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// StandardToHandshakeFPGA18
//===----------------------------------------------------------------------===//

def StandardToHandshakeFPGA18 : Pass<"lower-std-to-handshake-fpga18", "mlir::ModuleOp"> {
  let summary = "Lower Standard MLIR into Handshake IR following elastic pass described in FPGA18";
  let constructor = "dynamatic::createStandardToHandshakeFPGA18Pass()";
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
  let options =
    [Option<"idBasicBlocks", "id-basic-blocks", "bool", "false",
            "If true, will attach a 'bb' attribute to each operation in the "
            "resulting IR to identify the basic block from which the "
            "operation originates.">];
}

//===----------------------------------------------------------------------===//
// StandardToHandshakeFPGA18
//===----------------------------------------------------------------------===//

def ExportCFG : Pass<"export-cfg", "mlir::ModuleOp"> {
  let summary = "Exports the CFG of a standard-level function";
  let description = [{
    Exports a CFG graph corresponding to the function present in the module. The
    pass only supports one function at the moment and will fail if more than one
    function exists in the module.
  }];
  let constructor = "dynamatic::createExportCFGPass()";
}


//===----------------------------------------------------------------------===//
// ExportDOT
//===----------------------------------------------------------------------===//

def ExportDOT : Pass<"export-dot", "mlir::ModuleOp"> {
  let summary = "Emit the handshake-level IR to a DOT file";
  let description = [{
    Exports a DOT graph corresponding to the module for visualization and
    legacy-compatibility purposes. The pass only supports exporting the graph of
    a single Handshake function at the moment, and will fail if there is more
    than one Handhsake function in the module.
  }];
  let constructor = "dynamatic::createExportDOTPass()";
  let options =
    [Option<"legacy", "legacy", "bool", "false",
            "If true, the exported DOT file will be made compatible with "
            "legacy Dynamatic (i.e. the file will be parsable by legacy "
            "Dynamatic's post-processing passes). Enabling this option adds "
            "some constraints on the input IR, which must also be compatible "
            "with legacy Dynamatic's elastic circuits. In general, it is "
            "therefore required to run the --prepare-for-legacy pass before "
            "exporting a legacy-compatible DOT.">,
    Option<"prettyPrint", "pretty-print", "bool", "true",
            "If true, the exported DOT file will be pretty-printed, making its "
            "visualization easier to look at but harder to debug with. For "
            "example, node names will be shortened (and not uniqued) when "
            "pretty-printing.">];
}

//===----------------------------------------------------------------------===//
// HandshakeToNetlist
//===----------------------------------------------------------------------===//

def HandshakeToNetlist : Pass<"lower-handshake-to-netlist", "mlir::ModuleOp"> {
  let summary = "Lower Handshake to ESI/HW (netlist)";
  let description = [{
    Lower Handshake IR into a "netlist-level" representation made up of HW
    instances of external modules interconnected by ESI channels.
  }];
  let constructor = "dynamatic::createHandshakeToNetlistPass()";
  let dependentDialects = ["circt::hw::HWDialect", "circt::esi::ESIDialect"];
}

#endif // DYNAMATIC_CONVERSION_PASSES_TD
