//===-- Passes.td - Conversion passes definition ----------------*- C++ -*-===//
//
// This file contains definitions for all dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_PASSES_TD
#define DYNAMATIC_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

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
// ExportDOT
//===----------------------------------------------------------------------===//

def ExportDOT : Pass<"export-dot", "mlir::ModuleOp"> {
  let summary = "Emit the handshake-level IR to a DOT file";
  let description = [{
    This pass analyses a handshake::FuncOp operation and exports a .dot graph
    that corresponds to it. If multiple functions are present in the IR, the top
    level function will be printed, and called functions will be subgraphs
    within themain graph.
  }];
  let constructor = "dynamatic::createExportDOTPass()";
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
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
