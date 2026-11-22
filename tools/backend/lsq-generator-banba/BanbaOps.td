#ifndef BANBA_OPS_TD
#define BANBA_OPS_TD

// Shorter names for the types of the operations' inputs and outputs.
// These only check the kind of type (an integer, an array);
// the exact widths and sizes are checked against the config
def Int : TypeAlias<AnySignlessInteger>;
def ArrayInt : Type<CPred<"isa<circt::hw::ArrayType>($_self)">,
    "hw.array">;
def ArrayI1 : Type<
    CPred<"isa<circt::hw::ArrayType>($_self) && "
          "cast<circt::hw::ArrayType>($_self).getElementType().isInteger(1)">,
    "hw.array of i1">;
def ClockType : Type<CPred<"isa<circt::seq::ClockType>($_self)">,
    "seq.clock">;

// Checks the types of an operation's inputs and outputs
// against the ones it states through ExpectedTypes.
// The check itself is written in C++, in BanbaOps.h
def VerifyTypesByConfig : NativeOpTrait<"VerifyTypesByConfig"> {
  let cppNamespace = "::banba::OpTrait";
}

// Tells MLIR that the order of the operations inside a region does not matter:
// as with wires in hardware,
// a value may be used above the operation that produces it
def GraphRegion : TraitList<[
    NativeOpTrait<"HasOnlyGraphRegion">,
    RegionKindInterface]>;

// The top-level load-store queue.
// Its inputs and outputs are the LSQ's ports to the circuit
// and its channels to the memory
def LsqOp : Op<Banba_Dialect, "lsq",
    [GraphRegion, DeclareOpInterfaceMethods<ExpectedTypes>,
     VerifyTypesByConfig]> {
  let arguments = (ins
      ClockType:$clk,
      I1:$rst,
      // From the circuit.
      // Each array holds one value per group, per load port or per store port
      ArrayI1:$group_init_valid,
      ArrayInt:$ld_port_addr,
      ArrayI1:$ld_port_addr_valid,
      ArrayI1:$ld_port_data_ready,
      ArrayInt:$st_port_addr,
      ArrayI1:$st_port_addr_valid,
      ArrayInt:$st_port_data,
      ArrayI1:$st_port_data_valid,
      // From the memory
      I1:$rd_req_ready,
      I1:$rd_resp_valid,
      Int:$rd_resp_id,
      Int:$rd_resp_data,
      I1:$wr_req_ready,
      I1:$wr_resp_valid,
      Int:$wr_resp_id);
  let results = (outs
      // To the circuit.
      // Each array holds one value per group, per load port or per store port
      ArrayI1:$group_init_ready,
      ArrayI1:$ld_port_addr_ready,
      ArrayInt:$ld_port_data,
      ArrayI1:$ld_port_data_valid,
      ArrayI1:$st_port_addr_ready,
      ArrayI1:$st_port_data_ready,
      // To the memory
      I1:$rd_req_valid,
      Int:$rd_req_id,
      Int:$rd_req_addr,
      I1:$rd_resp_ready,
      I1:$wr_req_valid,
      Int:$wr_req_id,
      Int:$wr_req_addr,
      Int:$wr_req_data,
      I1:$wr_resp_ready,
      I1:$empty);

  // The inside of the LSQ, holding the components it is made of
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat =
      "`(` operands `)` attr-dict `:` functional-type(operands, results) $body";

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      unsigned numGroups = config.getNumBBs();
      unsigned numLoadPorts = config.getNumLoadPorts();
      unsigned numStorePorts = config.getNumStorePorts();
      unsigned addrWidth = config.getAddrWidth();
      unsigned dataWidth = config.getDataWidth();
      unsigned indexWidth = config.getIndexWidth();
      return {t.Clock(),                          // clk
              t.Int(1),                           // rst
              t.Array(numGroups, 1),              // group_init_valid
              t.Array(numLoadPorts, addrWidth),   // ld_port_addr
              t.Array(numLoadPorts, 1),           // ld_port_addr_valid
              t.Array(numLoadPorts, 1),           // ld_port_data_ready
              t.Array(numStorePorts, addrWidth),  // st_port_addr
              t.Array(numStorePorts, 1),          // st_port_addr_valid
              t.Array(numStorePorts, dataWidth),  // st_port_data
              t.Array(numStorePorts, 1),          // st_port_data_valid
              t.Int(1),                           // rd_req_ready
              t.Int(1),                           // rd_resp_valid
              t.Int(indexWidth),                  // rd_resp_id
              t.Int(dataWidth),                   // rd_resp_data
              t.Int(1),                           // wr_req_ready
              t.Int(1),                           // wr_resp_valid
              t.Int(indexWidth)};                 // wr_resp_id
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      unsigned numGroups = config.getNumBBs();
      unsigned numLoadPorts = config.getNumLoadPorts();
      unsigned numStorePorts = config.getNumStorePorts();
      unsigned addrWidth = config.getAddrWidth();
      unsigned dataWidth = config.getDataWidth();
      unsigned indexWidth = config.getIndexWidth();
      return {t.Array(numGroups, 1),              // group_init_ready
              t.Array(numLoadPorts, 1),           // ld_port_addr_ready
              t.Array(numLoadPorts, dataWidth),   // ld_port_data
              t.Array(numLoadPorts, 1),           // ld_port_data_valid
              t.Array(numStorePorts, 1),          // st_port_addr_ready
              t.Array(numStorePorts, 1),          // st_port_data_ready
              t.Int(1),                           // rd_req_valid
              t.Int(indexWidth),                  // rd_req_id
              t.Int(addrWidth),                   // rd_req_addr
              t.Int(1),                           // rd_resp_ready
              t.Int(1),                           // wr_req_valid
              t.Int(indexWidth),                  // wr_req_id
              t.Int(addrWidth),                   // wr_req_addr
              t.Int(dataWidth),                   // wr_req_data
              t.Int(1),                           // wr_resp_ready
              t.Int(1)};                          // empty
    }
  }];
}

// Ends the inside of a component,
// and carries the values of the component's outputs
def OutputOp : Op<Banba_Dialect, "output", [Terminator, ReturnLike, Pure]> {
  let summary = "Carries a component's output port values";
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = "($outputs^ `:` type($outputs))? attr-dict";
}

#endif // BANBA_OPS_TD
