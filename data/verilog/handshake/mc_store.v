`timescale 1ns/1ps
module mc_store #(
  parameter DATA_TYPE = 32,
  parameter ADDR_TYPE = 32
)(
  input  clk,
  input  rst,
  // Data from Circuit Channel
  input  [DATA_TYPE - 1 : 0] dataIn,
  input  dataIn_valid,
  output dataIn_ready,
  // Address from Circuit Channel
  input  [ADDR_TYPE - 1 : 0] addrIn,
  input  addrIn_valid,
  output addrIn_ready,
  // Data to Interface Channel
  output [DATA_TYPE - 1 : 0] dataToMem,
  output dataToMem_valid,
  input  dataToMem_ready,
  // Address to Interface Channel
  output [ADDR_TYPE - 1 : 0] addrOut,
  output addrOut_valid,
  input  addrOut_ready 
);

  wire join_valid;

  // Instantiate join
  join_type #(
    .SIZE(2)
  ) join_inst (
    .ins_valid  ({addrIn_valid, dataIn_valid}),
    .outs_ready (dataToMem_ready             ),
    .ins_ready  ({addrIn_ready, dataIn_ready}),
    .outs_valid  (join_valid                  )
  );

  // Address assignment
  assign addrOut = addrIn;
  assign addrOut_valid = join_valid;
  // Data assignment
  assign dataToMem = dataIn;
  assign dataToMem_valid = join_valid;

endmodule
