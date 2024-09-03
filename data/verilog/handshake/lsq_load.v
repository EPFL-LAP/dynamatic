`timescale 1ns/1ps
module lsq_load #(
  parameter DATA_TYPE = 32,
  parameter ADDR_TYPE = 32
)(
  input  clk,
  input  rst,
  // Address from Circuit Channel
  input  [ADDR_TYPE - 1 : 0] addrIn,
  input  addrIn_valid,
  output addrIn_ready,
  // Address to Interface Channel
  output [ADDR_TYPE - 1 : 0] addrOut,
  output addrOut_valid,
  input  addrOut_ready,
  // Data from Interface Channel
  input  [DATA_TYPE - 1 : 0] dataFromMem,
  input  dataFromMem_valid,
  output dataFromMem_ready,
  // Data from Memory Channel
  output [DATA_TYPE - 1 : 0] dataOut,
  output dataOut_valid,
  input  dataOut_ready
);
  // Data assignment
  assign dataOut = dataFromMem;
  assign dataOut_valid = dataFromMem_valid;
  assign dataFromMem_ready = dataOut_ready;

  // Addr assignment
  assign addrOut = addrIn;
  assign addrOut_valid = addrIn_valid;
  assign addrIn_ready = addrOut_ready;

endmodule
