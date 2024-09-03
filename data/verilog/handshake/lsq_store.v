`timescale 1ns/1ps
module lsq_store #(
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
  // Data assignment
  assign dataToMem = dataIn;
  assign dataToMem_valid = dataIn_valid;
  assign dataIn_ready = dataToMem_ready;

  // Address assignment
  assign addrOut = addrIn;
  assign addrOut_valid = addrIn_valid;
  assign addrIn_ready = addrOut_ready;

endmodule
