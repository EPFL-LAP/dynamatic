module mem_controller_loadless #(
  parameter CTRL_COUNT = 1,
  parameter STORE_COUNT = 1,
  parameter DATA_WIDTH = 32,
  parameter ADDR_WIDTH = 32
)(
  input clk,
  input rst,
  
);