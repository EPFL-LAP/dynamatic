`timescale 1ns/1ps

module ram_1w1r #(
  parameter DATA_WIDTH = 32,
  parameter ADDR_WIDTH = 10,
  parameter SIZE = 1024) (
  clk,
  rst,
  loadEn,
  loadAddr,
  storeEn,
  storeAddr,
  storeData,
  loadData
);

input clk;
input rst;
input loadEn;
input [ADDR_WIDTH - 1 : 0] loadAddr;
input storeEn;
input [ADDR_WIDTH - 1 : 0] storeAddr;
input [DATA_WIDTH - 1 : 0] storeData;
output [DATA_WIDTH - 1 : 0] loadData;

reg [DATA_WIDTH - 1 : 0] load_data_reg;
reg [DATA_WIDTH - 1 : 0] ram [SIZE - 1 : 0];

always@(posedge clk) begin
  if (loadEn) begin
    load_data_reg <= ram[loadAddr];
  end
end 

always@(posedge clk) begin
  if (storeEn) begin
    ram[storeAddr] <= storeData;
  end
end 

assign loadData = load_data_reg;

endmodule
