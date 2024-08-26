`timescale 1ns/1ps
module merge_notehb #(
  parameter INPUTS = 2,
  parameter DATA_TYPE = 32
) (
  input  clk,
  input  rst,
  input  [INPUTS * DATA_TYPE - 1 : 0] ins,   
  input  [INPUTS - 1 : 0] ins_valid,           
  output [INPUTS - 1 : 0] ins_ready,           
  output [DATA_TYPE - 1 : 0] outs,              
  output outs_valid,                      
  input  outs_ready                     
);
  reg [DATA_TYPE - 1 : 0] tmp_data_out = 0;
  reg tmp_valid_out = 0;

  integer i;

  always @(*) begin
		tmp_valid_out = 0;
		tmp_data_out = data_in_bus[0 * DATA_TYPE +: DATA_TYPE];
		for(i = INPUTS - 1; i >= 0; i = i - 1) begin
			if(ins_valid[i])begin
				tmp_data_out = ins[i * DATA_IN_SIZE +: DATA_IN_SIZE];
				tmp_valid_out = 1;
			end
		end
	end

  wire [DATA_TYPE - 1 : 0] tehb_data_in;
  wire tehb_pvalid;
  wire tehb_ready;

  assign tehb_data_in = tmp_data_out;
  assign tehb_pvalid = tmp_valid_out;

  assign tehb_ready = outs_ready;
  assign ins_ready = {INPUTS{tehb_ready}};
  assign outs_valid = tehb_pvalid;
  assign outs = tehb_data_in;

endmodule
