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

  reg tmp_valid_out;
  reg [INPUTS - 1 : 0] tmp_ready_out;
  reg [DATA_TYPE - 1 : 0] tmp_data_out;
	integer i;

	always @(*) begin
		tmp_valid_out = 0;
    tmp_ready_out = {INPUTS{1'b0}}; 
    tmp_data_out = data_in_bus[0 * DATA_TYPE +: DATA_TYPE];

		for (i = 0; i < INPUTS; i = i + 1) begin
			if (ins_valid[i]) begin
        tmp_data_out = ins[i * DATA_IN_SIZE +: DATA_IN_SIZE];
				tmp_valid_out = 1;
        tmp_ready_out[i] = outs_ready;
			end
		end
	end
  
  assign  outs        = tmp_data_out;
	assign	outs_valid  = tmp_valid_out;
	assign  ins_ready   = tmp_ready_out;

endmodule
