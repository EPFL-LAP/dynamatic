`timescale 1ns/1ps
module tehb_dataless (
	input  clk,
	input  rst,
  // Input Channel
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output outs_valid,	
	input  outs_ready
);
	reg fullReg = 0;
	
	always @(posedge clk) begin
		if (rst) begin
			fullReg <= 0;
		end else begin
			fullReg <= (ins_valid | fullReg) & ~outs_ready;
		end
	end

	assign ins_ready = ~fullReg;
	assign outs_valid = ins_valid | fullReg;

endmodule
