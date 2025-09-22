def generate_dataless_lazy_fork(name, params):
    size = params["size"]

    verilog_header = "`timescale 1ns/1ps\n"

    verilog_datalessFork = f"""
// Module of dataless_lazy_fork
`timescale 1ns/1ps
module {name} #(
  parameter SIZE = {size}
)(
  input  clk,
	input  rst,
  // Input Channel
  input  ins_valid,
  output ins_ready,
  // Output Channels
  output reg [SIZE - 1 : 0] outs_valid,
	input  [SIZE - 1: 0] outs_ready
);
  wire allnReady;
  assign allnReady = &outs_ready;

  // Process to handle output valid signals based on input valid and output readiness
  integer i, j;
  reg [SIZE - 1 : 0] tmp_ready;

  always @(*) begin
    tmp_ready = {{SIZE{{1'b1}}}};

    for (i = 0; i < SIZE; i = i + 1) begin
      for (j = 0; j < SIZE; j = j + 1) begin
        if (i != j) begin
          tmp_ready[i] = tmp_ready[i] & outs_ready[j];
        end
      end
    end

    for (i = 0; i < SIZE; i = i + 1) begin
      outs_valid[i] = ins_valid & tmp_ready[i];
    end
  end

  assign ins_ready = allnReady;

endmodule

"""


    return verilog_header + verilog_datalessFork
