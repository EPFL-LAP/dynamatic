def generate_dataless_oehb(name, params):
  verilog_oehb = f"""
  `timescale 1ns/1ps

//oehb dataless Module
module {name} (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  // Define internal signals
  reg outputValid = 0;

  always @(posedge clk) begin
    if (rst) begin
      outputValid <= 0;
    end else begin
      outputValid <= ins_valid | (~outs_ready & outputValid);
    end
  end

  assign ins_ready = ~outputValid | outs_ready;
  assign outs_valid = outputValid;
  
endmodule
"""

  return verilog_oehb