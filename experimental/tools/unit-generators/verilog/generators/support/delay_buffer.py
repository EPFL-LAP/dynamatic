
def generate_delay_buffer(name, params):
    size = params["size"]

    delay_buffer = f"""
`timescale 1ns/1ps

// Module of delay_buffer
module {name}(
  input  clk,
  input  rst,
  input  valid_in,
  input  ready_in,
  output valid_out
);
  integer i;
  reg [{size} - 1 : 0] regs = 0;

  always @(posedge clk) begin
    if (rst)
      regs[0] <= 0;
    else if (ready_in)
      regs[0] <= valid_in;
  end

  always @(posedge clk) begin
    if (rst) begin
      for (i = 1; i < {size}; i = i + 1) begin
        regs[i] <= 0;
      end
    end else if (ready_in) begin
      for (i = 1; i < {size}; i = i + 1) begin
        regs[i] <= regs[i - 1];
      end
    end
  end

  assign valid_out = regs[{size} - 1];
endmodule
"""
    return delay_buffer
