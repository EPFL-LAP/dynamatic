from generators.handshake.join import generate_join
from generators.handshake.dataless.dataless_oehb import generate_dataless_oehb
from generators.support.delay_buffer import generate_delay_buffer

from generators.handshake.binop import generate_binop

def generate_muli(name, params):
    bitwidth = params["bitwidth"]
    latency = params["latency"]

    assert(latency == 4)

    mul_4_stage_name = name + "_mul_4_stage"
    dependencies = f"""
// Module of mul_4_stage
module {mul_4_stage_name} #(
  parameter DATA_TYPE = {bitwidth}
)(
  // inputs
  input  clk,
  input  ce,
  input  [DATA_TYPE - 1 : 0] a,
  input  [DATA_TYPE - 1 : 0] b,
  // outputs
  output [DATA_TYPE - 1 : 0] p
);

  reg  [DATA_TYPE - 1 : 0] a_reg = 0;
  reg  [DATA_TYPE - 1 : 0] b_reg = 0;
  reg  [DATA_TYPE - 1 : 0] q0 = 0;
  reg  [DATA_TYPE - 1 : 0] q1 = 0;
  reg  [DATA_TYPE - 1 : 0] q2 = 0;
  wire  [DATA_TYPE - 1 : 0] mul;

  assign mul = a_reg * b_reg;

  always @(posedge clk) begin
    if (ce) begin
      a_reg <= a;
      b_reg <= b;
      q0 <= mul;
      q1 <= q0;
      q2 <= q1;
    end
  end

  assign p = q2;

endmodule
"""
    muli_body = f"""
  {mul_4_stage_name} #(
    .DATA_TYPE({bitwidth})
  ) mul_4_stage_inst (
    .clk(clk),
    .ce(oehb_ready),
    .a(lhs),
    .b(rhs),
    .p(result)
  );
"""

    return generate_binop(
      name=name,
      op_body=muli_body,
      handshake_op="muli",
      latency=latency,
      bitwidth=bitwidth,
      dependencies=dependencies,
      extra_signals=params.get("extra_signals", None))