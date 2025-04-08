`timescale 1ns/1ps
module mul_4_stage #(
  parameter DATA_TYPE = 32
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


module muli #(
  parameter DATA_TYPE = 32,
  parameter LATENCY = 4
)(
  // inputs
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] lhs,
  input  lhs_valid,
  input  [DATA_TYPE - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  // outputs
  output [DATA_TYPE - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  //assert(LATENCY != 4) else $fatal("muli only supports LATENCY = 4");

  wire join_valid;
  wire oehb_ready;
  wire buff_valid;

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (oehb_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (join_valid             )
  );

  mul_4_stage #(
    .DATA_TYPE(DATA_TYPE)
  ) mul_4_stage_inst (
    .clk(clk),
    .ce(oehb_ready),
    .a(lhs),
    .b(rhs),
    .p(result)
  );

  delay_buffer #(
    .SIZE( LATENCY - 1)
  ) buff (
    .clk(clk),
    .rst(rst),
    .valid_in(join_valid),
    .ready_in(oehb_ready),
    .valid_out(buff_valid)
  );

  oehb_dataless oehb_inst (
    .clk(clk),
    .rst(rst),
    .ins_valid(buff_valid),
    .ins_ready(oehb_ready),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  );


endmodule
