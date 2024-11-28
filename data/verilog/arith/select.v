`timescale 1ns/1ps
module antitokens (
  // inputs
  input  clk,
  input  reset,
  input  pvalid1,
  input  pvalid0,
  input  generate_at1,
  input  generate_at0,
  // outputs
  output  kill1,
  output  kill0,
  output  stop_valid
);

  wire reg_in0, reg_in1;
  reg reg_out0, reg_out1;

  always @(posedge reset) begin
    reg_out0 <= 1'b0;
    reg_out1 <= 1'b0;
  end

  always @(posedge clk) begin
    if ( !reset ) begin
      reg_out0 <= reg_in0;
      reg_out1 <= reg_in1;
    end
  end

  assign reg_in0 = !pvalid0 & (generate_at0 | reg_out0);
  assign reg_in1 = !pvalid1 & (generate_at1 | reg_out1);

  assign stop_valid = reg_out0 | reg_out1;

  assign kill0 = generate_at0 | reg_out0;
  assign kill1 = generate_at1 | reg_out1;

endmodule


module selector #(
  parameter DATA_TYPE = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  condition,
  input  condition_valid,
  input  [DATA_TYPE-1 : 0] trueValue,
  input  trueValue_valid,
  input  [DATA_TYPE-1 : 0] falseValue,
  input  falseValue_valid,
  input  result_ready,
  // outputs
  output  [DATA_TYPE-1 : 0] result,
  output  result_valid,
  output  condition_ready,
  output  trueValue_ready,
  output  falseValue_ready
);

  wire ee, validInternal, kill0, kill1, antitokenStop, g0, g1;

  // condition and one input
  assign ee = condition_valid & ( ( !condition & falseValue_valid) | (condition & trueValue_valid) );
  // propagate ee if not stopped antitoken
  assign validInternal = ee & !antitokenStop;

  assign g0 = !trueValue_valid & validInternal & result_ready;
  assign g1 = !falseValue_valid & validInternal & result_ready;

  assign result_valid = validInternal;
  assign trueValue_ready = !trueValue_valid | (validInternal & result_ready) | kill0; // normal join or antitoken
  assign falseValue_ready = !falseValue_valid | (validInternal & result_ready) | kill1; // normal join or antitoken
  assign condition_ready = !condition_valid | (validInternal & result_ready); // normal join

  assign result = condition ? trueValue : falseValue;

  antitokens antitokens (
    .clk(clk),
    .reset(rst),
    .pvalid0(trueValue_valid),
    .pvalid1(falseValue_valid),
    .generate_at0(g0),
    .generate_at1(g1),
    .kill0(kill0),
    .kill1(kill1),
    .stop_valid(antitokenStop)
  );

endmodule
