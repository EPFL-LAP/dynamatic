def generate_dataless_mux(name, params):
    # Number of input ports
    size = params["size"]
    select_type = params["select_type"]

    verilog_dataless_mux = f"""
`timescale 1ns/1ps
// Module of dataless_mux
module {name} #(
  parameter SIZE = {size},
  parameter SELECT_TYPE = {select_type}
)(
  input  clk,
  input  rst,
  // Data input channels
  input  [SIZE - 1 : 0] ins_valid,
  output reg [SIZE - 1 : 0] ins_ready,
  // Index input channel
  input  [SELECT_TYPE - 1 : 0] index,
  input  index_valid,
  output index_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  reg selectedData_valid;
  integer i;

  always @(*) begin
    selectedData_valid = 0;

    for (i = SIZE - 1; i >= 0 ; i = i - 1) begin
      if (((i[SELECT_TYPE - 1 : 0] == index) & index_valid & ins_valid[i] & outs_ready) | ~ins_valid[i])
        ins_ready[i] = 1;
      else
        ins_ready[i] = 0;

      if (index == i[SELECT_TYPE - 1 : 0] && index_valid && ins_valid[i]) begin
        selectedData_valid = 1;
      end    
    end
  end

  assign outs_valid = selectedData_valid;
  assign index_ready = ~index_valid | (selectedData_valid & outs_ready);

endmodule
"""

    return verilog_dataless_mux
