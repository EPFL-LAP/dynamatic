def generate_mux(name, params):
    
    size = params["size"]
    data_type = params["data_type"]
    select_type = params["select_type"]

    if(data_type == 0):
        return generate_dataless_mux(name, params)

    dataless_mux = f"""
`timescale 1ns/1ps
// Module of smux
module {name} #(
  parameter SIZE = {size},
  parameter DATA_TYPE = {data_type},
  parameter SELECT_TYPE = {select_type}
)(
  input  clk,
  input  rst,
  // Data input channels
  input  [(SIZE * DATA_TYPE) - 1 : 0] ins, 
  input  [SIZE - 1 : 0] ins_valid,
  output reg [SIZE - 1 : 0] ins_ready,
  // Index input channel
  input  [SELECT_TYPE - 1 : 0] index,
  input  index_valid,
  output index_ready,
  // Output channel
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  reg [DATA_TYPE - 1 : 0] selectedData;
  reg selectedData_valid;

  integer i;
  always @(*) begin
    if(rst) begin
      selectedData_valid = 0;
      for (i = DATA_TYPE - 1; i >= 0; i = i - 1) begin
        selectedData[i] = 0;
      end
      for (i = SIZE - 1; i >= 0; i = i - 1) begin
        ins_ready[i] = 0;
      end
    end else begin
      selectedData = ins[0 * DATA_TYPE +: DATA_TYPE];
      selectedData_valid = 0;

      for (i = SIZE - 1; i >= 0; i = i - 1) begin
        if (((i[SELECT_TYPE - 1 : 0] == index) & index_valid & outs_ready & ins_valid[i]) | ~ins_valid[i]) begin
          ins_ready[i] = 1;
        end else begin
          ins_ready[i] = 0;
        end

        if (index == i[SELECT_TYPE - 1 : 0] && index_valid && ins_valid[i]) begin
          selectedData = ins[i * DATA_TYPE +: DATA_TYPE];
          selectedData_valid = 1;
        end
      end
    end 

  end

  assign index_ready = ~index_valid | (selectedData_valid & outs_ready);
  assign outs = selectedData;
  assign outs_valid = selectedData_valid;

endmodule
"""

    return dataless_mux
