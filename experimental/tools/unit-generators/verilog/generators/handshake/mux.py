def generate_mux(name, params):
    
    size = params["size"]
    data_type = params["data_type"]
    select_type = params["select_type"]

    if(data_type == 0):
        return generate_dataless_mux(name, {"size": size, "select_type": select_type})

    dataless_mux = f"""
`timescale 1ns/1ps
// Module of smux
module {name}(
  input  clk,
  input  rst,
  // Data input channels
  input  [({size} * {data_type}) - 1 : 0] ins, 
  input  [{size} - 1 : 0] ins_valid,
  output reg [{size} - 1 : 0] ins_ready,
  // Index input channel
  input  [{select_type} - 1 : 0] index,
  input  index_valid,
  output index_ready,
  // Output channel
  output [{data_type} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  reg [{data_type} - 1 : 0] selectedData;
  reg selectedData_valid;

  integer i;
  always @(*) begin
    if(rst) begin
      selectedData_valid = 0;
      for (i = {data_type} - 1; i >= 0; i = i - 1) begin
        selectedData[i] = 0;
      end
      for (i = {size} - 1; i >= 0; i = i - 1) begin
        ins_ready[i] = 0;
      end
    end else begin
      selectedData = ins[0 * {data_type} +: {data_type}];
      selectedData_valid = 0;

      for (i = {size} - 1; i >= 0; i = i - 1) begin
        if (((i[{select_type} - 1 : 0] == index) & index_valid & outs_ready & ins_valid[i]) | ~ins_valid[i]) begin
          ins_ready[i] = 1;
        end else begin
          ins_ready[i] = 0;
        end

        if (index == i[{select_type} - 1 : 0] && index_valid && ins_valid[i]) begin
          selectedData = ins[i * {data_type} +: {data_type}];
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

def generate_dataless_mux(name, params):
    # Number of input ports
    size = params["size"]
    select_type = params["select_type"]

    dataless_mux = f"""
`timescale 1ns/1ps
// Module of dataless_mux
module {name}(
  input  clk,
  input  rst,
  // Data input channels
  input  [{size} - 1 : 0] ins_valid,
  output reg [{size} - 1 : 0] ins_ready,
  // Index input channel
  input  [{select_type} - 1 : 0] index,
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

    for (i = {size} - 1; i >= 0 ; i = i - 1) begin
      if (((i[{select_type} - 1 : 0] == index) & index_valid & ins_valid[i] & outs_ready) | ~ins_valid[i])
        ins_ready[i] = 1;
      else
        ins_ready[i] = 0;

      if (index == i[{select_type} - 1 : 0] && index_valid && ins_valid[i]) begin
        selectedData_valid = 1;
      end    
    end
  end

  assign outs_valid = selectedData_valid;
  assign index_ready = ~index_valid | (selectedData_valid & outs_ready);

endmodule
"""

    return dataless_mux
