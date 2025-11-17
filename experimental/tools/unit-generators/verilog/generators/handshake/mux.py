
def generate_mux(name, params):
    # Number of data input ports
    size = params["size"]

    data_bitwidth = params["data_bitwidth"]
    index_bitwidth = params["index_bitwidth"]

    if data_bitwidth == 0:
        return _generate_mux_dataless(name, size, index_bitwidth)
    else:
        return _generate_mux(name, size, index_bitwidth, data_bitwidth)


def _generate_mux(name, size, index_bitwidth, data_bitwidth):

    _mux_dataless = f"""
// Module of smux
module {name}(
  input  clk,
  input  rst,
  // Data input channels
  input  [({size} * {data_bitwidth}) - 1 : 0] ins, 
  input  [{size} - 1 : 0] ins_valid,
  output reg [{size} - 1 : 0] ins_ready,
  // Index input channel
  input  [{index_bitwidth} - 1 : 0] index,
  input  index_valid,
  output index_ready,
  // Output channel
  output [{data_bitwidth} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  reg [{data_bitwidth} - 1 : 0] selectedData;
  reg selectedData_valid;

  integer i;
  always @(*) begin
    if(rst) begin
      selectedData_valid = 0;
      for (i = {data_bitwidth} - 1; i >= 0; i = i - 1) begin
        selectedData[i] = 0;
      end
      for (i = {size} - 1; i >= 0; i = i - 1) begin
        ins_ready[i] = 0;
      end
    end else begin
      selectedData = ins[0 * {data_bitwidth} +: {data_bitwidth}];
      selectedData_valid = 0;

      for (i = {size} - 1; i >= 0; i = i - 1) begin
        if (((i[{index_bitwidth} - 1 : 0] == index) & index_valid & outs_ready & ins_valid[i]) | ~ins_valid[i]) begin
          ins_ready[i] = 1;
        end else begin
          ins_ready[i] = 0;
        end

        if (index == i[{index_bitwidth} - 1 : 0] && index_valid && ins_valid[i]) begin
          selectedData = ins[i * {data_bitwidth} +: {data_bitwidth}];
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
    return _mux_dataless


def _generate_mux_dataless(name, size, index_bitwidth):

    _mux_dataless = f"""
// Module of_mux_dataless
module {name}(
  input  clk,
  input  rst,
  // Data input channels
  input  [{size} - 1 : 0] ins_valid,
  output reg [{size} - 1 : 0] ins_ready,
  // Index input channel
  input  [{index_bitwidth} - 1 : 0] index,
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
      if (((i[{index_bitwidth} - 1 : 0] == index) & index_valid & ins_valid[i] & outs_ready) | ~ins_valid[i])
        ins_ready[i] = 1;
      else
        ins_ready[i] = 0;

      if (index == i[{index_bitwidth} - 1 : 0] && index_valid && ins_valid[i]) begin
        selectedData_valid = 1;
      end    
    end
  end

  assign outs_valid = selectedData_valid;
  assign index_ready = ~index_valid | (selectedData_valid & outs_ready);

endmodule
"""

    return _mux_dataless
