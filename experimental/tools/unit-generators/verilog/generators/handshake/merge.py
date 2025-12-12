
def generate_merge(name, params):
    # Number of intput ports
    size = params["size"]
    bitwidth = params["bitwidth"]
    
    if bitwidth == 0:
        return _generate_merge_dataless(name, size)
    else:
        return _generate_merge(name, size, bitwidth)


def _generate_merge(name, size, bitwidth):

    return f"""

// Module of merge

module {name}(
  input  clk,
  input  rst,
  // Input channels
  input  [{size} * {bitwidth} - 1 : 0] ins, 
  input  [{size} - 1 : 0] ins_valid,
  output [{size} - 1 : 0] ins_ready,
  // Output channel
  output [{bitwidth} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);

  reg tmp_valid_out;
  reg [{size} - 1 : 0] tmp_ready_out;
  reg [{bitwidth} - 1 : 0] tmp_data_out;
  integer i;
  integer cnt;

  always @(*) begin
    tmp_valid_out = 0;
    tmp_ready_out = {{{size}{{1'b0}}}};
    tmp_data_out = ins[0 +: {bitwidth}];

    cnt = 1;
    for (i = 0; i < {size}; i = i + 1) begin
      if (cnt == 1 && ins_valid[i]) begin
        tmp_data_out = ins[i * {bitwidth} +: {bitwidth}];
        tmp_valid_out = 1;
        tmp_ready_out[i] = outs_ready;
        cnt = 0;
      end
    end
  end
  
  // The outs channel is not persistent, meaning the data payload 
  // may change while valid remains high
  assign outs = tmp_data_out;
  assign outs_valid = tmp_valid_out;
  assign ins_ready = tmp_ready_out;

endmodule

"""


def _generate_merge_dataless(name, size):

    return f"""
// Module of merge_dataless
module {name}(
  input  clk,
  input  rst,
  // Input Channels
  input  [{size} - 1 : 0] ins_valid,
  output [{size} - 1 : 0] ins_ready,
  // Output Channel
  output outs_valid,
  input  outs_ready
);

  reg tmp_valid_out;
  reg [{size} - 1 : 0] tmp_ready_out;
  integer i;

  always @(*) begin
    tmp_valid_out = 0;
    tmp_ready_out = {{{size}{{1'b0}}}}; 
    for (i = 0; i < {size}; i = i + 1) begin
      if (ins_valid[i] && !tmp_valid_out) begin
        tmp_valid_out = 1;
        tmp_ready_out[i] = outs_ready;
      end
    end
  end
  
  assign outs_valid = tmp_valid_out;
  assign ins_ready = tmp_ready_out;

endmodule
"""
