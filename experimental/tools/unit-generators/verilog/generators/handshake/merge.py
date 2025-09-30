def generate_merge(name, params):
    # Number of input ports
    size = params["size"]
    datatype = params["datatype"]

    if(datatype == 0):
        return generate_dataless_merge(name, {"size": size})

    return f"""
`timescale 1ns/1ps

// Module of merge

module {name}(
  input  clk,
  input  rst,
  // Input channels
  input  [{size} * {datatype} - 1 : 0] ins, 
  input  [{size} - 1 : 0] ins_valid,
  output [{size} - 1 : 0] ins_ready,
  // Output channel
  output [{datatype} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);

  reg tmp_valid_out;
  reg [{size} - 1 : 0] tmp_ready_out;
  reg [{datatype} - 1 : 0] tmp_data_out;
  integer i;
  integer cnt;

  always @(*) begin
    tmp_valid_out = 0;
    tmp_ready_out = {{{size}{{1'b0}}}};
    tmp_data_out = ins[0 +: {datatype}];

    cnt = 1;
    for (i = 0; i < {size}; i = i + 1) begin
      if (cnt == 1 && ins_valid[i]) begin
        tmp_data_out = ins[i * {datatype} +: {datatype}];
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


def generate_dataless_merge(name, params):
    # Number of input ports
    size = params["size"]


    return f"""
`timescale 1ns/1ps
// Module of dataless_merge
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
