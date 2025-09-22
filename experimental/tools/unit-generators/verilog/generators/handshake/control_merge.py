from generators.handshake.dataless.dataless_control_merge import generate_dataless_control_merge
def generate_control_merge(name, params):

    size = params["size"]
    data_type = params["data_type"]
    index_type = params["index_type"]

    if(data_type == 0):
      return generate_dataless_control_merge(name, params)

    verilog_header = "`timescale 1ns/1ps\n"

    dataless_control_merge_name = name + "_control_merge_dataless"
    verilog_dataless_control_merge = generate_dataless_control_merge(dataless_control_merge_name, params)

    verilog_control_merge_body = f"""
// Module of control_merge
  module {name} #(
    parameter SIZE = {size},
    parameter DATA_TYPE = {data_type},
    parameter INDEX_TYPE = {index_type}
  )(
    input  clk,
    input  rst,
    // Input Channels
    input  [SIZE * (DATA_TYPE) - 1 : 0] ins,
    input  [SIZE - 1 : 0] ins_valid,
    output [SIZE - 1 : 0] ins_ready,
    // Data Output Channel
    output [DATA_TYPE - 1 : 0] outs,
    output outs_valid,
    input  outs_ready,
    // Index Output Channel
    output [INDEX_TYPE - 1 : 0] index,
    output index_valid,
    input  index_ready
  );
    wire [INDEX_TYPE - 1 : 0] index_internal;

    {dataless_control_merge_name} #(
      .SIZE(SIZE),
      .INDEX_TYPE(INDEX_TYPE)
    ) control (
      .clk          (clk            ),
      .rst          (rst            ),
      .ins_valid    (ins_valid      ),
      .ins_ready    (ins_ready      ),
      .outs_valid   (outs_valid     ),
      .outs_ready   (outs_ready     ),
      .index        (index_internal ),
      .index_valid  (index_valid    ),
      .index_ready  (index_ready    )
    );

    assign index = index_internal;

    assign outs = ins[index_internal * DATA_TYPE +: DATA_TYPE];
  endmodule

"""

    return verilog_header + verilog_dataless_control_merge + verilog_control_merge_body