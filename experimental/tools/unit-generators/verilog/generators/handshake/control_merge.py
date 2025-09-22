from generators.handshake.dataless.dataless_control_merge import generate_dataless_control_merge
def generate_control_merge(name, params):

    size = params["size"]
    data_type = params["data_type"]
    index_type = params["index_type"]

    if(data_type == 0):
      return generate_dataless_control_merge(name, {"size": size, "index_type": index_type})

    header = "`timescale 1ns/1ps\n"

    dataless_control_merge_name = name + "_control_merge_dataless"
    dataless_control_merge = generate_dataless_control_merge(dataless_control_merge_name, {"size": size, "index_type": index_type})

    control_merge_body = f"""
// Module of control_merge
  module {name}(
    input  clk,
    input  rst,
    // Input Channels
    input  [{size} * ({data_type}) - 1 : 0] ins,
    input  [{size} - 1 : 0] ins_valid,
    output [{size} - 1 : 0] ins_ready,
    // Data Output Channel
    output [{data_type} - 1 : 0] outs,
    output outs_valid,
    input  outs_ready,
    // Index Output Channel
    output [{index_type} - 1 : 0] index,
    output index_valid,
    input  index_ready
  );
    wire [{index_type} - 1 : 0] index_internal;

    {dataless_control_merge_name} control (
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

    assign outs = ins[index_internal * {data_type} +: {data_type}];
  endmodule

"""

    return header + dataless_control_merge + control_merge_body