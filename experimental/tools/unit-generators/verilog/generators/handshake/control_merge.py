from generators.handshake.fork import generate_datalessFork
from generators.handshake.tehb import generate_tehb
from generators.handshake.merge import generate_dataless_merge

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

def generate_dataless_control_merge(name, params):

    size = params["size"]
    index_type = params["index_type"]

    header = "`timescale 1ns/1ps\n"

    dataless_merge_name = name + "_dataless_merge"
    dataless_merge = generate_dataless_merge(dataless_merge_name, {"size": size})

    tehb_name = name + "_tehb"
    tehb = generate_tehb(tehb_name, {"data_type": index_type})

    fork_dataless_name = name + "_fork_dataless"
    fork_dataless = generate_datalessFork(fork_dataless_name, {"size": 2})

    dataless_controll_merge_body=f"""
// Module of dataless_control_merge
module {name}(
  input  clk,
  input  rst,
  // Input Channels, default 2 inputs
  input  [{size} - 1 : 0] ins_valid,
  output [{size} - 1 : 0] ins_ready,  
  // Data Output Channel
  output outs_valid,
  input  outs_ready,            
  // Index output Channel
  output [{index_type} - 1 : 0] index,
  output index_valid,
  input  index_ready
);
  wire dataAvailable;
  wire readyToFork;
  wire tehbOut_valid;
  wire tehbOut_ready;

  reg [{index_type} - 1 : 0] index_tehb;
  integer i;
  reg found;
  always @(ins_valid) begin
    index_tehb = {{{index_type}{{1'b0}}}};
    found = 1'b0;

    for (i = 0; i < {size}; i = i + 1) begin
      if (!found && ins_valid[i]) begin
        index_tehb = i[{index_type} - 1 : 0];
        found = 1'b1; // Set flag to indicate the value has been found
      end
    end
  end

  // Instantiate Merge_dataless
  {dataless_merge_name} merge_ins (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins_valid  (ins_valid    ),
    .ins_ready  (ins_ready    ),
    .outs_valid (dataAvailable),
    .outs_ready (tehbOut_ready)
  );

  // Instantiate TEHB
  {tehb_name} tehb (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins        (index_tehb   ),
    .ins_valid  (dataAvailable),
    .ins_ready  (tehbOut_ready),
    .outs       (index        ),
    .outs_valid (tehbOut_valid),
    .outs_ready (readyToFork  )
  );

  // Instantiate Fork_dataless
  {fork_dataless_name} fork_dataless (
    .clk        (clk                      ),
    .rst        (rst                      ),
    .ins_valid  (tehbOut_valid            ),
    .ins_ready  (readyToFork              ),
    .outs_valid ({{index_valid, outs_valid}}),
    .outs_ready ({{index_ready, outs_ready}})
  );

endmodule
"""

    return header + dataless_merge + tehb + fork_dataless + dataless_controll_merge_body
