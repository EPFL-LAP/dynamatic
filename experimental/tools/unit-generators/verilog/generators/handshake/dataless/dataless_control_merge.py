from generators.handshake.dataless.dataless_merge import generate_dataless_merge
from generators.handshake.dataless.datalessFork import generate_datalessFork
from generators.handshake.tehb import generate_tehb

def generate_dataless_control_merge(name, params):

    size = params["size"]
    index_type = params["index_type"]

    verilog_header = "`timescale 1ns/1ps\n"

    dataless_merge_name = name + "_dataless_merge"
    verilog_dataless_merge = generate_dataless_merge(dataless_merge_name, params)

    tehb_name = name + "_tehb"
    verilog_tehb = generate_tehb(tehb_name, {"data_type": index_type})

    fork_dataless_name = name + "_fork_dataless"
    verilog_fork_dataless = generate_datalessFork(fork_dataless_name, params)

    verilog_dataless_controll_merge_body=f"""
//dataless_control_merge module
module {name} #(
  parameter SIZE = {size},
  parameter INDEX_TYPE = {index_type}
)(
  input  clk,
  input  rst,
  // Input Channels, default 2 inputs
  input  [SIZE - 1 : 0] ins_valid,
  output [SIZE - 1 : 0] ins_ready,  
  // Data Output Channel
  output outs_valid,
  input  outs_ready,            
  // Index output Channel
  output [INDEX_TYPE - 1 : 0] index,
  output index_valid,
  input  index_ready
);
  wire dataAvailable;
  wire readyToFork;
  wire tehbOut_valid;
  wire tehbOut_ready;

  reg [INDEX_TYPE - 1 : 0] index_tehb;
  integer i;
  reg found;
  always @(ins_valid) begin
    index_tehb = {{INDEX_TYPE{{1'b0}}}};
    found = 1'b0;

    for (i = 0; i < SIZE; i = i + 1) begin
      if (!found && ins_valid[i]) begin
        index_tehb = i[INDEX_TYPE - 1 : 0];
        found = 1'b1; // Set flag to indicate the value has been found
      end
    end
  end

  // Instantiate Merge_dataless
  {dataless_merge_name} #(
    .SIZE(SIZE)
  ) merge_ins (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins_valid  (ins_valid    ),
    .ins_ready  (ins_ready    ),
    .outs_valid (dataAvailable),
    .outs_ready (tehbOut_ready)
  );

  // Instantiate TEHB
  {tehb_name} #(
    .DATA_TYPE(INDEX_TYPE)
  ) tehb (
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
  {fork_dataless_name} #(
    .SIZE(2)
  ) fork_dataless (
    .clk        (clk                      ),
    .rst        (rst                      ),
    .ins_valid  (tehbOut_valid            ),
    .ins_ready  (readyToFork              ),
    .outs_valid ({{index_valid, outs_valid}}),
    .outs_ready ({{index_ready, outs_ready}})
  );

endmodule
"""

    return verilog_header + verilog_dataless_merge + verilog_tehb + verilog_fork_dataless + verilog_dataless_controll_merge_body