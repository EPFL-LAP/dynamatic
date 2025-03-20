`timescale 1ns/1ps
module control_merge_dataless #(
  parameter SIZE = 2,
  parameter INDEX_TYPE = 1
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

  reg [INDEX_TYPE - 1 : 0] index_internal;
  integer i;
  reg found;
  always @(ins_valid) begin
    index_internal = {INDEX_TYPE{1'b0}};
    found = 1'b0;

    for (i = 0; i < SIZE; i = i + 1) begin
      if (!found && ins_valid[i]) begin
        index_internal = i[INDEX_TYPE - 1 : 0];
        found = 1'b1; // Set flag to indicate the value has been found
      end
    end
  end

  // Instantiate Merge_dataless
  merge_notehb_dataless #(
    .INPUTS(SIZE)
  ) merge_ins (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins_valid  (ins_valid    ),
    .ins_ready  (ins_ready    ),
    .outs_valid (dataAvailable),
    .outs_ready (readyToFork  )
  );

  assign index = index_internal;

  // Instantiate Fork_dataless
  fork_dataless #(
    .SIZE(2)
  ) fork_dataless (
    .clk        (clk                      ),
    .rst        (rst                      ),
    .ins_valid  (dataAvailable            ),
    .ins_ready  (readyToFork              ),
    .outs_valid ({index_valid, outs_valid}),
    .outs_ready ({index_ready, outs_ready})
  );

endmodule
