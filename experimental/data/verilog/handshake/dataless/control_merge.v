module control_merge_dataless #(
  parameter SIZE = 2,
  parameter INDEX_WIDTH = 1
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
  output [INDEX_WIDTH - 1 : 0] index,
  output index_valid,
  input  index_ready
);
  wire dataAvailable;
  wire readyToFork;
  wire tehbOut_valid;
  wire tehbOut_ready;

  //! Is it possible that the unit has more than 2 inputs?, Jiantao, 21/07/2024
  // reg [INDEX_WIDTH - 1 : 0] index_tehb;
  // integer i;
  // always @(ins_valid) begin
  //   index_tehb = {INDEX_WIDTH{1'b0}};
  //   for (i = 0; i < SIZE; i = i + 1) begin
  //     if (ins_valid[i]) begin
  //       index_tehb = i[INDEX_WIDTH - 1 : 0];
  //       break;      // Exit the loop on the first valid
  //     end
  //   end
  // end

  //! Assuming SIZE = 2
  wire [INDEX_WIDTH - 1 : 0] index_tehb;
  assign index_tehb = ~ins_valid[0];

  // Instantiate Merge_dataless
  merge_notehb_dataless #(
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
  tehb #(
    .DATA_WIDTH(INDEX_WIDTH)
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
  fork_dataless #(
    .SIZE(2)
  ) fork_dataless (
    .clk        (clk                      ),
    .rst        (rst                      ),
    .ins_valid  (tehbOut_valid            ),
    .ins_ready  (readyToFork              ),
    .outs_valid ({index_valid, outs_valid}),
    .outs_ready ({index_ready, outs_ready})
  );

endmodule
