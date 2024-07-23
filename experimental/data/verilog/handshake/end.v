module end_sync #(
  parameter DATA_WIDTH = 32,
  parameter MEM_INPUTS = 2
)(
  input  clk,
  input  rst,
  // Input Channel
  input  [DATA_WIDTH - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Memory Input Channels
  input  [MEM_INPUTS - 1 : 0] memDone_valid,
  output [MEM_INPUTS - 1 : 0] memDone_ready,
  // Output Channel
  output [DATA_WIDTH - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  assign memDone_valid = {MEM_INPUTS{1'b1}};

  end_sync_dataless #(
    .MEM_INPUTS(MEM_INPUTS)
  ) control (
    .clk            (clk          ),
    .rst            (rst          ),
    .ins_valid      (ins_valid    ),
    .ins_ready      (ins_ready    ),
    .memDone_valid  (memDone_valid),
    .memDone_ready  (memDone_ready),
    .outs_valid     (outs_valid   ),
    .outs_ready     (outs_ready   )
  );

  assign outs = ins;
endmodule
