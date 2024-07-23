module end_sync_dataless #(
  parameter NUM_MEMORIES = 2
)(
  input  clk,
  input  rst,
  // Input Channel
  input  ins_valid,
  output ins_ready,
  // Memory Input Channels
  input  [NUM_MEMORIES - 1 : 0] memDone_valid,
  output [NUM_MEMORIES - 1 : 0] memDone_ready,
  // Output Channel
  output outs_valid,
  input  outs_ready
);
  wire memReady, allMemDone;  //! Dangling wire for memReady

  assign memDone_ready = {NUM_MEMORIES{1'b1}};

  // Instantiate of and_n
  and_n #(
    .SIZE(NUM_MEMORIES)
  ) mem_and (
    .ins  (memDone_valid),
    .outs (allMemDone   )
  );

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_ins_mem (
    .ins_valid  ({allMemDone, ins_valid}),
    .outs_ready (outs_ready             ),
    .ins_ready  ({memReady, ins_ready}  ),
    .outs_valid (outs_valid             )
  );

endmodule
