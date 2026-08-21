`timescale 1ns/1ps
module burst_load #(
  parameter DATA_TYPE = 32,
  parameter ADDR_TYPE = 32,
  parameter BURST_TYPE = 4
)(
  input clk,
  input rst,
  // Address from Circuit Channel
  input  [ADDR_TYPE - 1 : 0] addrIn,
  input  addrIn_valid,
  output addrIn_ready,
  // Address to Memory Channel
  output [ADDR_TYPE - 1 : 0] addrOut,
  output addrOut_valid,
  input  addrOut_ready,
  // Burst length from Circuit Channel
  input  [BURST_TYPE - 1 : 0] burstLength,
  input  burstLength_valid,
  output burstLength_ready,
  // Burst length to Memory Channel
  output [BURST_TYPE - 1 : 0] burstLengthResult,
  output burstLengthResult_valid,
  input  burstLengthResult_ready,
  // Data from Interface Channel
  input  [DATA_TYPE - 1 : 0] dataFromMem,
  input  dataFromMem_valid,
  output dataFromMem_ready,
  // Data from Memory Channel
  output [DATA_TYPE - 1 : 0] dataOut,
  output dataOut_valid,
  input  dataOut_ready
);
  tehb #(
    .DATA_TYPE(ADDR_TYPE)
  ) addr_tehb (
    .clk        (clk            ),
    .rst        (rst            ),
    .ins        (addrIn         ),
    .ins_valid  (addrIn_valid   ),
    .ins_ready  (addrIn_ready   ),
    .outs       (addrOut        ),
    .outs_valid (addrOut_valid  ),
    .outs_ready (addrOut_ready  )
  );

  tehb #(
    .DATA_TYPE(DATA_TYPE)
  ) data_tehb (
    .clk        (clk                ),
    .rst        (rst                ),
    .ins        (dataFromMem        ),
    .ins_valid  (dataFromMem_valid  ),
    .ins_ready  (dataFromMem_ready  ),
    .outs       (dataOut            ),
    .outs_valid (dataOut_valid      ),
    .outs_ready (dataOut_ready      )
  );

  tehb #(
    .DATA_TYPE(BURST_TYPE)
  ) burst_tehb (
    .clk        (clk                 ),
    .rst        (rst                 ),
    .ins        (burstLength         ),
    .ins_valid  (burstLength_valid   ),
    .ins_ready  (burstLength_ready   ),
    .outs       (burstLengthResult   ),
    .outs_valid (burstLengthResult_valid),
    .outs_ready (burstLengthResult_ready)
  );

endmodule
