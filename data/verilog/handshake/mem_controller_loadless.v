`timescale 1ns / 1ps
module mem_controller_loadless #(
  parameter NUM_CONTROLS = 1,
  parameter NUM_STORES   = 1,
  parameter DATA_TYPE   = 32,
  parameter ADDR_TYPE   = 32
) (
  input                                      clk,
  input                                      rst,
  // start input control
  input                                      memStart_valid,
  output                                     memStart_ready,
  // end output control
  output                                     memEnd_valid,
  input                                      memEnd_ready,
  // "no more requests" input control
  input                                      ctrlEnd_valid,
  output                                     ctrlEnd_ready,
  // Control Input Channels
  input  [      (NUM_CONTROLS * 32) - 1 : 0] ctrl,
  input  [             NUM_CONTROLS - 1 : 0] ctrl_valid,
  output [             NUM_CONTROLS - 1 : 0] ctrl_ready,
  // Store Address Input Channels
  input  [(NUM_STORES * ADDR_TYPE) - 1 : 0] stAddr,
  input  [               NUM_STORES - 1 : 0] stAddr_valid,
  output [               NUM_STORES - 1 : 0] stAddr_ready,
  // Store Data Input Channels
  input  [(NUM_STORES * DATA_TYPE) - 1 : 0] stData,
  input  [               NUM_STORES - 1 : 0] stData_valid,
  output [               NUM_STORES - 1 : 0] stData_ready,
  // Interface to Dual-port BRAM
  input  [               DATA_TYPE - 1 : 0] loadData,
  output                                     loadEn,
  output [               ADDR_TYPE - 1 : 0] loadAddr,
  output                                     storeEn,
  output [               ADDR_TYPE - 1 : 0] storeAddr,
  output [               DATA_TYPE - 1 : 0] storeData
);
  // Internal Signals
  wire [31 : 0] remainingStores;
  wire [NUM_STORES - 1 : 0] storePorts_valid, storePorts_ready;
  wire allRequestsDone;

  // Local Parameter
  localparam [31:0] zeroStore = 32'b0;
  localparam [NUM_CONTROLS-1:0] zeroCtrl = {NUM_CONTROLS{1'b0}};

  assign loadEn   = 0;
  assign loadAddr = {ADDR_TYPE{1'b0}};

  // Instantiate write memory arbiter
  write_memory_arbiter #(
    .ARBITER_SIZE(NUM_STORES),
    .ADDR_TYPE  (ADDR_TYPE),
    .DATA_TYPE  (DATA_TYPE)
  ) write_arbiter (
    .rst           (rst),
    .clk           (clk),
    .pValid        (stAddr_valid),
    .ready         (storePorts_ready),
    .address_in    (stAddr),
    .data_in       (stData),
    .nReady        ({NUM_STORES{1'b1}}),
    .valid         (storePorts_valid),
    .write_enable  (storeEn),
    .write_address (storeAddr),
    .data_to_memory(storeData)
  );

  assign stData_ready = storePorts_ready;
  assign stAddr_ready = storePorts_ready;
  assign ctrl_ready   = {NUM_CONTROLS{1'b1}};

  integer          i;
  reg     [31 : 0] counter;

  // Counting Stores
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      counter = 32'd0;
    end else begin
      for (i = 0; i <= NUM_CONTROLS - 1; i = i + 1) begin
        if (ctrl_valid[i]) begin
          counter = counter + ctrl[i*32+:32];
        end
      end
      if (storeEn) begin
        counter = counter - 1;
      end
    end
  end

  assign remainingStores = counter;

  // NOTE: (lucas-rami) In addition to making sure there are no stores pending,
  // we should also check that there are no loads pending as well. To achieve 
  // this the control signals could simply start indicating the total number
  // of accesses in the block instead of just the number of stores.
  assign allRequestsDone = (remainingStores == zeroStore && ctrl_valid == zeroCtrl) ? 1'b1 : 1'b0;

  mc_control control (
    .rst            (rst),
    .clk            (clk),
    .memStart_valid (memStart_valid),
    .memStart_ready (memStart_ready),
    .memEnd_valid   (memEnd_valid),
    .memEnd_ready   (memEnd_ready),
    .ctrlEnd_valid  (ctrlEnd_valid),
    .ctrlEnd_ready  (ctrlEnd_ready),
    .allRequestsDone(allRequestsDone)
  );

endmodule
