`timescale 1ns / 1ps
module mem_controller_storeless #(
  parameter NUM_LOADS  = 1,
  parameter DATA_TYPE = 32,
  parameter ADDR_TYPE = 32
) (
  input                                     clk,
  input                                     rst,
  // start input control
  input                                     memStart_valid,
  output                                    memStart_ready,
  // end output control
  output                                    memEnd_valid,
  input                                     memEnd_ready,
  // "no more requests" input control
  input                                     ctrlEnd_valid,
  output                                    ctrlEnd_ready,
  // Load address input channels
  input  [(NUM_LOADS * ADDR_TYPE) - 1 : 0] ldAddr,
  input  [               NUM_LOADS - 1 : 0] ldAddr_valid,
  output [               NUM_LOADS - 1 : 0] ldAddr_ready,
  // Load data output channels
  output [(NUM_LOADS * DATA_TYPE) - 1 : 0] ldData,
  output [               NUM_LOADS - 1 : 0] ldData_valid,
  input  [               NUM_LOADS - 1 : 0] ldData_ready,
  // Interface to dual-port BRAM
  input  [              DATA_TYPE - 1 : 0] loadData,
  output                                    loadEn,
  output [              ADDR_TYPE - 1 : 0] loadAddr,
  output                                    storeEn,
  output [              ADDR_TYPE - 1 : 0] storeAddr,
  output [              DATA_TYPE - 1 : 0] storeData
);
  wire allRequestsDone;

  // No stores will ever be issused
  assign storeAddr = {ADDR_TYPE{1'b0}};
  assign storeData = {DATA_TYPE{1'b0}};
  assign storeEn   = 1'b0;

  // MC is "always done with stores"

  read_memory_arbiter #(
    .ARBITER_SIZE(NUM_LOADS),
    .ADDR_TYPE  (ADDR_TYPE),
    .DATA_TYPE  (DATA_TYPE)
  ) read_arbiter (
    .rst             (rst),
    .clk             (clk),
    .pValid          (ldAddr_valid),
    .ready           (ldAddr_ready),
    .address_in      (ldAddr),
    .nReady          (ldData_ready),
    .valid           (ldData_valid),
    .data_out        (ldData),
    .read_enable     (loadEn),
    .read_address    (loadAddr),
    .data_from_memory(loadData)
  );

  // NOTE: (lucas-rami) In addition to making sure there are no stores pending,
  // we should also check that there are no loads pending as well. To achieve 
  // this the control signals could simply start indicating the total number
  // of accesses in the block instead of just the number of stores.
  assign allRequestsDone = 1'b1;

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
