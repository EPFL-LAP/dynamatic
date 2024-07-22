module mem_controller_storeless #(
  parameter LOAD_COUNT = 1,
  parameter DATA_WIDTH = 32,
  parameter ADDR_WIDTH = 32
)(
  input  clk,
  input  rst,
  // Load address input channels
  input  [(LOAD_COUNT * ADDR_WIDTH) - 1 : 0] ldAddr,
  input  [LOAD_COUNT - 1 : 0] ldAddr_valid,
  output [LOAD_COUNT - 1 : 0] ldAddr_ready,
  // Load data output channels
  output [(LOAD_COUNT * DATA_WIDTH) - 1 : 0] ldData,
  output [LOAD_COUNT - 1 : 0] ldData_valid,
  input  [LOAD_COUNT - 1 : 0] ldData_ready,
  // Memory done channel
  output memDone_valid,
  input  memDone_ready,
  // Interface to dual-port BRAM
  input  [DATA_WIDTH - 1 : 0] loadData,
  output loadEn,
  output [ADDR_WIDTH - 1 : 0] loadAddr,
  output storeEn,
  output [ADDR_WIDTH - 1 : 0] storeAddr,
  output [DATA_WIDTH - 1 : 0] storeData
);
  // No stores will ever be issused
  assign storeAddr = {ADDR_WIDTH{1'b0}};
  assign storeData = {DATA_WIDTH{1'b0}};
  assign storeEn = 1'b0;

  // MC is "always done with stores"
  assign memDone_valid = 1'b1;

  read_memory_arbiter #(
    .ARBITER_SIZE (LOAD_COUNT),
    .ADDR_WIDTH   (ADDR_WIDTH),
    .DATA_WIDTH   (DATA_WIDTH)
  ) read_arbiter (
    .rst             (rst         ),
    .clk             (clk         ),
    .pValid          (ldAddr_valid),
    .ready           (ldAddr_ready),
    .address_in      (ldAddr      ),
    .nReady          (ldData_ready),
    .valid           (ldData_valid),
    .data_out        (ldData      ),
    .read_enable     (loadEn      ),
    .read_address    (loadAddr    ),
    .data_from_memory(loadData    )
  );

endmodule
