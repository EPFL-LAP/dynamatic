module mem_controller_loadless #(
  parameter NUM_CONTROL = 1,
  parameter NUM_STORE = 1,
  parameter DATA_WIDTH = 32,
  parameter ADDR_WIDTH = 32
)(
  input  clk,
  input  rst,
  // Control Input Channels
  input  [(NUM_CONTROL * 32) - 1 : 0] ctrl,
  input  [NUM_CONTROL - 1 : 0] ctrl_valid,
  output [NUM_CONTROL - 1 : 0] ctrl_ready,
  // Store Address Input Channels
  input  [(NUM_STORE * ADDR_WIDTH) - 1 : 0] stAddr,
  input  [NUM_STORE - 1 : 0] stAddr_valid,
  output [NUM_STORE - 1 : 0] stAddr_ready,
  // Store Data Input Channels
  input  [(NUM_STORE * DATA_WIDTH) - 1 : 0] stData,
  input  [NUM_STORE - 1 : 0] stData_valid,
  output [NUM_STORE - 1 : 0] stData_ready,
  // Memory Done Channel
  output memDone_valid,
  input  memDone_ready,
  // Interface to Dual-port BRAM
  input  [DATA_WIDTH - 1 : 0] loadData,
  output loadEn,
  output [ADDR_WIDTH - 1 : 0] loadAddr,
  output storeEn,
  output [ADDR_WIDTH - 1 : 0] storeAddr,
  output [DATA_WIDTH - 1 : 0] storeData
);
  // Internal Signals
  reg [31 : 0] remainingStores;
  wire [NUM_STORE - 1 : 0] storePorts_valid, storePorts_ready;

  // Local Parameter
  localparam [31:0] zeroStore = 32'b0;
  localparam [NUM_CONTROL-1:0] zeroCtrl = {NUM_CONTROL{1'b0}};

  assign loadEn = 0;
  assign loadAddr = {ADDR_WIDTH{1'b0}};

  // Instantiate write memory arbiter
  write_memory_arbiter #(
    .ARBITER_SIZE (NUM_STORE),
    .ADDR_WIDTH   (ADDR_WIDTH),
    .DATA_WIDTH   (DATA_WIDTH)
  ) write_arbiter (
    .rst            (rst                ),
    .clk            (clk                ),
    .pValid         (stAddr_valid       ),
    .ready          (storePorts_ready   ),
    .address_in     (stAddr             ),
    .data_in        (stData             ),
    .nReady         ({NUM_STORE{1'b1}}),
    .valid          (storePorts_valid   ),
    .write_enable   (storeEn            ),
    .write_address  (storeAddr          ),
    .data_to_memory (storeData          )
  );

  assign stData_ready = storePorts_ready;
  assign stAddr_ready = storePorts_ready;
  integer i;
  reg [31 : 0] counter;

  // Counting Stores
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      counter <= 32'd0;
    end else begin
      for (i = 0; i < NUM_CONTROL; i = i + 1) begin
        if (ctrl_valid[i]) begin
          counter <= remainingStores + ctrl[(i * 32) + 31 -: 32];
        end
      end
      if (storeEn) begin
        counter <= remainingStores - 1;
      end
      remainingStores <= counter;
    end
  end

  // Memory Done logic
  assign memDone_valid = (remainingStores == zeroStore && ctrl_valid == zeroCtrl) ? 1'b1 : 1'b0;
  assign ctrl_ready = {NUM_CONTROL{1'b1}};

endmodule
