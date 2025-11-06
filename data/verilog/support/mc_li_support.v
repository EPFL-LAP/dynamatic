
//
//  read_data_signals_li Module
//

module read_data_signals_li #(
  parameter ARBITER_SIZE = 1,
  parameter DATA_TYPE   = 32
) (
  input                                          rst,
  input                                          clk,
  input      [             ARBITER_SIZE - 1 : 0] sel,
  input                                          pValid,
  input      [               DATA_TYPE - 1 : 0] read_data,
  output reg [ARBITER_SIZE * DATA_TYPE - 1 : 0] out_data,
  output reg [             ARBITER_SIZE - 1 : 0] valid,
  input      [             ARBITER_SIZE - 1 : 0] nReady
);
  integer                                      i;
  initial valid = 0;

  always @(posedge clk) begin
    if (rst) begin
      valid    <= 0;
    end else begin
      valid <= 0;
      for (i = 0; i <= ARBITER_SIZE - 1; i = i + 1)
        if (sel[i] && pValid) begin 
          valid[i] <= 1;
          out_data[i*DATA_TYPE+:DATA_TYPE] = read_data;
        end
    end
  end


endmodule


// burst_manager_li Module

module burst_manager_li #(
  parameter ARBITER_SIZE = 2,
  parameter BURST_TYPE = 16
) (
  input                         clk,
  input                         rst,
  input [ARBITER_SIZE * BURST_TYPE - 1 :0]     burst_length_in,
  input [ARBITER_SIZE - 1 :0]   pValid_burst,
  input                         pValid_data_from_memory,
  input                         any_valid_request,
  input [ARBITER_SIZE - 1 :0]   tmp_priorityOut,
  output reg [ARBITER_SIZE - 1 :0] priorityOut,
  output reg                    active_transaction,
  output reg                    new_transaction,
  output reg [BURST_TYPE - 1 :0] burst_length_value,
  output reg [ARBITER_SIZE - 1 : 0] burst_length_ready
);

  reg [BURST_TYPE - 1 : 0] burst_count;
  integer i;

  always @(posedge clk) begin
    if (rst) begin
      burst_count          <= 0;
      active_transaction   <= 0;
      priorityOut          <= 0;
      burst_length_value   <= 0;
      burst_length_ready   <= 0;
      new_transaction      <= 0;
    end else begin
      burst_length_ready   <= 0;
      new_transaction      <= 0;
      if (!active_transaction && any_valid_request) begin
        // Start new burst
        for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
          if (pValid_burst[i] && tmp_priorityOut[i]) begin
            burst_count <= burst_length_in[i*BURST_TYPE +: BURST_TYPE];
            burst_length_value <= burst_length_in[i*BURST_TYPE +: BURST_TYPE];
            priorityOut      <= tmp_priorityOut;
            active_transaction <= 1;
            burst_length_ready[i] <= 1;
            new_transaction  <= 1;
          end
        end
      end
      else if (active_transaction && pValid_data_from_memory) begin
        // End of burst
        if (burst_count <= 1) begin
          active_transaction <= 0;
          priorityOut <= 0;
        end
        // Decrement burst counter
        if (burst_count > 0)
          burst_count <= burst_count - 1;
      end
    end
  end

endmodule

//
// read_li_memory_arbiter Module
//

module read_li_memory_arbiter #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_TYPE   = 32,
  parameter DATA_TYPE   = 32
) (
  input                                      clk,
  input                                      rst,
  //// Interface to previous
  // address
  input  [             ARBITER_SIZE - 1 : 0] pValid,
  output [             ARBITER_SIZE - 1 : 0] ready,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] address_in,
  //// Interface to next
  input  [             ARBITER_SIZE - 1 : 0] nReady,
  output [             ARBITER_SIZE - 1 : 0] valid,
  output [ARBITER_SIZE * DATA_TYPE - 1 : 0] data_out,
  //// Interface to memory
  // Data from memory
  output                                     ready_data,
  input  [               DATA_TYPE - 1 : 0] data_from_memory,
  input                                      pValid_data,
  // Address to memory
  input                                     nReady_address,
  output                                     valid_address,
  output [               ADDR_TYPE - 1 : 0] read_address
);

  wire [ARBITER_SIZE - 1 : 0] priorityOut;
  wire [ARBITER_SIZE - 1 : 0] all_nReady;

  assign all_nReady = nReady & {ARBITER_SIZE{nReady_address}};

  // Instance of read_priority
  read_priority #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) prio (
    .req         (pValid),
    .data_ready  (all_nReady),
    .priority_out(priorityOut)
  );

  // Instance of read_address_mux
  read_address_mux #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) addressing (
    .sel     (priorityOut),
    .addr_in (address_in),
    .addr_out(read_address)
  );

  // Instance of read_address_ready
  read_address_ready #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) adderessReady (
    .sel   (priorityOut),
    .nReady(all_nReady),
    .ready (ready)
  );

  // Instance of read_data_signals_li
  read_data_signals_li #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .DATA_TYPE  (DATA_TYPE)
  ) data_signals_inst (
    .rst      (rst),
    .clk      (clk),
    .sel      (priorityOut),
    .pValid   (pValid_data),
    .read_data(data_from_memory),
    .out_data (data_out),
    .valid    (valid),
    .nReady   (all_nReady)
  );

  assign valid_address = |priorityOut;
  assign ready_data    = |(nReady & priorityOut);

endmodule

//
//  write_li_memory_arbiter Module
//

module write_li_memory_arbiter #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_TYPE   = 32,
  parameter DATA_TYPE   = 32
) (
  input                                      rst,
  input                                      clk,
  // Interface to previous
  input  [             ARBITER_SIZE - 1 : 0] pValid,
  output [             ARBITER_SIZE - 1 : 0] ready,
  input  [ADDR_TYPE * ARBITER_SIZE - 1 : 0] address_in,
  input  [DATA_TYPE * ARBITER_SIZE - 1 : 0] data_in,
  // Interface to memory   
  input    nReady_address,
  input   nReady_data,
  output  valid_address,
  output  valid_data,
  output [               ADDR_TYPE - 1 : 0] write_address,
  output [               DATA_TYPE - 1 : 0] data_to_memory
);
  wire [ARBITER_SIZE - 1 : 0] priorityOut;
  wire [ARBITER_SIZE - 1 : 0] nReady;
  wire [ARBITER_SIZE - 1 : 0] valid;

  // Priority handling
  write_priority #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) prio (
    .req         (pValid),
    .data_ready  (nReady),
    .priority_out(priorityOut)
  );

  // Address multiplexing
  write_address_mux #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) addressing (
    .sel     (priorityOut),
    .addr_in (address_in),
    .addr_out(write_address)
  );

  // Ready signal handling
  write_address_ready #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) addressReady (
    .sel   (priorityOut),
    .nReady(nReady),
    .ready (ready)
  );

  // Data handling
  write_data_signals #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .DATA_TYPE  (DATA_TYPE)
  ) data_signals_inst (
    .rst       (rst),
    .clk       (clk),
    .sel       (priorityOut),
    .in_data   (data_in),
    .write_data(data_to_memory),
    .valid     (valid)
  );

  assign valid_address = |valid;
  assign valid_data    = |valid;
  assign nReady = {ARBITER_SIZE{nReady_data & nReady_address}};

endmodule


//  read burst arbiter


module read_li_memory_arbiter_burst #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_TYPE   = 32,
  parameter DATA_TYPE   = 32,
  parameter BURST_TYPE   = 16
) (
  input                                      clk,
  input                                      rst,
  //// Interface to previous
  // address
  input  [             ARBITER_SIZE - 1 : 0] pValid,
  output [             ARBITER_SIZE - 1 : 0] ready,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] address_in,
  // burst length
  input  [             ARBITER_SIZE - 1 : 0] pValid_burst,
  output [             ARBITER_SIZE - 1 : 0] ready_burst,
  input  [ARBITER_SIZE * BURST_TYPE - 1 : 0] burst_length_in,
  //// Interface to next
  input  [             ARBITER_SIZE - 1 : 0] nReady,
  output [             ARBITER_SIZE - 1 : 0] valid,
  output [ARBITER_SIZE * DATA_TYPE - 1 : 0] data_out,
  //// Interface to memory
  // Data from memory
  output                                     ready_data,
  input  [               DATA_TYPE - 1 : 0] data_from_memory,
  input                                      pValid_data,
  // Address to memory
  input                                     nReady_address,
  output                                     valid_address,
  output [               ADDR_TYPE - 1 : 0] read_address,
  output [               BURST_TYPE - 1 : 0] read_burst_length
);

  wire [ARBITER_SIZE - 1 : 0] tmp_priorityOut;
  wire [ARBITER_SIZE - 1 : 0] priorityOut;
  wire [ARBITER_SIZE - 1 : 0] all_nReady;
  wire  active_transaction;
  wire tmp_valid;
  wire any_valid_request;

  assign all_nReady = nReady & {ARBITER_SIZE{nReady_address}};

  // Instance of read_priority
  read_priority #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) prio (
    .req         (pValid),
    .data_ready  (all_nReady),
    .priority_out(tmp_priorityOut)
  );

  // Instance of read_address_mux
  read_address_mux #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) addressing (
    .sel     (priorityOut),
    .addr_in (address_in),
    .addr_out(read_address)
  );

  // Instance of read_address_burst_ready
  read_address_burst_ready #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) addressReady (
    .sel   (tmp_priorityOut),
    .nReady(all_nReady),
    .new_transaction(valid_address),
    .ready (ready)
  );

  // Instance of read_data_signals_li
  read_data_signals_li #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .DATA_TYPE  (DATA_TYPE)
  ) data_signals_inst (
    .rst      (rst),
    .clk      (clk),
    .sel      (priorityOut),
    .pValid   (pValid_data),
    .read_data(data_from_memory),
    .out_data (data_out),
    .valid    (valid),
    .nReady   (all_nReady)
  );

  // Instance of burst management
  burst_manager_li #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .BURST_TYPE (BURST_TYPE)
  ) burst_mgr (
    .clk                 (clk),
    .rst                 (rst),
    .burst_length_in     (burst_length_in),
    .pValid_burst       (pValid_burst),
    .pValid_data_from_memory(pValid_data),
    .any_valid_request   (any_valid_request),
    .tmp_priorityOut     (tmp_priorityOut),
    .priorityOut         (priorityOut),
    .active_transaction  (active_transaction),
    .new_transaction     (valid_address),
    .burst_length_value  (read_burst_length),
    .burst_length_ready  (ready_burst)
  );

  // Read to get data from memory only when there is an active transaction
  assign ready_data = active_transaction;
  assign any_valid_request = |tmp_priorityOut;

endmodule
