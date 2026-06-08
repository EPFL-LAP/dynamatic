def generate_mc_support(name, params):
    read_address_mux_name = name + "_read_address_mux"
    read_address_ready_name = name + "_read_address_ready"
    read_data_signals_name = name + "_read_data_signals"
    read_priority_name = name + "_read_priority"
    read_memory_arbiter_name = name + "_read_memory_arbiter"

    write_address_mux_name = name + "_write_address_mux"
    write_address_ready_name = name + "_write_address_ready"
    write_data_signals_name = name + "_write_data_signals"
    write_priority_name = name + "_write_priority"
    write_memory_arbiter_name = name + "_write_memory_arbiter"

    mc_control_name = name + "_mc_control"

    return f"""
// Module of mc_support

module {read_address_mux_name} #(
  parameter ARBITER_SIZE = 1,
  parameter ADDR_TYPE   = 32
) (
  input  [             ARBITER_SIZE - 1 : 0] sel,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] addr_in,
  output [               ADDR_TYPE - 1 : 0] addr_out
);
  integer                      i;
  reg     [ADDR_TYPE - 1 : 0] addr_out_var;

  always @(*) begin
    addr_out_var = {{ADDR_TYPE{{1'b0}}}};

    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      if (sel[i]) begin
        addr_out_var = addr_in[i*ADDR_TYPE+:ADDR_TYPE];
      end
    end
  end

  assign addr_out = addr_out_var;

endmodule

//
//  read_address_ready Module
//

module {read_address_ready_name} #(
  parameter ARBITER_SIZE = 1
) (
  input  [ARBITER_SIZE-1:0] sel,
  input  [ARBITER_SIZE-1:0] nReady,
  output [ARBITER_SIZE-1:0] ready
);
  assign ready = nReady & sel;

endmodule

//
//  read_data_signals Module
//

module {read_data_signals_name} #(
  parameter ARBITER_SIZE = 1,
  parameter DATA_TYPE   = 32
) (
  input                                          rst,
  input                                          clk,
  input      [             ARBITER_SIZE - 1 : 0] sel,
  input      [               DATA_TYPE - 1 : 0] read_data,
  output reg [ARBITER_SIZE * DATA_TYPE - 1 : 0] out_data,
  output reg [             ARBITER_SIZE - 1 : 0] valid,
  input      [             ARBITER_SIZE - 1 : 0] nReady
);
  reg     [              ARBITER_SIZE - 1 : 0] sel_prev = 0;
  reg     [ARBITER_SIZE * DATA_TYPE  - 1 : 0] out_reg = 0;

  integer                                      i;
  initial valid = 0;

  always @(posedge clk) begin
    if (rst) begin
      valid    <= 0;
      sel_prev <= 0;
    end else begin
      sel_prev <= sel;
      for (i = 0; i <= ARBITER_SIZE - 1; i = i + 1)
      if (sel[i]) valid[i] <= 1;
      else if (nReady[i]) valid[i] <= 0;
    end
  end

  always @(posedge clk) begin
    for (i = 0; i <= ARBITER_SIZE - 1; i = i + 1)
      if (rst)
        out_reg[i*DATA_TYPE+:DATA_TYPE] <= 0;
      else if (sel_prev[i])
        out_reg[i*DATA_TYPE+:DATA_TYPE] <= read_data;
    end

  always @(*) begin
    for (i = 0; i <= ARBITER_SIZE - 1; i = i + 1) begin
      if (sel_prev[i]) out_data[i*DATA_TYPE+:DATA_TYPE] = read_data;
      else out_data[i*DATA_TYPE+:DATA_TYPE] = out_reg[i*DATA_TYPE+:DATA_TYPE];
    end
  end

endmodule

//
//  read_priority Module
//

module {read_priority_name} #(
  parameter ARBITER_SIZE = 1
) (
  input      [ARBITER_SIZE - 1 : 0] req,
  input      [ARBITER_SIZE - 1 : 0] data_ready,
  output reg [ARBITER_SIZE - 1 : 0] priority_out
);
  reg     prio_req = 0;
  integer i;

  always @(req, data_ready) begin
    priority_out[0] = req[0] & data_ready[0];
    prio_req        = req[0] & data_ready[0];

    for (i = 1; i <= ARBITER_SIZE - 1; i = i + 1) begin
      priority_out[i] = ~prio_req & req[i] & data_ready[i];
      prio_req        = prio_req | (req[i] & data_ready[i]);
    end
  end
endmodule

module {read_memory_arbiter_name} #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_TYPE   = 32,
  parameter DATA_TYPE   = 32
) (
  input                                      clk,
  input                                      rst,
  // Interface to previous
  input  [             ARBITER_SIZE - 1 : 0] pValid,
  output [             ARBITER_SIZE - 1 : 0] ready,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] address_in,
  // Interface to next
  input  [             ARBITER_SIZE - 1 : 0] nReady,
  output [             ARBITER_SIZE - 1 : 0] valid,
  output [ARBITER_SIZE * DATA_TYPE - 1 : 0] data_out,
  // Interface to memory
  output                                     read_enable,
  output [               ADDR_TYPE - 1 : 0] read_address,
  input  [               DATA_TYPE - 1 : 0] data_from_memory
);

  wire [ARBITER_SIZE - 1 : 0] priorityOut;

  // Instance of read_priority
  {read_priority_name} #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) prio (
    .req         (pValid),
    .data_ready  (nReady),
    .priority_out(priorityOut)
  );

  // Instance of read_address_mux
  {read_address_mux_name} #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) addressing (
    .sel     (priorityOut),
    .addr_in (address_in),
    .addr_out(read_address)
  );

  // Instance of read_address_ready
  {read_address_ready_name} #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) adderssReady (
    .sel   (priorityOut),
    .nReady(nReady),
    .ready (ready)
  );

  // Instance of read_data_signals
  {read_data_signals_name} #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .DATA_TYPE  (DATA_TYPE)
  ) data_signals_inst (
    .rst      (rst),
    .clk      (clk),
    .sel      (priorityOut),
    .read_data(data_from_memory),
    .out_data (data_out),
    .valid    (valid),
    .nReady   (nReady)
  );

  assign read_enable = |priorityOut;

endmodule


//
//  write_address_mux Module
//

module {write_address_mux_name} #(
  parameter ARBITER_SIZE = 1,
  parameter ADDR_TYPE   = 32
) (
  input  [             ARBITER_SIZE - 1 : 0] sel,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] addr_in,
  output [               ADDR_TYPE - 1 : 0] addr_out
);
  integer                      i;
  reg     [ADDR_TYPE - 1 : 0] addr_out_var;

  always @(*) begin
    //! May need to chaneg the following line to remove the assignment to 0
    addr_out_var = {{ADDR_TYPE{{1'b0}}}};

    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      if (sel[i]) begin
        addr_out_var = addr_in[i*ADDR_TYPE+:ADDR_TYPE];
      end
    end
  end

  assign addr_out = addr_out_var;

endmodule

//
//  write_address_ready Module
//

module {write_address_ready_name} #(
  parameter ARBITER_SIZE = 1
) (
  input  [ARBITER_SIZE-1:0] sel,
  input  [ARBITER_SIZE-1:0] nReady,
  output [ARBITER_SIZE-1:0] ready
);
  assign ready = nReady & sel;

endmodule

//
//  write_data_signals Module
//

module {write_data_signals_name} #(
  parameter ARBITER_SIZE = 1,
  parameter DATA_TYPE   = 32
) (
  input                                           clk,
  input                                           rst,
  input      [              ARBITER_SIZE - 1 : 0] sel,
  output     [                DATA_TYPE - 1 : 0] write_data,
  input      [ARBITER_SIZE * DATA_TYPE  - 1 : 0] in_data,
  output     [              ARBITER_SIZE - 1 : 0] valid
);
  integer                      i;
  reg     [DATA_TYPE - 1 : 0] data_out_var = 0;
  reg     [ARBITER_SIZE - 1: 0] valid_out_var = 0;

  always @(*) begin
    data_out_var = {{DATA_TYPE{{1'b0}}}};
    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      if (sel[i]) begin
        data_out_var = in_data[i*DATA_TYPE+:DATA_TYPE];
      end
    end
  end

  assign write_data = data_out_var;

  always @(posedge clk) begin
    if (rst) begin
      valid_out_var <= 0;
    end else begin
      valid_out_var <= sel;
    end
  end

  assign valid = valid_out_var;

endmodule

//
//  write_priority Module
//

module {write_priority_name} #(
  parameter ARBITER_SIZE = 1
) (
  input      [ARBITER_SIZE - 1 : 0] req,
  input      [ARBITER_SIZE - 1 : 0] data_ready,
  output reg [ARBITER_SIZE - 1 : 0] priority_out
);
  reg     prio_req = 0;
  integer i;

  always @(req, data_ready) begin
    priority_out[0] = req[0] & data_ready[0];
    prio_req        = req[0] & data_ready[0];

    for (i = 1; i <= ARBITER_SIZE - 1; i = i + 1) begin
      priority_out[i] = ~prio_req & req[i] & data_ready[i];
      prio_req        = prio_req | (req[i] & data_ready[i]);
    end
  end

endmodule

//
//  write_memory_arbiter Module
//

module {write_memory_arbiter_name} #(
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
  // Interface to next    
  input  [             ARBITER_SIZE - 1 : 0] nReady,
  output [             ARBITER_SIZE - 1 : 0] valid,
  // Interface to memory   
  output                                     write_enable,
  output                                     enable,
  output [               ADDR_TYPE - 1 : 0] write_address,
  output [               DATA_TYPE - 1 : 0] data_to_memory
);
  wire [ARBITER_SIZE - 1 : 0] priorityOut;

  // Priority handling
  {write_priority_name} #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) prio (
    .req         (pValid),
    .data_ready  (nReady),
    .priority_out(priorityOut)
  );

  // Address multiplexing
  {write_address_mux_name} #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) addressing (
    .sel     (priorityOut),
    .addr_in (address_in),
    .addr_out(write_address)
  );

  // Ready signal handling
  {write_address_ready_name} #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) addressReady (
    .sel   (priorityOut),
    .nReady(nReady),
    .ready (ready)
  );

  // Data handling
  {write_data_signals_name} #(
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

  assign write_enable = |priorityOut;
  assign enable       = |priorityOut;

endmodule

//
//  mc_control
//

module {mc_control_name} (
  input  wire clk,
  input  wire rst,

  // start input control
  input  wire memStart_valid,
  output wire memStart_ready,

  // end output control
  output wire memEnd_valid,
  input  wire memEnd_ready,

  // "no more requests" input control
  input  wire ctrlEnd_valid,
  output wire ctrlEnd_ready,

  // all requests completed
  input  wire allRequestsDone
);

  // FSM states
  localparam IDLE    = 1'b0;
  localparam RUNNING = 1'b1;

  reg fsm_q;
  reg no_more_requests_q;

  wire fsm_running;
  wire function_return;

  // FSM running flag
  assign fsm_running = (fsm_q == RUNNING);

  // Function return condition
  // 1. No more requests
  // 2. All requests done
  // 3. MemEnd ready
  // 4. FSM is running
  assign function_return =
      no_more_requests_q &
      allRequestsDone &
      memEnd_ready &
      fsm_running;

  // Ready to start only when IDLE
  assign memStart_ready = (fsm_q == IDLE);

  // Memory end is valid when returning conditions met (except ready)
  assign memEnd_valid =
      no_more_requests_q &
      allRequestsDone &
      fsm_running;

  // Accept ctrlEnd only once per execution
  assign ctrlEnd_ready = ~no_more_requests_q;

  // FSM state register
  always @(posedge clk) begin
    if (rst) begin
      fsm_q <= IDLE;
    end else begin
      if (fsm_q == IDLE) begin
        if (memStart_valid) begin
          fsm_q <= RUNNING;
        end
      end else if (function_return) begin
        fsm_q <= IDLE;
      end
    end
  end

  // no_more_requests register
  always @(posedge clk) begin
    if (rst) begin
      no_more_requests_q <= 1'b0;
    end else begin
      if (function_return) begin
        no_more_requests_q <= 1'b0;
      end else if (ctrlEnd_valid) begin
        no_more_requests_q <= 1'b1;
      end
    end
  end

endmodule
"""
