// one_slot_break_dvr : one_slot_break_dvr({'data_type': 32})

`timescale 1ns/1ps

`timescale 1ns/1ps
// dataless one_slot_break_dvr Module
module one_slot_break_dvr_dataless (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);

  // Define internal signals
  reg outputValid = 0;
  reg inputReady = 1;
  wire enable;
  wire stop;

  // Sequential logic for inputReady
  always @(posedge clk) begin
    if (rst) begin
      inputReady <= 1;
    end else begin
      inputReady <= (~stop) & (~enable);
    end
  end

  // Sequential logic for outputValid
  always @(posedge clk) begin
    if (rst) begin
      outputValid <= 0;
    end else begin
      outputValid <= enable | stop;
    end
  end

  // Combinational logic
  assign enable = ins_valid & inputReady;
  assign stop   = outputValid & (~outs_ready);
  // Output assignments
  assign ins_ready  = inputReady;
  assign outs_valid = outputValid;

endmodule

// one_slot_break_dvr Module
module one_slot_break_dvr #(
  parameter DATA_TYPE = 32
) (
  input  clk,
  input  rst,
  // Input channel
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  wire enable, inputReady;
  reg [DATA_TYPE - 1 : 0] dataReg = 0;
  
  // Instance of one_slot_break_dvr_dataless to manage handshaking
  one_slot_break_dvr_dataless control (
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (ins_valid ),
    .ins_ready  (inputReady),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  always @(posedge clk) begin
    if (rst) begin
      dataReg <= {DATA_TYPE{1'b0}};
    end else if (enable) begin
      dataReg <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign enable    = ins_valid & inputReady;
  assign outs      = dataReg;

endmodule

