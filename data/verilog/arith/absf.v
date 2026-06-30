module absf #(
    parameter DATA_TYPE = 32
)(
    // inputs
    input  wire                     clk,
    input  wire                     rst,
    input  wire [DATA_TYPE-1:0]     ins,
    input  wire                     ins_valid,
    input  wire                     outs_ready,

    // outputs
    output wire [DATA_TYPE-1:0]     outs,
    output wire                     outs_valid,
    output wire                     ins_ready
);

    // Absolute value = clear sign bit
    assign outs = {1'b0, ins[DATA_TYPE-2:0]};

    // Handshake (purely combinational)
    assign outs_valid = ins_valid;
    assign ins_ready  = outs_ready;

endmodule
