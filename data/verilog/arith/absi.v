module absi #(
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

    assign outs = ins[DATA_TYPE-1] ? (~ins + 1'b1) : ins;

    // Handshake (purely combinational)
    assign outs_valid = ins_valid;
    assign ins_ready  = outs_ready;

endmodule
