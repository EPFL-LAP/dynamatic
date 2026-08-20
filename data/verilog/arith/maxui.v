module maxui #(
    parameter DATA_TYPE = 32
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire [DATA_TYPE-1:0]     lhs,
    input  wire                     lhs_valid,
    input  wire [DATA_TYPE-1:0]     rhs,
    input  wire                     rhs_valid,
    input  wire                     result_ready,

    output wire [DATA_TYPE-1:0]     result,
    output wire                     result_valid,
    output wire                     lhs_ready,
    output wire                     rhs_ready
);
    assign result = ($unsigned(lhs) > $unsigned(rhs)) ? lhs : rhs;

    assign result_valid = lhs_valid & rhs_valid;
    assign lhs_ready     = result_ready & rhs_valid;
    assign rhs_ready     = result_ready & lhs_valid;
endmodule
